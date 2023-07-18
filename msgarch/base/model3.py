import numpy as np
import pandas as pd
from scipy.optimize import minimize
import yfinance as yf
from scipy.stats import norm

from statsmodels.tsa.regime_switching.markov_switching import MarkovSwitching


class Model:

    def __init__(self, endog: np.ndarray):

        self.endog = endog
        if isinstance(endog, pd.DataFrame):
            self.data = endog.values
        if isinstance(endog, (list, tuple)):
            self.endog = np.asarray(endog)

        if self.endog is None:
            raise RuntimeError("Cannot fit a model without endog parameter")

        self.num_obs = len(endog)

    def log_likelihood(self, *args, **kwargs):
        """

        Parameters
        ----------
        params: ndarray
            Parameters of the log likelihood function

        Returns
        -------
        ll : float
            value of the log likelihood
        """
        raise NotImplementedError

    def hessian(self, params):
        raise NotImplementedError

    def fit(self):
        raise NotImplementedError


class LikelihoodModel(Model):

    def fit(self, *args, **kwargs):
        """
        Fit a model to data.
        """
        if not hasattr(self, "log_likelihood"):
            raise RuntimeError("Cannot fit a likelihood model without self.log_likelihood function implemented")


class MSGarch(LikelihoodModel):

    def __init__(self, endog: np.ndarray, k_regimes: int = None, **kwargs):
        if k_regimes is None:
            self.k_regimes = 2
        else:
            self.k_regimes = k_regimes

        self.error_type = kwargs.get("error_type")
        self.mu = np.zeros(k_regimes)
        self.phi = np.zeros(k_regimes)
        self.omega = np.zeros(k_regimes)
        self.alpha = np.zeros((k_regimes, 1))
        self.beta = np.zeros((k_regimes, 1))
        self.p = np.zeros((k_regimes, k_regimes))
        self.params = None

        super().__init__(endog=endog)

    def log_likelihood(self, params):
        print(params)
        T = len(self.endog)
        # H = np.zeros((T, self.k_regimes))
        filtered_prob = np.zeros((T + 1, self.k_regimes))
        predict_prob = np.zeros((T, self.k_regimes))
        # loglik = np.zeros((T, 1))
        mu = params[:self.k_regimes]
        omega = params[self.k_regimes:2 * self.k_regimes]
        alpha = params[2 * self.k_regimes:3 * self.k_regimes]
        beta = params[3 * self.k_regimes:4 * self.k_regimes]
        tmp_gamma = params[4 * self.k_regimes:4 * self.k_regimes + (self.k_regimes * self.k_regimes)]

        gamma = np.zeros((self.k_regimes, self.k_regimes))
        # gamma = gamma.reshape(self.k_regimes, self.k_regimes)  # doit etre vecteur colonne de taille 6 pour JD
        gamma[~np.eye(*gamma.shape, dtype=bool)] = tmp_gamma
        # #défnir la matrice P à partir des gamma. Regarder les notes. Réfléchir aux indices.
        # Matrice P qui contient les proba de transition et qui est écrit en fonction de gamma
        P = np.zeros((self.k_regimes, self.k_regimes))
        for i in range(0, self.k_regimes):
            for j in range(0, self.k_regimes):
                if i != j:
                    P[i][j] = np.exp(gamma[i][j]) / (1 + np.sum(np.exp(gamma[i])) - np.exp(gamma.diagonal()[i]))
                    # P[np.diag_indices_from(P)] = 1 / (1 + np.sum(np.exp(gamma[i])) - np.exp(gamma.diagonal()[i]))
        np.fill_diagonal(P, 1 - P.sum(axis=1))

        A = np.vstack((np.eye(self.k_regimes) - P, np.ones((1, self.k_regimes))))
        I3 = np.eye(self.k_regimes + 1)
        c3 = I3[:, self.k_regimes]
        pinf = np.linalg.solve(A.T @ A, A.T @ c3)
        filtered_prob[:2, :] = np.tile(pinf.T, (2, 1))

        eps = np.zeros((T, self.k_regimes))
        # Data and sigma2 are T by 1 vectors
        sigma2 = np.zeros((T, self.k_regimes))
        sigma2[0, :] = self.endog.var()
        loglik = np.zeros((T, 1))
        for t in range(1, T):
            eps[t - 1, :] = self.endog[t - 1] - mu
            sigma2[t, :] = filtered_prob[t, :] @ (omega + alpha * eps[t - 1, :] ** 2 + beta * sigma2[t - 1])
            LL = 0.5 * (np.log(2 * np.pi) + np.log(sigma2[t, :]) + eps[t, :] ** 2 / sigma2[t, :])
            predict_prob[t, :] = LL / np.sum(LL)
            filtered_prob[t + 1, :] = predict_prob[t, :] @ P
            loglik[t] = LL.sum()

        return -sum(loglik)/ 10000000

    def fit(self):

        scaling_omega = 0  # different de 0!!  # scaling omega = gamma sur note JD
        scaling_alpha = 0.1
        scaling_beta = 0.8  # WARNING: la somme de scaling_alpha et scaling_beta <1

        # Réfléchir à cette boucle
        compo = list()
        for i in range(0, self.k_regimes):
            compo.append(np.exp(scaling_omega * ((self.k_regimes / 2) - i)))

        vec = np.array(compo)  # note Jean David page 3. C'est un vecteur de taille k de nombre de regime de marché
        omega_init = np.var(self.endog) * (1 - scaling_alpha - scaling_beta) * vec

        alpha_init = scaling_alpha * np.ones(self.k_regimes)
        beta_init = scaling_beta * np.ones(self.k_regimes)

        mu_init = np.ones(self.k_regimes) * np.mean(self.endog)
        mu_init[0] = min(self.endog)*mu_init[0]
        mu_init[1] = self.endog.mean() ** 2
        mu_init[2] = max(self.endog) * mu_init[0]

        # mu_init[0] = -0.01
        # mu_init[1] = 0
        # mu_init[2] = 0.01

        # prob_reshape = matrix_prob.reshape((self.k_regimes * self.k_regimes))

        matrix_init = -2 * np.ones((self.k_regimes, self.k_regimes))

        matrix_prob = np.ones((self.k_regimes, self.k_regimes))
        for i in range(0, self.k_regimes):
            for j in range(0, self.k_regimes):
                matrix_prob[i][j] = np.exp(matrix_init[i][j]) / (
                            1 + np.sum(np.exp(matrix_init[i])) - np.exp(matrix_init.diagonal()[i]))
                matrix_prob[np.diag_indices_from(matrix_prob)] = 1 / (
                            1 + np.sum(np.exp(matrix_init[i])) - np.exp(matrix_init.diagonal()[i]))

        for i in range(0, self.k_regimes):
            try:
                np.sum(matrix_prob[i]) == 1
            except ValueError:
                print(f"Sum of the line {i} of the matrix =! 1")

        prob_reshape = matrix_prob[~np.eye(*matrix_prob.shape, dtype=bool)]
        params_init = np.concatenate(
            (mu_init, omega_init, alpha_init, beta_init, prob_reshape))  # rajouter les mu

        bounds = []

        bounds += [(-1, 1) for _ in range(len(mu_init))]
        bounds += [(-1, 1) for _ in range(len(omega_init))]
        bounds += [(0, 1) for _ in range(len(alpha_init))]
        bounds += [(0, 1) for _ in range(len(beta_init))]
        bounds += [(0, 1) for _ in range(len(prob_reshape))]

        bounds = tuple(bounds)  # tuple((-10, 10) for _ in range(params_init.shape[0]))

        mu1_mu2_constraint = {'type': 'ineq',
                              'fun': lambda params: params[1] - params[0]}

        mu2_mu3_constraint = {'type': 'ineq',
                              'fun': lambda params: params[2] - params[1]}  # positif ?

        # mu 1 < mu2 < mu3.

        ab1_constraint = {'type': 'ineq',
                          'fun': lambda params: 1 - params[7] - params[10]}

        ab2_constraint = {'type': 'ineq',
                          'fun': lambda params: 1 - params[8] - params[11]}

        ab3_constraint = {'type': 'ineq',
                          'fun': lambda params: 1 - params[9] - params[12]}

        res = minimize(self.log_likelihood, params_init, method="SLSQP",
                       constraints=(mu1_mu2_constraint, mu2_mu3_constraint,
                                    ab1_constraint, ab2_constraint, ab3_constraint),
                       bounds=bounds,
                       options={"disp": True}
                       )
        print(res.x)



        return res

    def predict(self, start=None, end=None, dynamic=False):
        if start is None:
            start = 0
        if end is None:
            end = len(self.endog) - 1

        mu = self.params[:self.k_regimes]
        print(mu)

        omega = self.params[self.k_regimes * 2:self.k_regimes * 3]
        alpha = self.params[self.k_regimes * 3:self.k_regimes * 4]
        beta = self.params[self.k_regimes * 4:self.k_regimes * 5]
        p = self.params[self.k_regimes * 5:].reshape((self.k_regimes, self.k_regimes))

        T = len(self.endog)
        mu_pred = np.zeros(T)
        sigma_pred = np.zeros(T)

        z = self.filtered_prob()

        for t in range(start, end + 1):
            mu_pred[t] = np.dot(z[:, t], mu)
            sigma_pred[t] = np.sqrt(np.dot(z[:, t], alpha * self.endog[t - 1] ** 2 + beta * omega))

            if dynamic and t < end:
                z_next = np.zeros(self.k_regimes)
                for k in range(self.k_regimes):
                    for j in range(self.k_regimes):
                        z_next[k] += z[j, t] * p[j, k] * np.exp(
                            -0.5 * (self.endog[t + 1] - mu[k]) ** 2 / (
                                (alpha[k] * self.endog[t] ** 2 + beta[k] * omega[k])))
                z_next /= np.sum(z_next)
                z[:, t + 1] = z_next

        return mu_pred[start:end + 1], sigma_pred[start:end + 1]


if __name__ == '__main__':
    btc_usd = yf.Ticker("BTC-USD")
    history = btc_usd.history(period="3Y")
    log_returns = np.diff(np.log(history["Close"]))
    ms_garch = MSGarch(log_returns, 3)
    ms_garch.fit()
    # ms_garch.predict()
