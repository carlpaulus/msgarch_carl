import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
from scipy.stats import norm


def norm_pdf(x, mean=0, std_dev=1):
    return (1 / (np.sqrt(2 * np.pi * std_dev ** 2))) * np.exp(-(x - mean) ** 2 / (2 * std_dev ** 2))


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

    def fit(self):
        pass


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

    def _prepare(self):
        pass

    def log_likelihood(self, params):
        T = len(self.endog)
        H = np.zeros((T, self.k_regimes))
        mu = params[:self.k_regimes]
        omega = params[self.k_regimes:2 * self.k_regimes]
        alpha = params[2 * self.k_regimes:3 * self.k_regimes]
        beta = params[3 * self.k_regimes:4 * self.k_regimes]
        tmp_gamma = params[4 * self.k_regimes:4 * self.k_regimes + (self.k_regimes * self.k_regimes)]

        gamma = np.zeros((self.k_regimes, self.k_regimes))
        # gamma = gamma.reshape(self.k_regimes, self.k_regimes)  # doit etre vecteur colonne de taille 6 pour JD
        gamma[~np.eye(*gamma.shape, dtype=bool)] = tmp_gamma

        P = np.zeros((self.k_regimes, self.k_regimes))
        for i in range(0, self.k_regimes):
            for j in range(0, self.k_regimes):
                if i != j:
                    P[i][j] = np.exp(gamma[i][j]) / (1 + np.sum(np.exp(gamma[i])) - np.exp(gamma.diagonal()[i]))
                    # P[np.diag_indices_from(P)] = 1 / (1 + np.sum(np.exp(gamma[i])) - np.exp(gamma.diagonal()[i]))
        np.fill_diagonal(P, 1 - P.sum(axis=1))

        filtered_prob = np.zeros((T + 1, self.k_regimes))
        predict_prob = np.zeros((T, self.k_regimes))
        loglik = np.zeros((T, 1))

        A = np.vstack((np.eye(self.k_regimes) - P, np.ones(self.k_regimes)))
        I3 = np.eye(self.k_regimes + 1)
        c3 = I3[:, -1]
        # pinf = np.linalg.lstsq(A.T @ A, A.T @ c3, rcond=None)[0]
        pinf = np.ones(self.k_regimes) / self.k_regimes
        filtered_prob[:2] = np.repeat(pinf[np.newaxis, :], 2, axis=0)
        Ht = np.zeros(T)
        H[0,:]=(self.endog[0]-mu)**2
        Ht[0]= pinf @ H[0, :]  # Note *10
        for t in range(1, T):
            H[t, :] = omega + alpha * (self.endog[t - 1] - mu) ** 2 + H[t - 1, :] @ beta
            LL = filtered_prob[t, :self.k_regimes] * norm.pdf(self.endog[t] - mu, 0, np.sqrt(H[t, :self.k_regimes]))
            predict_prob[t, :] = LL / np.sum(LL)
            filtered_prob[t + 1, :] = P @ predict_prob[t, :]
            loglik[t, 0] = np.log(LL@ np.ones(self.k_regimes))
            Ht[t] = predict_prob[t, :] @ H[t, :]  # s'arrête à la 820è #Dubitatif # Quelle est l'objectif de cette ligne ?
        return -sum(loglik)

    def _extract_params(self, params):

        pass

    def fit(self):
        scaling_omega = 0  # different de 0!!  # scaling omega = gamma sur note JD
        scaling_alpha = 0.1
        scaling_beta = 0.8  # WARNING: la somme de scaling_alpha et scaling_beta <1

        # Réfléchir à cette boucle
        compo = list()
        for i in range(0, self.k_regimes):
            compo.append(np.exp(((self.k_regimes / 2) - i)))

        vec = np.array(compo)  # note Jean David page 3. C'est un vecteur de taille k de nombre de regime de marché
        omega_init = np.var(self.endog) * (1 - scaling_alpha - scaling_beta) * vec

        alpha_init = scaling_alpha * np.ones(self.k_regimes)
        beta_init = scaling_beta * np.ones(self.k_regimes)

        mu_init = np.ones(self.k_regimes) * np.mean(self.endog)
        mu_init[0] = -0.001
        mu_init[1] = 0
        mu_init[2] = 0.001

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
        bounds += [(0, 1) for _ in range(len(omega_init))]
        bounds += [(0, 0.5) for _ in range(len(alpha_init))]
        bounds += [(0.5, 0.95) for _ in range(len(beta_init))]
        bounds += [(-np.inf, np.inf) for _ in range(len(prob_reshape))]

        bounds = tuple(bounds)  # tuple((-10, 10) for _ in range(params_init.shape[0]))

        mu1_mu2_constraint = {'type': 'ineq',
                              'fun': lambda params: params[1] - params[0]}

        mu2_mu3_constraint = {'type': 'ineq',
                              'fun': lambda params: params[2] - params[1]}

        ab1_constraint = {'type': 'ineq',
                          'fun': lambda params: np.ones(self.k_regimes) - params[7] - params[10]}

        ab2_constraint = {'type': 'ineq',
                          'fun': lambda params: np.ones(self.k_regimes) - params[8] - params[11]}

        ab3_constraint = {'type': 'ineq',
                          'fun': lambda params: np.ones(self.k_regimes) - params[9] - params[12]}

        # todo: alpha + beta <=1

        res = minimize(
            self.log_likelihood, params_init,
            method="SLSQP",
            constraints=(mu1_mu2_constraint, mu2_mu3_constraint,
                         ab1_constraint, ab2_constraint, ab3_constraint),
            bounds=bounds,
            options={"disp": True}
        )
        print(res.x)
        pass


if __name__ == '__main__':
    btc_usd = yf.Ticker("XRP-USD")
    history = btc_usd.history(period="7Y")
    log_returns = np.diff(np.log(history["Close"]))  # Pas sur les pct change ?
    ms_garch = MSGarch(log_returns, 3)
    ms_garch.fit()
