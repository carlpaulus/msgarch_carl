import numpy as np
import pandas as pd
from scipy.optimize import minimize
import yfinance as yf
from scipy.stats import norm


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
        T = len(self.endog)
        H = np.zeros((T, self.k_regimes))
        filtered_prob = np.zeros((T + 1, self.k_regimes))
        predict_prob = np.zeros((T, self.k_regimes))
        loglik = np.zeros((T, 1))

        omega = params[:self.k_regimes]
        alpha = params[self.k_regimes:2 * self.k_regimes]
        beta = np.diag(params[2 * self.k_regimes:3 * self.k_regimes])
        P = params[3 * self.k_regimes:3 * self.k_regimes + (self.k_regimes * self.k_regimes)]
        P = P.reshape(self.k_regimes, self.k_regimes)

        A = np.vstack((np.eye(self.k_regimes) - P, np.ones((1, self.k_regimes))))
        I3 = np.eye(self.k_regimes + 1)
        c3 = I3[:, self.k_regimes]
        pinf = np.linalg.solve(A.T @ A, A.T @ c3)
        filtered_prob[:2, :] = np.tile(pinf.T, (2, 1))
        Ht = np.zeros((T, 1))

        for t in range(1, T):
            H[t, :] = omega + alpha * self.endog[t - 1] ** 2 + H[t - 1, :] @ beta
            LL = filtered_prob[t, :] * norm.pdf(self.endog[t], 0, np.sqrt(H[t, :]))  # proba density function
            predict_prob[t, :] = LL / np.sum(LL)
            filtered_prob[t + 1, :] = predict_prob[t, :] @ P
            loglik[t, 0] = np.log(LL.sum())
            Ht[t, 0] = predict_prob[t, :] @ H[t, :]

            llg = -loglik.sum()
            # print(llg)

        return llg

    def fit(self):
        # omega_init = np.var(self.endog) * np.ones(self.k_regimes)  #mettre vec à la place de np.ones
        scaling_omega = 0  # different de 0!!  # scaling omega = gamma sur note JD
        scaling_alpha = 0.1
        scaling_beta = 0.8  # WARNING: la somme de scaling_alpha et scaling_beta <1

        compo = list()
        for i in range(0, self.k_regimes):
            compo.append(np.exp(scaling_omega * ((self.k_regimes / 2) - i)))

        vec = np.array(compo)  # note Jean David page 3. C'est un vecteur de taille k
        omega_init = np.var(self.endog) * (1-scaling_alpha-scaling_beta) * vec

        alpha_init = scaling_alpha * np.ones(self.k_regimes)
        beta_init = scaling_beta * np.ones(self.k_regimes)

        print("omega_init = ", omega_init)

        p_init = np.ones((self.k_regimes, self.k_regimes)) / self.k_regimes
        params_init = np.concatenate(
            (omega_init, alpha_init, beta_init, p_init.reshape((self.k_regimes * self.k_regimes))))

        res = minimize(self.log_likelihood, params_init, method='Nelder-Mead', options={"xtol": 1e-7, 'disp': True})

        p = norm.cdf(res.x[self.k_regimes * 3:].reshape((self.k_regimes, self.k_regimes)))  # p = matrice de transition, Cumulative distribution function
        print("p = ", p)
        estimate_params = {
            "omega_est": res.x[:self.k_regimes],
            "alpha_est": res.x[self.k_regimes:self.k_regimes * 2],
            "beta_est": res.x[self.k_regimes * 2:self.k_regimes * 3],
            "p_est": p/p.sum(axis=1),
        }
        print("estimate_params = ", estimate_params)
        return res

    def predict(self, start=None, end=None, dynamic=False):
        if start is None:
            start = 0
        if end is None:
            end = len(self.endog) - 1

        mu = self.params[:self.k_regimes]
        print(mu)
        phi = self.params[self.k_regimes:self.k_regimes * 2]
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
                                    (alpha[k] * self.endog[t] ** 2 + beta[k] * omega[k]) * phi[j]))
                z_next /= np.sum(z_next)
                z[:, t + 1] = z_next

        return mu_pred[start:end + 1], sigma_pred[start:end + 1]


if __name__ == '__main__':
    btc_usd = yf.Ticker("BTC-USD")
    history = btc_usd.history(period="1Y")
    log_returns = np.diff(np.log(history["Close"]))
    # stdev = np.std(log_returns)
    # print(stdev**2)
    # ms_garch = MSGarch(log_returns, 3)
    # ms_garch.fit()
    # ms_garch.predict()
