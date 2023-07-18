import pandas as pd
import yfinance as yf
import numpy as np
from scipy.stats import skew, kurtosis
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller


class Statistics:

    """
    Nourrir une section sur les propriétés statistiques des cryptos-actifs i.e. indicateurs de base
    sur les rendements des pricipaux tokens (BTC, ETH, XRP, ADA) et sortir des graphs.
    """

    def __init__(self, asset, time):
        self.asset = asset
        self.time = time
        self._prices = None
        self._returns = None
        self._logreturns = None
        self._variance = None
        self._volatility = None
        self._skew = None
        self._kurt = None

    def prices(self):
        self._prices = yf.download(self.asset, period=self.time).ffill().dropna()["Close"]

    def returns(self):
        self._returns = (self._prices - self._prices.shift(1)) / self._prices  # self._prices.pct_change()

    def logreturns(self):
        self._logreturns = np.log(self._prices / self._prices.shift(1))  # np.log(self._prices.pct_change().tail(-1))

    def variance(self):
        self._variance = np.var(self._logreturns)

    def volatility(self):
        self._volatility = round(self._logreturns.std() * 252 ** 0.5, 2)
        # df = pd.concat([self._variance, self._volatility], axis=1)
        # df.columns = ["Variance", "Volatility"]
        # print(df)

    def mean_returns(self):
        """
        Moyenne des rendements sur plusieurs périodes (up, flat, down) pour justifier l'approche par
        régimes (moyenne des rendements en période de bull, bear, et flat)
        """

        print(self.asset)
        up = ("Up", "2020-09-27", "2021-03-08")
        flat = ("Flat", "2020-01-05", "2020-07-05")
        down = ("Down", "2021-11-07", "2022-06-12")

        # Building the dataframe
        log = list()
        vol = list()
        i = list()
        periods = [up, flat, down]
        for period in periods:
            prices = yf.download(self.asset, start=period[1], end=period[2]).ffill().dropna()["Close"]
            logret = np.log((prices / (prices.shift(1))))
            volat = logret.std() * 252 ** 0.5

            log.append(str(round(np.mean(logret) * 100, 2))+'%')
            vol.append(str(round(np.mean(volat) * 100))+'%')

            i.append(period[0])

        data = {'avg log': log, 'avg vol': vol, 'ETH-USD': i}
        print(pd.DataFrame.from_dict(data).set_index('ETH-USD').T)

    def skewness_kurtosis(self):
        """
        https://www.geeksforgeeks.org/how-to-calculate-skewness-and-kurtosis-in-python/#
        - coefficient d'asymmétrie (skewness)
        - coefficient d'applatissement (kurtosis)
        """
        self._skew = skew(self._logreturns.tail(-1))
        self._kurt = kurtosis(self._logreturns.tail(-1))

        print(self._skew, self._kurt)

        data = {'Skewness': self._skew, 'Kurtosis': self._kurt}
        print(pd.DataFrame.from_dict(data).set_index(self._logreturns.columns))

    def autocorr_ret(self):
        """
        - autocorrélation des rendements: corrélations entre r_t et r_{t-1}, r_t et r_{t-2},..., r_t et r_{t-p}, p=20.
        - print autocorrelation data
        - plot them and remove the first observation (i.e. 1) by zooming in
        """

        # autocorrelation data
        acorr = sm.tsa.acf(self._logreturns.tail(-1), nlags=30)
        print(acorr)

        # plot
        plt.plot(acorr)
        plot_acf(self._logreturns.tail(-1), lags=30)  # lags=20
        plt.show()

    def autocorr_abs(self):
        """
        - autocorrélation des valeurs absolues des rendements:
        corrélations entre |r_t| et |r_{t-1}|, |r_t| et |r_{t-2}|,..., |r_t| et |r_{t-p}|, p=20.
        Normalement, il doivent décroitre vers zéro, mais lentement.
        Cela prouve que les rendements ne suivent pas un processus de marche aléatoire.
        """

        # autocorrelation data
        acorr = sm.tsa.acf(abs(self._logreturns.tail(-1)), nlags=30)
        print(acorr)
        plt.plot(acorr)

        # plot
        plot_acf(abs(self._logreturns.tail(-1)), lags=30)  # lags=20
        plt.show()


class PlottingData(Statistics):

    def __init__(self, asset, data, time):
        super().__init__(asset, time)
        self.prices()
        self.returns()
        self.logreturns()
        self.variance()
        self.volatility()
        # self.skewness_kurtosis()

        # print(self._prices)

        if data == "prices":
            # self.data = (self._prices / self._prices.iloc[0]) * 100
            self.data = self._prices

        elif data == "returns":
            self.data = self._returns.tail(-1)

        else:
            self.data = self._logreturns.tail(-1)

    def stationarity_tests(self):
        """
        Sationarity tests determine how strongly a time series is defined by a trend.
        Stationarity of returns will be rejected as p-values > 0.05
        Stationarity of logreturns will be accepted
        """

        # easy
        # result = adfuller(self.data)
        # print('ADF Stastistic: %f' % stat)
        # print('p-value: %f' % p)

        # Building the dataframe
        stats = list()
        p = list()
        for d in self.data:
            print(d)
            result = adfuller(self.data[d])
            stats.append(result[0])
            p.append(result[1])

        data = {'ADF': stats, 'p-value': p}
        print(pd.DataFrame.from_dict(data).set_index(self.data.columns))

    def plotter(self):
        plt.plot(self.data, label=self.asset, linewidth=1, alpha=0.6)
        plt.legend(loc='upper left', fontsize=12)
        plt.title('price comparison')
        plt.show()


if __name__ == '__main__':
    cryptos = ["BTC-USD", "ETH-USD", "XRP-USD", "ADA-USD"]

    # run crypto by crypto for autocorrelations and mean_returns
    for crypto in cryptos:
        print(crypto)
        PlottingData(asset=crypto, data="returns", time="5Y").mean_returns()

    # PlottingData(asset=cryptos, data="logreturns", time="5Y").plotter()

