import yfinance as yf
import numpy as np


class Study:

    CRYPTO = "BTC-USD"

    def __init__(self, start, end):
        self.start = start
        self.end = end

    def logreturns(self):
        prices = yf.download(self.CRYPTO, start=self.start, end=self.end).ffill().dropna()["Close"]
        return np.diff(np.log(prices))

    def volatility(self):
        return round(self.logreturns().std() * 252 ** 0.5, 2)


if __name__ == '__main__':
    up = ("up", "2020-09-27", "2021-03-08")
    flat = ("flat", "2020-01-05", "2020-07-05")
    down = ("down", "2021-11-07", "2022-06-12")

    periods = [up, flat, down]
    for period in periods:
        study = Study(start=period[1], end=period[2])
        print(f"Period {period[0]} = {period[slice(1, 3)]}"
              f"\n average logreturn: {round(np.mean(study.logreturns()) * 100, 2)}%"
              f"\n average volatility: {np.mean(study.volatility()) * 100}%")
