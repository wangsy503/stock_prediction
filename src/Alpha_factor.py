import math
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
import numpy as np
from numpy import abs
from numpy import log
from numpy import sign
from scipy.stats import rankdata
import pandas as pd
import sklearn.preprocessing as prep
import matplotlib
import matplotlib.pyplot as plt
from keras.models import Model

class alpha_factor_helper:
    def ts_sum(df, window=10):
        return df.rolling(window).sum()

    def sma(df, window=10):
        return df.rolling(window).mean()

    def stddev(df, window=10):
        return df.rolling(window).std()

    def correlation(x, y, window=10):
        return x.rolling(window).corr(y)

    def covariance(x, y, window=10):
        return x.rolling(window).cov(y)

    def rolling_rank(na):
        return rankdata(na)[-1]

    def ts_rank(df, window=10):
        return df.rolling(window).apply(rolling_rank)

    def rolling_prod(na):
        return np.prod(na)

    def product(df, window=10):
        return df.rolling(window).apply(rolling_prod)

    def ts_min(df, window=10):
        return df.rolling(window).min()

    def ts_max(df, window=10):
        return df.rolling(window).max()

    def delta(df, period=1):
        return df.diff(period)

    def delay(df, period=1):
        return df.shift(period)

    def rank(df):
        return df.rank(pct=True)

    def scale(df, k=1):
        return df.mul(k).div(np.abs(df).sum())

    def ts_argmax(df, window=10):
        return df.rolling(window).apply(np.argmax) + 1 

    def ts_argmin(df, window=10):
        return df.rolling(window).apply(np.argmin) + 1

    def decay_linear(df, period=10):
        if df.isnull().values.any():
            df.fillna(method='ffill', inplace=True)
            df.fillna(method='bfill', inplace=True)
            df.fillna(value=0, inplace=True)
        na_lwma = np.zeros_like(df)
        na_lwma[:period, :] = df.iloc[:period, :] 
        na_series = df.as_matrix()

        divisor = period * (period + 1) / 2
        y = (np.arange(period) + 1) * 1.0 / divisor
        # Estimate the actual lwma with the actual close.
        # The backtest engine should assure to be snooping bias free.
        for row in range(period - 1, df.shape[0]):
            x = na_series[row - period + 1: row + 1, :]
            na_lwma[row, :] = (np.dot(x.T, y))
        return pd.DataFrame(na_lwma, index=df.index, columns=['CLOSE'])  
    # endregion

    def get_alpha(df):
        stock=Alphas(df)
        df['alpha005']=stock.alpha005()
        df['alpha021']=stock.alpha021()
        df['alpha028']=stock.alpha028()
        df['alpha033']=stock.alpha033()
        df['alpha041']=stock.alpha041()
        df['alpha042']=stock.alpha042()
        df['alpha044']=stock.alpha044()
        df['alpha054']=stock.alpha054()
        df['alpha055']=stock.alpha055()
        return df

class Alphas(object):
    def __init__(self, df_data):

        self.open = df_data['open'] 
        self.high = df_data['high'] 
        self.low = df_data['low']   
        self.close = df_data['close'] 
        self.volume = df_data['vol']*100 
        self.returns = df_data['pct_chg'] 
        self.vwap = (df_data['amount']*1000)/(df_data['vol']*100+1) 
        
    # Alpha#5	 (rank((open - (sum(vwap, 10) / 10))) * (-1 * abs(rank((close - vwap)))))
    def alpha005(self):
        return  (alpha_factor_helper.rank((self.open - (sum(self.vwap, 10) / 10))) * (-1 * abs(alpha_factor_helper.rank((self.close - self.vwap)))))

    # Alpha#21	 ((((sum(close, 8) / 8) + stddev(close, 8)) < (sum(close, 2) / 2)) ? (-1 * 1) : (((sum(close,2) / 2) < ((sum(close, 8) / 8) - stddev(close, 8))) ? 1 : (((1 < (volume / adv20)) || ((volume /adv20) == 1)) ? 1 : (-1 * 1))))
    def alpha021(self):
        cond_1 = alpha_factor_helper.sma(self.close, 8) + alpha_factor_helper.stddev(self.close, 8) < alpha_factor_helper.sma(self.close, 2)
        cond_2 = alpha_factor_helper.sma(self.volume, 20) / self.volume < 1
        alpha = pd.DataFrame(np.ones_like(self.close), index=self.close.index
                             )
        alpha[cond_1 | cond_2] = -1
        return alpha

    # Alpha#28	 scale(((correlation(adv20, low, 5) + ((high + low) / 2)) - close))
    def alpha028(self):
        adv20 = alpha_factor_helper.sma(self.volume, 20)
        df = alpha_factor_helper.correlation(adv20, self.low, 5)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        return alpha_factor_helper.scale(((df + ((self.high + self.low) / 2)) - self.close))

    # Alpha#33	 rank((-1 * ((1 - (open / close))^1)))
    def alpha033(self):
        return alpha_factor_helper.rank(-1 + (self.open / self.close))
    
    # Alpha#41	 (((high * low)^0.5) - vwap)
    def alpha041(self):
        return pow((self.high * self.low),0.5) - self.vwap
    
    # Alpha#42	 (rank((vwap - close)) / rank((vwap + close)))
    def alpha042(self):
        return alpha_factor_helper.rank((self.vwap - self.close)) / alpha_factor_helper.rank((self.vwap + self.close))
    
    # Alpha#44	 (-1 * correlation(high, rank(volume), 5))
    def alpha044(self):
        df = alpha_factor_helper.correlation(self.high, alpha_factor_helper.rank(self.volume), 5)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        return -1 * df

    # Alpha#54	 ((-1 * ((low - close) * (open^5))) / ((low - high) * (close^5)))
    def alpha054(self):
        inner = (self.low - self.high).replace(0, -0.0001)
        return -1 * (self.low - self.close) * (self.open ** 5) / (inner * (self.close ** 5))

    # Alpha#55	 (-1 * correlation(rank(((close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low,12)))), rank(volume), 6))
    def alpha055(self):
        divisor = (alpha_factor_helper.ts_max(self.high, 12) - alpha_factor_helper.ts_min(self.low, 12)).replace(0, 0.0001)
        inner = (self.close - alpha_factor_helper.ts_min(self.low, 12)) / (divisor)
        df = alpha_factor_helper.correlation(alpha_factor_helper.rank(inner), alpha_factor_helper.rank(self.volume), 6)
        return -1 * df.replace([-np.inf, np.inf], 0).fillna(value=0)
    
