import pandas as pd
import time
import math
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
import numpy as np
from numpy import abs
from numpy import log
from numpy import sign
from scipy.stats import rankdata
import sklearn.preprocessing as prep
import matplotlib
import matplotlib.pyplot as plt
from keras.models import Model
from sklearn.neural_network import MLPRegressor
import import_ipynb
import multi_stock_split_concat
from Ranking_prediction_helper import data_process
from Alpha_factor import alpha_factor_helper
from tqdm import tqdm
from sklearn.neural_network import MLPClassifier

import keras
### initialize API ###
import tushare as ts
ts.set_token('5d2dd2c56bb822ac0e818aaa4b0b344f95c7d5e7b9c83c69e2ed90ff')
pro = ts.pro_api()

Q_available = pd.read_csv("csv_files/intrinsic_feature_with_codename_available.csv", index_col=0)
top_stocks = pd.read_csv("csv_files/top_stocks.csv", index_col=0)

Z_model = keras.models.load_model('model/m_0808.h5')
    
stock_list = pd.read_csv("csv_files/stock_list_avaliable.csv", index_col=0).index.values.tolist()
start = 0
end = 15
for i in tqdm(range(start,end)):
    print(i)
    window = 20
    stock_code = stock_list[i]
    get_stock_info = pd.read_csv('singleStockData/'+stock_code+".csv", index_col=0)
    
    trade_date = get_stock_info['trade_date'].to_list()[window-1:-2]
    train_len = round(0.9 * len(trade_date))
    train_date = trade_date[:train_len]
    test_date = trade_date[train_len:]
    
    alpha_factor_helper.get_alpha(get_stock_info)
    get_stock_info.insert(get_stock_info.shape[1],'close_price',get_stock_info['close'])
    get_stock_info.drop(['ts_code', 'trade_date', 'close'],axis=1,inplace=True)
    
    X_train_, y_train_, X_test_, y_test_ = data_process.split_data(get_stock_info[:: -1], window)
    
    X_train_input = data_process.input_cancat(X_train_, train_date, Z_model, Q_available, top_stocks)
    X_test_input = data_process.input_cancat(X_test_, test_date, Z_model, Q_available, top_stocks)
    y_train_=y_train_.astype('float')
    
    if stock_code == stock_list[start]:
        X_train = X_train_input
        y_train = y_train_
        X_test = X_test_input
        y_test = y_test_
    else:
        X_train = np.concatenate((X_train, X_train_input))
        y_train = np.concatenate((y_train, y_train_))
        X_test = np.concatenate((X_test, X_test_input))
        y_test = np.concatenate((y_test, y_test_))
    
clf=MLPRegressor(solver='sgd',alpha=1e-5,hidden_layer_sizes=(128,16),random_state=1)
clf.fit(X_train,y_train)

data_process.plot_predict(clf, X_test, y_test)