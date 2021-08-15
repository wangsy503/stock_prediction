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
import pandas as pd
import sklearn.preprocessing as prep
import matplotlib
import matplotlib.pyplot as plt
from keras.models import Model
from Ranking_prediction_helper import data_process

from Alpha_factor import Alphas
from Alpha_factor import alpha_factor_helper

def build_model(layers):
    model = Sequential()
    verbose = 0
    
    # By setting return_sequences to True we are able to stack another LSTM layer
    model.add(LSTM(input_dim=layers[0],units=layers[1],return_sequences=True))
    # do not change these two dropouts, the result are not better
    model.add(Dropout(0.4))

    model.add(LSTM(layers[2],return_sequences=False))
    model.add(Dropout(0.3))

    model.add(Dense(units=layers[3]))
    
    model.add(Activation("linear"))

    # start = time.time()
    
    from keras import metrics
    import keras.backend as K

    
    model.compile(loss='mean_squared_error',
              optimizer='rmsprop',
              metrics=['accuracy'])

    # print("Compilation Time : ", time.time() - start)
    
    return model


if __name__ == "__main__":

    stock_list = pd.read_csv("csv_files/stock_list_avaliable.csv", index_col=0).index.values.tolist()
    unavail_stock_list = []
    start = 0
    end = 1000
    for i in range(start,end):
        stock_code = stock_list[i]

        get_stock_info = pd.read_csv('singleStockData/'+stock_code+".csv", index_col=0)
        #### IMPORTANT!!! Since the date in single stock data is arranged from newest to oldest! ####
        get_stock_info = get_stock_info.iloc[::-1]
        
        if get_stock_info.shape[0] != 730:
            print(stock_code, get_stock_info.shape[0])
            unavail_stock_list.append(stock_code)
            continue
        alpha_factor_helper.get_alpha(get_stock_info)
        get_stock_info.insert(get_stock_info.shape[1],'close_price',get_stock_info['close'])
        get_stock_info.drop(['ts_code', 'trade_date', 'close'],axis=1,inplace=True)

        window = 20
        X_train_, y_train_, X_test_, y_test_ = data_process.split_data(get_stock_info[:: -1], window)
        if stock_code == stock_list[start]:
            X_train = X_train_
            y_train = y_train_
            X_test = X_test_
            y_test = y_test_
        else:
            X_train = np.concatenate((X_train, X_train_))
            y_train = np.concatenate((y_train, y_train_))
            X_test = np.concatenate((X_test, X_test_))
            y_test = np.concatenate((y_test, y_test_))
            
    model = build_model([X_train.shape[2], window, 100, 1])
    
    model.fit(X_train,y_train,batch_size=5,epochs=10,validation_split=0.1,verbose=1)

    trainScore = model.evaluate(X_train, y_train, verbose=1)
    print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore[0], math.sqrt(trainScore[0])))

    testScore = model.evaluate(X_test, y_test, verbose=1)
    print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore[0], math.sqrt(testScore[0])))

    data_process.plot_predict(model, X_test, y_test)

    dense1_layer_model = Model(inputs=model.input,outputs=model.layers[-1].output)
    #以这个model的预测值作为输出
    dense1_output = dense1_layer_model.predict(X_test)

    print(dense1_output) # --> which is Z
    