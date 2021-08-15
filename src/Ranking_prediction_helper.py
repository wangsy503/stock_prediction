import numpy as np
import pandas as pd
import keras
import import_ipynb
import os
import sklearn.preprocessing as prep
import matplotlib.pyplot as plt

class data_process:
    
    def normalization(X_train, X_test):
        #normalize, for fear that different scales might do harm to the training
        train_samples, train_nx, train_ny = X_train.shape
        test_samples, test_nx, test_ny = X_test.shape

        X_train = X_train.reshape((train_samples, train_nx * train_ny))
        X_test = X_test.reshape((test_samples, test_nx * test_ny))

        global preprocessor
        preprocessor = prep.StandardScaler().fit(X_train)
        X_train = preprocessor.transform(X_train)
        X_test = preprocessor.transform(X_test)

        X_train = X_train.reshape((train_samples, train_nx, train_ny))
        X_test = X_test.reshape((test_samples, test_nx, test_ny))

        return X_train, X_test
    
    def split_data(stock,seq_len):
        # validation:train=1:9
        amount_of_features=len(stock.columns)
        data=stock.values
    #     sequence_length=seq_len+1
        sequence_length=seq_len+1
        # 'result' is to do time slicing day by day
        result=[]
        for index in range(len(data)-sequence_length):
            result.append(data[index : index + sequence_length])
        result=np.array(result)

        row=round(0.9 * result.shape[0])

        train=result[:int(row), :]

        train, result=data_process.normalization(train, result)

        X_train = train[:, :-1]
        y_train = train[:, -1][:, -1]
        X_test = result[int(row):, :-1]
        y_test = result[int(row):, -1][ : ,-1]

        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], amount_of_features))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], amount_of_features))

        return [X_train, y_train, X_test, y_test]

    def input_cancat(X_input, dates, Z_model, Q_available, top_stocks):
        Z_ = Z_model.predict(X_input) # len: 638
        D_ = pd.DataFrame(columns = Q_available.columns)
        for j in dates:
            ### market representation calculation ###
            date = j
            top_list = np.array(top_stocks.loc[[j]]).tolist()
            # construct a new frame to store top 10 stocks' intrinsics
            top_stock_in = pd.DataFrame(columns = Q_available.columns)
            top_stock_in.index.name = Q_available.index.name
            for stock in top_list[0]:
                if stock not in Q_available.index.tolist():
                    continue
                tmp = Q_available.loc[[stock]]
                top_stock_in = top_stock_in.append(tmp)
            top_stock_avg_in = top_stock_in.mean()
            S = top_stock_avg_in.to_list()  # Now we've got S for this day.
            #tmp = Q_available.loc[[stock_code]]
            for i in range(len(S)):
                tmp.iloc[:,i] = tmp.iloc[:,i].apply(lambda x: x*S[i])
            D_ = D_.append(tmp)

            '''
            Now we've got D. Let's concatenate D with Z!
            '''
        #     print(D)
        D_ = D_.to_numpy()
        Input = np.concatenate((Z_, D_), axis=1)
        return Input

    def plot_predict(model, X_test, y_test):
        diff = []
        ratio = []
        pred = model.predict(X_test)
        for u in range(len(y_test)):
            pr = pred[u]
            ratio.append((y_test[u] / pr) - 1)
            diff.append(abs(y_test[u] - pr))

        get_ipython().run_line_magic('matplotlib', 'inline')

        plt.plot(pred, color='red', label='Prediction')
        plt.plot(y_test, color='black', label='Ground Truth')
        plt.legend(loc='upper left')
        plt.show()