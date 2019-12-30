import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from steamgen_LSTM_model import model

if __name__ == "__main__":

    df = pd.read_excel('steam_generator_data.xlsx')
    X = df[['fuel_in', 'disturb_in']]
    y = df['Steamflow_out']

    sc_x = MinMaxScaler()
    sc_y = MinMaxScaler()

    X_sc = sc_x.fit_transform(X)
    y_sc = sc_y.fit_transform(y.values.reshape(-1, 1))

    n_features = 1
    X_sc_reshape = X_sc.reshape(X_sc.shape[0], X_sc.shape[1], n_features)
    y_sc_reshape = y_sc.reshape(y_sc.shape[0], y_sc.shape[1], n_features)

    train_size = int(0.7 * len(X_sc_reshape))

    X_train = X_sc_reshape[:train_size]
    X_test = X_sc_reshape[train_size:]

    y_train = y_sc_reshape[:train_size]
    y_test = y_sc_reshape[train_size:]

    model = model()
    model.fit(X_train,y_train,100)

    y_pred = model.predict(X_test)


