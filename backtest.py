import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tqsdk import TqApi, TqAuth, TqBacktest,TargetPosTask, TqSim
from datetime import date 
from deeplob_model import create_deeplob 
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

T = 100
deeplob = create_deeplob('OF', T)
checkpoint_filepath = './model_check/deeplob.weights.h5'
deeplob.load_weights(checkpoint_filepath)

ss = None


def trade(data: pd.DataFrame):
    col = ['bid_price1', 'bid_volume1', 'ask_price1', 'ask_volume1', 'bid_price2', 'bid_volume2', 'ask_price2', 'ask_volume2', 'bid_price3', 'bid_volume3', 'ask_price3', 'ask_volume3', 'bid_price4', 'bid_volume4', 'ask_price4', 'ask_volume4', 'bid_price5', 'bid_volume5', 'ask_price5', 'ask_volume5']
    data = data.loc[:, col].copy()
    data.dropna(inplace=True)
    data.reset_index(drop=True, inplace=True)
    
    of_data = pd.DataFrame(index=data.index)
    for i in range(1, 6):
        of_data[f'bOF_{i}'] = 0
        of_data[f'aOF_{i}'] = 0

        bid_price_diff = data[f'bid_price{i}'].diff()
        ask_price_diff = data[f'ask_price{i}'].diff()
        bid_volume_diff = data[f'bid_volume{i}'].diff()
        ask_volume_diff = data[f'ask_volume{i}'].diff()

        of_data.loc[bid_price_diff > 0, f'bOF_{i}'] = data.loc[bid_price_diff > 0, f'bid_volume{i}']
        of_data.loc[bid_price_diff < 0, f'bOF_{i}'] = -data.loc[bid_price_diff < 0, f'bid_volume{i}']
        of_data.loc[bid_price_diff == 0, f'bOF_{i}'] = bid_volume_diff

        of_data.loc[ask_price_diff > 0, f'aOF_{i}'] = -data.loc[ask_price_diff > 0, f'ask_volume{i}']
        of_data.loc[ask_price_diff < 0, f'aOF_{i}'] = data.loc[ask_price_diff < 0, f'ask_volume{i}']
        of_data.loc[ask_price_diff == 0, f'aOF_{i}'] = ask_volume_diff


    global ss
    if ss is None:
        ss = StandardScaler().fit(of_data)

    of_data = of_data.tail(T+1)
    real_data_normalized = pd.DataFrame(ss.transform(of_data))
    # Prepare the data for the model
    realX_CNN, _ = data_classification(real_data_normalized, np.zeros(len(real_data_normalized)), T)

    # Perform inference
    predictions = deeplob.predict(realX_CNN[-1:])

    return predictions[0] 


def data_classification(X, Y, T):
    [N, D] = X.shape
    df = np.array(X)
    dY = np.array(Y)
    dataY = dY[T - 1:N]
    dataX = np.zeros((N - T + 1, T, D),dtype='float16')
    for i in range(T, N + 1):
        dataX[i - T] = df[i - T:i, :]
    dataY += 1
    return dataX.reshape(dataX.shape + (1,)), dataY

try:

    api = TqApi(
        web_gui="localhost:8888",
        account=TqSim(init_balance=100000),
        backtest=TqBacktest(start_dt=date(2024, 10, 29), end_dt=date(2024, 10, 30)), 
        auth=TqAuth(os.environ['tqname'], os.environ['tqpassword']))


    symbol = "INE.nr2412"
    ticks = api.get_tick_serial(symbol, T*2)

    target_pos = TargetPosTask(api, symbol)

    while True:
        api.wait_update()
        print(pd.to_datetime(ticks.iloc[-1]['datetime']))
        pred = trade(ticks)
        label = np.argmax(pred)

        if pred[0] > 0.7:
            target_pos.set_target_volume(-1)
        elif pred[1] > 0.7:
            target_pos.set_target_volume(0)
        elif pred[2] > 0.7:
            target_pos.set_target_volume(1)


finally:
    api.close()

