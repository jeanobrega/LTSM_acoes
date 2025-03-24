import streamlit as st
import pandas as pd
import numpy as np
from urllib.request import Request,urlopen
from datetime import datetime
import json
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
pd.options.mode.chained_assignment = None
tf.random.set_seed(0)

if 'dados' not in st.session_state:
    st.session_state['dados']=pd.DataFrame()
if 'predict' not in st.session_state:
    st.session_state['predict']=pd.DataFrame()

def limpa(): #limpa ao mudar o selectbox
    st.session_state['dados']=pd.DataFrame()
    st.session_state['predict']=pd.DataFrame()

def get_data(symbol='BOV^BOVA11',freq='DAILY',period='1Y'):
    try:
        req = Request('https://br.advfn.com/common/api/charts/GetHistory?symbol='+symbol+'&frequency='+freq+'&resolution='+period,headers={'User-Agent':'Mozilla/5.0'})
        webURL = urlopen(req)
        data=webURL.read()
        encoding=webURL.info().get_content_charset('utf-8')
        dict = json.loads(data.decode(encoding))
        date = dict['result']['data']['t']
        #date = [datetime.utcfromtimestamp(d) for d in date]
        #date = [d.strftime('%d/%m/%Y') for d in date]
        open = dict['result']['data']['o']
        close = dict['result']['data']['c']
        high = dict['result']['data']['h'] 
        low = dict['result']['data']['l']
        dados = pd.DataFrame(
            {'Date':date,
            'Open':open,
            'High':high,
            'Low':low,
            'Close':close
            }
        )
        dados['Date'] = pd.to_datetime(dados['Date'],unit='s') # transforma Date de epoch para datetime
        dados.set_index('Date',inplace=True)
        st.session_state['dados']=dados
    except Exception as error:
        print('Erro (getData): ',error)
        return {'Erro': error}

def predict(df):
    y = df['Close']
    y = y.values.reshape(-1, 1)

    # scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(y)
    y = scaler.transform(y)

    # generate the input and output sequences
    samples = len(y)
    #n_lookback = 60  # length of input sequences (lookback period)
    n_lookback = int(samples*0.25)
    #n_forecast = 30  # length of output sequences (forecast period)
    n_forecast = int(n_lookback/2)

    X = []
    Y = []

    for i in range(n_lookback, len(y) - n_forecast + 1):
        X.append(y[i - n_lookback: i])
        Y.append(y[i: i + n_forecast])

    X = np.array(X)
    Y = np.array(Y)

    # fit the model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(n_lookback, 1)))
    model.add(LSTM(units=50))
    model.add(Dense(n_forecast))

    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X, Y, epochs=100, batch_size=32, verbose=2)

    # generate the forecasts
    X_ = y[- n_lookback:]  # last available input sequence
    X_ = X_.reshape(1, n_lookback, 1)

    Y_ = model.predict(X_).reshape(-1, 1)
    Y_ = scaler.inverse_transform(Y_)

    # organize the results in a data frame
    ##df_past = df[['Close']].reset_index()
    df_past = df[['Close']]
    df_past.rename(columns={'index': 'Date', 'Close': 'Actual'}, inplace=True)
    #df_past['Date'] = pd.to_datetime(df['Date'],unit='s') # transforma Date de epoch para datetime
    ##df_past['Date'] = df['Date']
    df_past['Date'] = df.index # index é a data
    df_past['Forecast'] = np.nan
    df_past['Forecast'].iloc[-1] = df_past['Actual'].iloc[-1]

    df_future = pd.DataFrame(columns=['Date', 'Actual', 'Forecast'])
    df_future['Date'] = pd.date_range(start=df_past['Date'].iloc[-1] + pd.Timedelta(days=1), periods=n_forecast)
    df_future['Forecast'] = Y_.flatten()
    df_future['Actual'] = np.nan
   
    results = pd.concat([df_past,df_future]).set_index('Date')
    st.session_state['predict']=results
    


with st.container(border=True):
    st.title("Previsão Ações")
    acao = st.selectbox("Ação:",["BOVA11","PETR4","BBAS3"],on_change=limpa)
    periodo = st.select_slider("Período anterior (meses)", [3,6,12,36,60],12,on_change=limpa)
    symbol = 'BOV^'+acao
    frequencia='WEEKLY' if periodo > 12 else 'DAILY'
    period=str(periodo)+'M' if periodo < 12 else str(int(periodo/12))+'Y'
    st.button("Carregar Dados",on_click=get_data,args=[symbol,frequencia,period])
    st.button("Iniciar Previsão!",on_click=predict,args=[st.session_state['dados']])
print(period)
tab1, tab2 = st.tabs(["Dataframe", "Chart"])#,"Prediction"])
dados=st.session_state['dados']
prediction = st.session_state['predict']


tab1.dataframe(dados, height=250, use_container_width=True)

if not dados.empty:
    print(dados['Close'])
    fig, ax = plt.subplots()
    ax=dados['Close'].plot()
    ax.grid()
    tab2.pyplot(fig)

if not prediction.empty:
    print(prediction)
    fig, ax = plt.subplots()
    ax=prediction['Actual'].plot()
    ax=prediction['Forecast'].plot()
    ax.grid()
    tab2.pyplot(fig)








