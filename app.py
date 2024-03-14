import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from vnstock import * #import all functions
from adding_features import technical_analysis_indicator, take_news_parameter, add_new_working_days
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.momentum import RSIIndicator
import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from catboost import CatBoostRegressor
import plotly.express as px
from vnstock.chart import candlestick_chart, bollinger_bands, bollinger_bands_chart # import chart functions

st.set_page_config(page_title="Stock Prediction App", page_icon="📈", layout="wide")

# Apply custom CSS to change the color of st.sidebar.info
st.markdown(
    """
    <style>

    </style>
    """,
    unsafe_allow_html=True
)
# # Load external CSS file
with open("main.css") as f:
    st.markdown(f"""<style>{f.read()}</style>""", unsafe_allow_html=True)

st.title('Stock Price Predictions')
st.sidebar.info('Welcome to the Stock Price Prediction App. Choose your options below')
st.sidebar.info("Created and designed by [Dylan Nguyen](https://www.linkedin.com/in/quang-nguyen-4b6a52287/)")

def main():
    option = st.sidebar.selectbox('Make a choice', ['Visualize','Recent Data', 'Predict'])
    if option == 'Visualize':
        tech_indicators(data)
    elif option == 'Recent Data':
        dataframe()
    elif option == 'Predict':
        predict()



@st.cache_resource
def download_data(op, start_date, end_date, index_option='stock'):
    start_date = start_date.strftime("%Y-%m-%d")
    end_date = end_date.strftime("%Y-%m-%d")
    if index_option == "stock":
        df =  stock_historical_data(symbol=op, 
                                start_date=start_date,
                                end_date=end_date, resolution='1D', type=index_option, beautify=True, decor=False)
    else:
        df =  stock_historical_data(symbol=op, 
                        start_date=start_date,
                        end_date=end_date, resolution='1D', type=index_option, beautify=True, decor=False, source="TCBS")
    # df = yf.download(op, start=start_date, end=end_date, progress=False)
    return df

@st.cache_resource
def take_sp500(start_date, end_date):
    start_date = start_date.strftime("%Y-%m-%d")
    end_date = end_date.strftime("%Y-%m-%d")
    ticker_symbol = "^GSPC"
    # Fetch the historical data
    sp500_data = yf.download(ticker_symbol, start=start_date, end=end_date)
    sp500_data.reset_index(inplace=True)
    sp500_data.rename(columns={'Date': 'time'}, inplace=True) # Rename the 'Date' column to 'time'
    sp500_data.drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis=1, inplace=True)
    # Display the data
    return sp500_data

@st.cache_resource
def take_vnindex(start_date, end_date):
    start_date = start_date.strftime("%Y-%m-%d")
    end_date = end_date.strftime("%Y-%m-%d")
    df =  stock_historical_data(symbol="VNINDEX", 
                            start_date=start_date,
                            end_date=end_date, resolution='1D', type="index", beautify=True, decor=False, source='TCBS')
    df.rename(columns={'close': 'vnindex close'}, inplace=True) # Rename the 'Date' column to 'time'
    df.drop(['open', 'high', 'low', 'volume', 'ticker'], axis=1, inplace=True)
    return df

listing_stocks = listing_companies(True)
HOSE_listing_stocks = listing_stocks[listing_stocks["comGroupCode"] == "HOSE"].copy()['ticker'].to_list()
listing_indexes = ["VNINDEX", "VN30", "HNX30"]
# Single select dropdown
default_ix = HOSE_listing_stocks.index("VHM")
index_option = st.radio("Select stock or index:", ['index', 'stock'], index=1)


if index_option == "stock":
    option = st.selectbox("Select a stock:", HOSE_listing_stocks, index=default_ix)
else:
    option = st.selectbox("Select a stock:", listing_indexes, index=1)
    
option = option.upper()
today = datetime.date.today()
# today = datetime.date.today() - timedelta(days=1)
# define date
duration = st.sidebar.number_input('Enter the duration', value=1000)
before = datetime.date(2019, 1, 1)
start_date = st.sidebar.date_input('Start Date', value=before)
end_date = st.sidebar.date_input('End date', today)

if st.sidebar.button('Send'):
    if start_date < end_date:
        st.sidebar.success('Start date: `%s`\n\nEnd date: `%s`' %(start_date, end_date))
        download_data(option, start_date, end_date)
    else:
        st.sidebar.error('Error: End date must fall after start date')

data = download_data(option, start_date, end_date, index_option)
sp500_data = take_sp500(start_date, end_date)
vnindex_data = take_vnindex(start_date, end_date)
scaler = StandardScaler()


def tech_indicators(data):
    st.header('Technical Indicators')
    option = st.radio('Choose a Technical Indicator to Visualize', ['Candlestick chart','Close', 'BB', 'MACD', 'RSI', 'SMA', 'EMA'])

    # MACD
    macd = MACD(data.close).macd()
    # RSI
    rsi = RSIIndicator(data.close).rsi()
    # SMA
    sma = SMAIndicator(data.close, window=14).sma_indicator()
    # EMA
    ema = EMAIndicator(data.close).ema_indicator()

    if option == 'Candlestick chart':
        st.write("Candlestick chart")
        # Create the candlestick chart
        fig = candlestick_chart(data, ma_periods=[20, 200], show_volume=False, reference_period=300,
                                title=f'{option} - Candlestick Chart with MA and Volume', x_label='Date', y_label='Price',
                                reference_colors=('black', 'blue'))
        # Update layout
        fig.update_layout(
            xaxis_title='Time',
            yaxis_title='Price',
            hovermode='x',
            yaxis=dict(autorange = True,
                    fixedrange= False),
            # plot_bgcolor='#f5f5f5',  # Background color
            # xaxis_tickcolor='rgba(0,0,0,0.5)',  # Tick color
            # yaxis_tickcolor='rgba(0,0,0,0.5)',  # Tick color
            font=dict(color='black', size=14),  # Text color
        )
        fig.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list([
                    dict(count=7, label="1w", step="day", stepmode="backward"),
                    dict(count=14, label="2w", step="day", stepmode="backward"),
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            )
        )
        data = data.drop(['20-day MA', '200-day MA','lowest_low', 'highest_high'], axis=1)
        # Display the chart in the Streamlit app with responsive sizing
        st.plotly_chart(fig, use_container_width=True)
    elif option == 'Close':
        st.write('Close Price')
        st.line_chart(data.close)
    elif option == 'BB':
        st.write('BollingerBands')
        data = bollinger_bands(data)
        fig = bollinger_bands_chart(data, fig_size=(15, 8), chart_title='Bollinger Bands Chart',show_volume=False,
                                    xaxis_title='Date', yaxis_title='Price', bollinger_band_colors=('gray', 'orange', 'gray'), 
                                    volume_colors=('#00F4B0', '#FF3747'))
        # Update layout
        fig.update_layout(
            xaxis_title='Time',
            yaxis_title='Price',
            hovermode='x',
            yaxis=dict(autorange = True,
                    fixedrange= False),
            # plot_bgcolor='#f5f5f5',  # Background color
            # xaxis_tickcolor='rgba(0,0,0,0.5)',  # Tick color
            # yaxis_tickcolor='rgba(0,0,0,0.5)',  # Tick color
            font=dict(color='black', size=14),  # Text color
        )
        fig.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list([
                    dict(count=7, label="1w", step="day", stepmode="backward"),
                    dict(count=14, label="2w", step="day", stepmode="backward"),
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            )
        )
        data = data.drop(['upper_band', 'lower_band','middle_band'], axis=1)
        st.plotly_chart(fig, use_container_width=True)
    elif option == 'MACD':
        st.write('Moving Average Convergence Divergence')
        st.line_chart(macd)
    elif option == 'RSI':
        st.write('Relative Strength Indicator')
        st.line_chart(rsi)
    elif option == 'SMA':
        st.write('Simple Moving Average')
        st.line_chart(sma)
    else:
        st.write('Expoenetial Moving Average')
        st.line_chart(ema)


def dataframe():
    st.header('Recent Data')
    st.dataframe(data.tail(10))


def predict():
    model = st.radio('Choose a model', ['RandomForestRegressor', 'ExtraTreesRegressor', 'LSTM', 'XGBoostRegressor', 'CatBoostRegressor'])
    num = st.number_input('How many days forecast?', value=5)
    num = int(num)
    
    if "df_w_visualization" not in st.session_state:
        st.session_state["df_w_visualization"] = pd.DataFrame()
    if f"model_{option}_{num}" not in st.session_state:
        st.session_state[f"model_{option}_{num}"] = None
    if "prediction_result" not in st.session_state:
        st.session_state["prediction_result"] = pd.DataFrame()
    
    if st.button('Predict'):
        if model == 'RandomForestRegressor':
            engine = RandomForestRegressor(n_estimators=200, criterion="absolute_error")
        elif model == 'ExtraTreesRegressor':
            engine = ExtraTreesRegressor()
        elif model == 'KNeighborsRegressor':
            engine = KNeighborsRegressor()
        elif model == 'CatBoostRegressor':
            engine = CatBoostRegressor(n_estimators=600, loss_function='RMSE')
        elif model == 'XGBoostRegressor':
            engine = XGBRegressor(base_score=0.5, booster='gblinear',    
                           n_estimators=900,
                           objective='reg:squarederror',
                           learning_rate=0.03)
        if st.session_state[f"model_{option}_{num}"] is not None:
            existed_model_engine(st.session_state[f"model_{option}_{num}"], num)
        else:
            model_engine(engine, num)
        # Selectbox for time period options
    # if st.session_state["df_w_visualization"] is not None:
        # time_period_option = st.radio("Select Time Period", ["1 week", "2 weeks", "1 month", "1 year", "all"], index=4)

        # if time_period_option == "1 week":
        # Create the line chart based on user input
        # fig = create_predicted_chart(st.session_state["df_w_visualization"])
        # st.plotly_chart(fig, use_container_width=True)
        # elif time_period_option == "2 weeks":
        #     fig = create_predicted_chart(st.session_state.df_w_visualization, 15)
        #     st.plotly_chart(fig, use_container_width=True)
        # elif time_period_option == "1 month":
        #     fig = create_predicted_chart(st.session_state.df_w_visualization, 22)
        #     st.plotly_chart(fig, use_container_width=True)
        # elif time_period_option == "1 year":
        #     fig = create_predicted_chart(st.session_state.df_w_visualization, 255)
        #     st.plotly_chart(fig, use_container_width=True)
        # else:
        #     fig = create_predicted_chart(st.session_state.df_w_visualization)
        #     st.plotly_chart(fig, use_container_width=True)
        # st.dataframe(st.session_state.prediction_result)


def model_preprocess(df, num):
    # Display the updated DataFrame
    df, next_working_days = add_new_working_days(df, num)
    # shifting the closing price based on number of days forecast
    # Calculate the 'return' column
    df['return'] = df['close'].pct_change() * 100
    vnindex_data['vnindex return'] = vnindex_data['vnindex close'].pct_change() * 100
    sp500_data['SP500 return'] = sp500_data['Adj Close'].pct_change() * 100
    
    # Assuming df is your DataFrame and 'date_column' is the column to convert
    df['time'] = pd.to_datetime(df['time']).dt.date
    
    # merge dfs
    sp500_data['time'] = pd.to_datetime(sp500_data['time']).dt.date
    vnindex_data['time'] = pd.to_datetime(vnindex_data['time']).dt.date

    df = pd.merge(df, sp500_data, how='left', on="time")
    df = pd.merge(df, vnindex_data, how='left', on="time")
    df.rename(columns={'Adj Close': 'SP500 Close'}, inplace=True) # Rename the 'Date' column to 'time'
    # Replace missing values in 'SP500 return' with the mean
    df[['SP500 return', 'SP500 Close',
                        'vnindex return', 'vnindex close']] = df[['SP500 return', 'SP500 Close',
                                                                                'vnindex return', 'vnindex close']].apply(lambda col: col.fillna(col.mean()))
    
    
    # adding news param
    # take news df
    lag = 1
    news_df = take_news_parameter(["News_headlines/nlp_for_all_news_data.csv"], lag=lag)

    df = pd.merge(df, news_df, how='left', on="time")

    df[['roberta_neg', 'roberta_pos',
                        'roberta_neu']] = df[['roberta_neg', 'roberta_pos',
                                                                'roberta_neu']].apply(lambda col: col.fillna(col.mean()))
    
    
    # adding technical ana
    TAI_converter = technical_analysis_indicator(df, 'return')

    TAI_converter.calculate_rsi(7)
    TAI_converter.calculate_rsi(14)
    # Display the modified DataFrame with the 'rsi' column
    TAI_converter.calculate_sma(5)
    TAI_converter.calculate_sma(40)
    TAI_converter.calculate_ema(3)
    TAI_converter.calculate_ema(100)
    TAI_converter.calculate_macd()
    df_w_technical_features = TAI_converter.calculate_bollinger_bands()
    
    # Adding lag variables
    # if 22 meaning we use return today to predict one month later, a month has 22 working days
    num_of_days_to_lag = 30
    TAI_converter_lag = technical_analysis_indicator(df_w_technical_features)
    TAI_converter_lag.create_lag_variables('time', ['close', 'SP500 Close', 'vnindex close'], lag=num, num_of_days_to_lag=num_of_days_to_lag)
    # TAI_converter.create_lag_variables('time', ['roberta_neg', 'roberta_pos'], lag=1)
    TAI_converter_lag.create_lag_variables('time', ['open','high', 'low','return', 'SP500 return','vnindex return'], lag=num, num_of_days_to_lag=5)
    df = TAI_converter_lag.get_data()
    
    df_features_select = df[100:].drop(['open','high', 'low','SP500 Close', 'SP500 return', 'return',
                                                                                                     'vnindex return', 'vnindex close'],axis=1) 
    
    df_preprocessed = df_features_select.set_index('time')
    df_preprocessed = df_preprocessed.sort_index()
    return df_preprocessed


def create_predicted_chart(df):
    # Filter data based on the selected time period

    # Create an interactive line chart
    fig = px.line(df, x=df.index, y=['close', 'Predicted Close'], markers=True,
                  title=f'Comparison between predicted close and real close price {option}')

    # Update trace colors individually
    fig.update_traces(line_color='#ee7527', selector=dict(name='close'))  # Primary color
    fig.update_traces(line_color='#0B66C5', selector=dict(name='Predicted Close'))  # Text color
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=7, label="1w", step="day", stepmode="backward"),
                dict(count=14, label="2w", step="day", stepmode="backward"),
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        ),
        range=[df.index.min(), df.index.max()]
    )
    
    # Automatically adjust y-axis range based on data within selected time period
    # fig.update_yaxes(range=[df['close'].min(), df['close'].max()])

    # Update layout
    fig.update_layout(
        xaxis_title='Time',
        yaxis_title='Price',
        hovermode='x',
        yaxis=dict(autorange = True,
                   fixedrange= False),
        # plot_bgcolor='#f5f5f5',  # Background color
        # xaxis_tickcolor='rgba(0,0,0,0.5)',  # Tick color
        # yaxis_tickcolor='rgba(0,0,0,0.5)',  # Tick color
        font=dict(color='black', size=14),  # Text color
    )

    return fig


# def show_prediction_result(model, df_w_visualization): 

def model_engine(model, num):
    # getting only the closing price
    df = data[['time','close','open','high','low']]
    df_preprocessed = model_preprocess(df, num)
    
    # scaling the data
    df_available = df_preprocessed[df_preprocessed['isFuture']==False].copy()
    # if it passed 14h30 then apply close price of today, else not
    current_hour = datetime.datetime.now().hour
    current_minute = datetime.datetime.now().minute
    if (current_hour < 14 and current_minute < 30)  and (current_hour > 9 and current_minute>15):
        df_available = df_available[:-1]
        print(df_available.tail())
        print(current_hour)
        print(current_minute)
    
    TARGET = 'close'
    X = df_available.drop(columns=[TARGET, 'isFuture']).values
    # X = df.drop(['time','isFuture','close'], axis=1).values
    X = scaler.fit_transform(X)
    # getting the preds column
    y = df_available[TARGET].values
    # storing the last num_days data
    df_new = df_preprocessed[df_preprocessed['isFuture']==True].copy()
    x_forecast = df_new.drop(['isFuture',TARGET], axis=1).values
    x_forecast = scaler.transform(x_forecast)
    
    #spliting the data
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=.2, shuffle=False)
    # training the model
    progress_bar = st.progress(0)
    for perc_completed in range(100):
        time.sleep(0.000001)
        progress_bar.progress(perc_completed+1)
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    st.success("Model built successful!")
    col1, col2, col3, col4 = st.columns([1,1,1,1])
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)*100
    mae = mean_absolute_error(y_test, preds)
    mape =  np.mean(np.abs((y_test - preds) / y_test))*100
    
    col1.metric(label="RMSE", value=round(rmse, 2), delta=f"{round(rmse - 0.01 * df_available.close.mean(), 2)}", delta_color="inverse")
    col2.metric(label="R-squared", value=f"{round(r2, 2)}%", delta=f"{round(r2 - 91, 2)}%")
    col3.metric(label="MAE", value=round(mae, 2), delta=f"{round(mae - 0.01 * df_available.close.mean(), 2)}", delta_color="inverse")
    col4.metric(label="MAPE", value=f"{round(mape, 2)}%", delta=f"{round(mape - 1.2, 2)}%", delta_color="inverse")
    # st.write(f'r2_score: {r2:.2f}% \
    #         \nMAE: {mae:.2f}\
    #         \nRMSE: {rmse:.2f}\
    #         \nMAPE: {mape:.2f}%')
    # predicting stock price based on the number of days
    y_new = model.predict(x_forecast)
    # st.session_state.model = model
    df_w_visualization = df_preprocessed[['close', 'isFuture']].copy()
    y_pred_all = model.predict(X)
    print("Length of y_pred_all:", len(y_pred_all))
    print("Number of rows with 'isFuture' False:", len(df_w_visualization[df_w_visualization['isFuture'] == False]))
    # Create a new column 'y_pred_new' and initialize it with NaN
    df_w_visualization['Predicted Close'] = np.nan
    # Update 'y_pred_new' column for rows where 'isFuture' is True
    df_w_visualization['Predicted Close'][:df_available.shape[0]] = y_pred_all
    
    # Replace 'close' values with predicted values when 'isFuture' is True
    df_w_visualization['Predicted Close'][df_available.shape[0]:] = y_new
    
    # Create an interactive line chart
    st.session_state["df_w_visualization"] = df_w_visualization 
    fig = create_predicted_chart(st.session_state["df_w_visualization"])
    st.plotly_chart(fig, use_container_width=True)

    # Show the plot using Streamlit
    prediction_result = df_w_visualization.copy()
    prediction_result['ticker'] = option
    prediction_result['RMSE'] = rmse
    prediction_result['MAE'] = mae
    prediction_result['r-squared'] = r2
    st.session_state["prediction_result"] = prediction_result
    
    # Existed model
    st.session_state[f"model_{option}_{num}"] = model
    col1, col2 = st.columns([1,1])
    col1.dataframe(st.session_state["prediction_result"][-num:])
    excel_file_path = 'prediction_result.xlsx'  # Adjust the file path as needed
    prediction_result.to_excel(excel_file_path)
    col2.download_button(
        label="Download data as Excel",
        data=open(excel_file_path, 'rb').read(),
                        file_name=f"{option}_prediction_{num}_days.xlsx",
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',)

def existed_model_engine(model, num):
        # getting only the closing price
    df = data[['time','close','open','high','low']]
    df_preprocessed = model_preprocess(df, num)
    
    # scaling the data
    df_available = df_preprocessed[df_preprocessed['isFuture']==False].copy()
    # if it passed 14h30 then apply close price of today, else not
    current_hour = datetime.datetime.now().hour
    current_minute = datetime.datetime.now().minute
    if (current_hour < 14 and current_minute < 30)  and (current_hour > 9 and current_minute>15):
        df_available = df_available[:-1]
        print(df_available.tail())
        print(current_hour)
        print(current_minute)
    
    TARGET = 'close'
    X = df_available.drop(columns=[TARGET, 'isFuture']).values
    # X = df.drop(['time','isFuture','close'], axis=1).values
    X = scaler.fit_transform(X)
    # getting the preds column
    y = df_available[TARGET].values
    # storing the last num_days data
    df_new = df_preprocessed[df_preprocessed['isFuture']==True].copy()
    x_forecast = df_new.drop(['isFuture',TARGET], axis=1).values
    x_forecast = scaler.transform(x_forecast)

    # predicting stock price based on the number of days
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=.2, shuffle=False)
    
    st.success("Model built successful!")
    preds = model.predict(x_test)
    col1, col2, col3, col4 = st.columns([1,1,1,1])
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)*100
    mae = mean_absolute_error(y_test, preds)
    mape =  np.mean(np.abs((y_test - preds) / y_test))*100
    
    col1.metric(label="RMSE", value=round(rmse, 2), delta=f"{round(rmse - 0.01 * df_available.close.mean(), 2)}", delta_color="inverse")
    col2.metric(label="R-squared", value=f"{round(r2, 2)}%", delta=f"{round(r2 - 91, 2)}%")
    col3.metric(label="MAE", value=round(mae, 2), delta=f"{round(mae - 0.01 * df_available.close.mean(), 2)}", delta_color="inverse")
    col4.metric(label="MAPE", value=f"{round(mape, 2)}%", delta=f"{round(mape - 1.2, 2)}%", delta_color="inverse")

    preds = model.predict(x_test)
    y_new = model.predict(x_forecast)
    # st.session_state.model = model
    df_w_visualization = df_preprocessed[['close', 'isFuture']].copy()
    y_pred_all = model.predict(X)
    print("Length of y_pred_all:", len(y_pred_all))
    print("Number of rows with 'isFuture' False:", len(df_w_visualization[df_w_visualization['isFuture'] == False]))
    # Create a new column 'y_pred_new' and initialize it with NaN
    df_w_visualization['Predicted Close'] = np.nan
    # Update 'y_pred_new' column for rows where 'isFuture' is True
    df_w_visualization.loc[:df_available.shape[0],'Predicted Close'] = y_pred_all
    
    # Replace 'close' values with predicted values when 'isFuture' is True
    df_w_visualization.loc[df_available.shape[0]:,'Predicted Close'] = y_new
    
    # Create an interactive line chart
    # st.session_state["df_w_visualization"] = df_w_visualization 
    fig = create_predicted_chart(st.session_state["df_w_visualization"])
    st.plotly_chart(fig, use_container_width=True)

    # Show the plot using Streamlit
    prediction_result = df_w_visualization.copy()
    prediction_result['ticker'] = option
    prediction_result['RMSE'] = rmse
    prediction_result['MAE'] = mae
    prediction_result['r-squared'] = r2
    # st.session_state["prediction_result"] = prediction_result
    # Create a download button for Excel
    col1, col2 = st.columns([3,1])
    col1.dataframe(st.session_state["prediction_result"][-num:])
    excel_file_path = 'prediction_result.xlsx'  # Adjust the file path as needed
    prediction_result.to_excel(excel_file_path)
    col2.download_button(
        label="Download data as Excel",
        data=open(excel_file_path, 'rb').read(),
                        file_name=f"{option}_prediction_{num}_days.xlsx",
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',)
    
if __name__ == '__main__':
    main()
