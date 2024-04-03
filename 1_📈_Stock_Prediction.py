import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from vnstock import * #import all functions
from adding_features import technical_analysis_indicator, take_news_parameter, add_new_working_days
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
import plotly.graph_objects as go
from st_functions import load_css, ColourWidgetText
from plotly.subplots import make_subplots

st.set_page_config(page_title="Stock Prediction App", page_icon="📈", layout="wide")


# Apply custom CSS to change the color of st.sidebar.info
load_css("main.css")

st.title('Stock Price Prediction')
st.sidebar.info('Welcome to the Stock Price Prediction App. Choose your options below')
st.sidebar.markdown("Created and designed by: <a style='color: #ff9421;' href='https://www.linkedin.com/in/quang-nguyen-4b6a52287/'>Dylan Nguyen</a>", unsafe_allow_html=True)

def main():
    sidebar_option = st.sidebar.selectbox('Make a choice', ['Financial Dashboard','Visualize','Recent Data', 'Predict'])
    if sidebar_option == 'Financial Dashboard':
        if index_option == 'stock':
            financial_dashboard(data, option)
        else:
            financial_dashboard_index(data, option)
    elif sidebar_option == 'Visualize':
        tech_indicators(data, option)
        dataframe()
    elif sidebar_option == 'Predict':
        predict()
    else:
        st.warning("not a function")


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


class SessionState(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

# Function to get unique hash
def get_hash(text):
    return hash(','.join(sorted(text)))

listing_stocks = listing_companies(True)
listing_stocks = listing_stocks.copy()['ticker'].to_list()
listing_indexes = ["VNINDEX", "VN30", "HNX30"]
# Single select dropdown
default_ix = listing_stocks.index("VHM")
index_option = st.radio("Select stock or index:", ['index', 'stock'], horizontal=True, index=0)
if index_option == "stock":
    # Initialize SessionState
    session_state = SessionState(hash=0)

    # Input field for users to enter their pinned options
    pinned_options = st.multiselect('Select options to pin', listing_stocks)

    # Save pinned options to session state
    if pinned_options:
        session_state.hash = get_hash(pinned_options)
        session_state.pinned_options = pinned_options

# Create a selectbox with pinned options
try:
    if hasattr(session_state, 'pinned_options'):
        pinned_options_with_icon = [f'📌 {option}' for option in session_state.pinned_options]
        option = st.selectbox("Select a stock:", options=pinned_options_with_icon + [option for option in listing_stocks if option not in session_state.pinned_options], index=default_ix)
    else:
        option = st.selectbox("Select a stock:", options=listing_stocks, index=default_ix)
except:
    option = st.selectbox("Select an index:", listing_indexes, index=1)    

# Remove the icon before passing the option to the download_data function
option = option.split('📌 ')[-1] if '📌' in option else option
option = option.upper()
today = datetime.date.today()
# today = datetime.date.today() - timedelta(days=1)
# define date
duration = st.sidebar.number_input('Enter the duration', value=1000)
before = datetime.date(2019, 1, 1)
start_date = st.sidebar.date_input('Start Date', value=before)
end_date = st.sidebar.date_input('End date', today)

sp500_data = take_sp500(start_date, end_date)
vnindex_data = take_vnindex(start_date, end_date)

if st.sidebar.button('Send'):
    if start_date < end_date:
        st.sidebar.success('Start date: `%s`\n\nEnd date: `%s`' %(start_date, end_date))
        download_data(option, start_date, end_date)
    else:
        st.sidebar.error('Error: End date must fall after start date')

data = download_data(option, start_date, end_date, index_option)
data['return'] = data['close'].pct_change() * 100
scaler = StandardScaler()

def add_macd_to_chart(data, fig, row, col):
    macd_trace = go.Scatter(
        x=data['time'],
        y=data['macd'],
        mode='lines',
        name="MACD"
    )
    macd_signal_trace = go.Scatter(
        x=data['time'],
        y=data['macd_signal_line'],
        mode='lines',
        name="MACD signal line"
    )
    macd_hist = go.Bar(
        x=data['time'],
        y=data['macd_histogram'],
        name="MACD hist"
    )
    fig.add_trace(macd_trace, row=row,col=col)
    fig.add_trace(macd_signal_trace, row=row,col=col)
    fig.add_trace(macd_hist, row=row,col=col)

def add_rsi_to_chart(data, fig, row, col):
    rsi_trace = go.Scatter(
        x=data['time'],
        y=data['rsi'],
        mode='lines',
        name="RSI"
    )
    rsi_up_trace = go.Scatter(
        x=data['time'],
        y=[70]*len(data["time"]),
        mode='lines',
        name="RSI",
        line = dict(color = 'red',
            width = 0.5,
            dash = 'dot')
    )
    rsi_down_trace = go.Scatter(
        x=data['time'],
        y=[30]*len(data['time']),
        mode='lines',
        name="RSI",
        line = dict(color = 'red',
            width = 0.5,
            dash = 'dot')
    )
    fig.add_trace(rsi_trace, row=row,col=col)
    fig.add_trace(rsi_up_trace, row=row,col=col)
    fig.add_trace(rsi_down_trace, row=row,col=col)

def add_smi_to_chart(data, fig, row, col):
    # tech_ana_converter.calculate_smi()
    # tech_ana_converter.calculate_chaikin()
    smi_trace = go.Scatter(
        x=data['time'],
        y=data['smi_raw'],
        mode='lines',
        name="SMI"
    )
    signal_trace = go.Scatter(
        x=data['time'],
        y=data['smi_signal_line'],
        mode='lines',
        name="SMI signal line"
    )
    smi_up_trace = go.Scatter(
        x=data['time'],
        y=[70]*len(data["time"]),
        mode='lines',
        name="overbought sign",
        line = dict(color = 'red',
            width = 0.5,
            dash = 'dot')
    )
    smi_down_trace = go.Scatter(
        x=data['time'],
        y=[30]*len(data['time']),
        mode='lines',
        name="oversold sign",
        line = dict(color = 'red',
            width = 0.5,
            dash = 'dot')
    )
    fig.add_trace(smi_trace, row=row,col=col)
    fig.add_trace(signal_trace, row=row,col=col)
    fig.add_trace(smi_up_trace, row=row,col=col)
    fig.add_trace(smi_down_trace, row=row,col=col)

def add_cci_to_chart(data, fig, row, col):
    # tech_ana_converter.calculate_smi()
    # tech_ana_converter.calculate_chaikin()
    cci_trace = go.Scatter(
        x=data['time'],
        y=data['cci'],
        mode='lines',
        name="CCI (Commodity Channel Index)"
    )

    cci_up_trace = go.Scatter(
        x=data['time'],
        y=[100]*len(data["time"]),
        mode='lines',
        name="overbought sign",
        line = dict(color = 'red',
            width = 0.5,
            dash = 'dot')
    )
    cci_down_trace = go.Scatter(
        x=data['time'],
        y=[-100]*len(data['time']),
        mode='lines',
        name="oversold sign",
        line = dict(color = 'red',
            width = 0.5,
            dash = 'dot')
    )
    fig.add_trace(cci_trace, row=row,col=col)
    fig.add_trace(cci_up_trace, row=row,col=col)
    fig.add_trace(cci_down_trace, row=row,col=col)

def create_overview_chart(data, time_option=-1):
    if time_option==-1:
        df_filtered = data.copy()
    else:
        df_filtered = data[-time_option:]
        
    fig = candlestick_chart(df_filtered,  show_volume=False, reference_period=200, figure_size=(10, 5), 
                        title=f'{option} - Candlestick Chart with MA and Volume', x_label='Date', y_label='Price', 
                        colors=('lightgray', 'gray'), reference_colors=('black', 'blue'))
    fig.update_layout(
        hovermode='x',
        yaxis=dict(autorange = True,
                fixedrange= False),
    )
    return fig

def financial_report(stock, type):
    if type == "Balance Sheet":
        balance_sheet = financial_flow(symbol=stock, report_type='balancesheet', report_range='quarterly')
        balance_sheet.drop(['ticker'], axis=1,inplace=True)
        return balance_sheet.T
          
    elif type == "Income Statement":
        income_statement = financial_flow(symbol=stock, report_type='incomestatement', report_range='quarterly')
        income_statement.drop(['ticker'], axis=1,inplace=True)
        return income_statement.T
        
    elif type == "Cashflow Statement":
        cashflow = financial_flow(symbol=stock, report_type='cashflow', report_range='quarterly')
        cashflow.drop(['ticker'], axis=1,inplace=True)
        return cashflow.T

def financial_dashboard_index(data, index):
    st.header("Index Information", divider="rainbow")
    latest_time = data['time'].iloc[-1]
    updated_close = data['close'].iloc[-1]
    return_percent = data['return'].iloc[-1]
    col1, col_temp = st.columns([1,5])

    if return_percent < 0:
        color = "red"
    elif return_percent == 0:
        color = "blue"
    else:
        color = "green"

    with col1:
        st.subheader(f":red[{index}]", divider='blue')
        st.metric(
            label=f"***Updated close {latest_time}***",
            value=f"{updated_close:,.0f}",
            delta=f"{return_percent:,.2f}%"
        )
        # ColourWidgetText('Metric1', '#00B0F0')  # colour only metric text
        st.metric(
            label=f"***Volume {latest_time}***",
            value= f"{data['volume'].iloc[-1]:,.0f}",
            delta= f"{data['volume'].iloc[-1]-data['volume'].iloc[-2]:,.0f}",
            delta_color="off"
        )
        # Display "High" value with color
        st.write(
            f"***High:*** <span style='color: green;'>{data['high'].iloc[-1]:,.0f}</span>",
            unsafe_allow_html=True
        )

        # Display "Low" value with color
        st.write(
            f"***Low:*** <span style='color: red;'>{data['low'].iloc[-1]:,.0f}</span>",
            unsafe_allow_html=True
        )
        st.write(
            f"***Open:*** <span style='color: blue;'>{data['open'].iloc[-1]:,.0f}</span>",
            unsafe_allow_html=True
        )
    with col_temp:
        st.subheader(f":orange[Historical price]", divider='blue')
        time_period_option = st.radio("Select Time Period", ["1 week", "2 weeks", "1 month", "1 year", "all"], horizontal=True, index=4, label_visibility='hidden')
        if time_period_option == "1 week":
        # Create the line chart based on user input
            fig = create_overview_chart(data, 5)
            st.plotly_chart(fig, use_container_width=True)
        elif time_period_option == "2 weeks":
            fig = create_overview_chart(data, 10)
            st.plotly_chart(fig, use_container_width=True)
        elif time_period_option == "1 month":
            fig = create_overview_chart(data, 22)
            st.plotly_chart(fig, use_container_width=True)
        elif time_period_option == "1 year":
            fig = create_overview_chart(data, 225)
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig = create_overview_chart(data)
            st.plotly_chart(fig, use_container_width=True)
    ColourWidgetText(f"{updated_close:,.0f}", color)       # colour metric value
    st.divider()

def financial_dashboard(data, stock):
    st.header("Company Information", divider="rainbow")
    
    company_info = company_profile(stock)
    # company_fundamentals = company_fundamental_ratio(symbol=stock, mode='simplify', missing_pct=0.8).T
    company_news_title = company_news(symbol=stock, page_size=1000, page=0)
    company_fin_ratio = financial_ratio(stock, 'yearly').T
    col1, col_temp = st.columns([1,5])
    latest_time = data['time'].iloc[-1].strftime('%d/%m/%y')
    updated_close = data['close'].iloc[-1]
    return_percent = data['return'].iloc[-1]
    
    roe = company_fin_ratio['roe'].iloc[0]
    roa = company_fin_ratio['roa'].iloc[0]
    eps = company_fin_ratio['earningPerShare'].iloc[0]
    priceToEarning = company_fin_ratio['priceToEarning'].iloc[0]
    try:
        bookValuePerShare = company_fin_ratio['bookValuePerShare'].iloc[0]
        badDebtPercentage = company_fin_ratio['badDebtPercentage'].iloc[0]
    except:
        pass
    financialLeverage = company_fin_ratio['assetOnEquity'].iloc[0]

    if return_percent < 0:
        color = "red"
    elif return_percent == 0:
        color = "blue"
    else:
        color = "green"

    with col1:
        st.subheader(f":red[{stock}]", divider='blue')
        st.metric(
            label=f"***Updated close {latest_time}***",
            value=f"{updated_close:,.0f}",
            delta=f"{return_percent:,.2f}%"
        )
        # ColourWidgetText('Metric1', '#00B0F0')  # colour only metric text
        st.metric(
            label=f"***Volume {latest_time}***",
            value= f"{data['volume'].iloc[-1]:,.0f}",
            delta= f"{data['volume'].iloc[-1]-data['volume'].iloc[-2]:,.0f}",
            delta_color="off"
        )
        # Display "High" value with color
        st.write(
            f"***High:*** <span style='color: green;'>{data['high'].iloc[-1]:,.0f}</span>",
            unsafe_allow_html=True
        )

        # Display "Low" value with color
        st.write(
            f"***Low:*** <span style='color: red;'>{data['low'].iloc[-1]:,.0f}</span>",
            unsafe_allow_html=True
        )
        st.write(
            f"***Open:*** <span style='color: blue;'>{data['open'].iloc[-1]:,.0f}</span>",
            unsafe_allow_html=True
        )
        st.write(
            f"**EPS:** <span style='color: #652f2f;'>{eps:,.0f}</span>",
            unsafe_allow_html=True
        )
        st.write(
            f"**ROA:** <span style='color: #652f2f;'>{roa:,.2f}</span>",
            unsafe_allow_html=True
        )
        st.write(
            f"**ROE:** <span style='color: #652f2f;'>{roe:,.2f}</span>",
            unsafe_allow_html=True
        )
        st.write(
            f"**P/E:** <span style='color: #652f2f;'>{priceToEarning:,.2f}</span>",
            unsafe_allow_html=True
        )
        st.write(
            f"**Leverage ratio:** <span style='color: #652f2f;'>{financialLeverage:,.2f}</span>",
            unsafe_allow_html=True
        )
        try:
            st.write(
                f"**BVPS:** <span style='color: #652f2f;'>{bookValuePerShare:,.2f}</span>",
                unsafe_allow_html=True
            )
            st.write(
                f"**badDebtPercentage:** <span style='color: #652f2f;'>{badDebtPercentage:,.2f}</span>",
                unsafe_allow_html=True
            )
        except:
            pass
    
    ColourWidgetText(f"{updated_close:,.0f}", color)       # colour metric value
    with col_temp:
        col2,col3 = st.columns([2, 3])
        with col2:
            st.markdown("<h3 style='color: #FF5733;'>Company Name</h3>", unsafe_allow_html=True)
            st.write(company_info['companyName'][0])
            st.markdown("<h3 style='color: #C70039;'>Business Strategies</h3>", unsafe_allow_html=True)
            st.write(company_info['businessStrategies'][0])
            
        with col3:
            st.markdown("<h3 style='color: #C70039;'>Company Profile</h3>", unsafe_allow_html=True)
            st.write(company_info['companyProfile'][0])

            st.markdown("<h3 style='color: #900C3F;'>Business Risk</h3>", unsafe_allow_html=True)
            st.write(company_info['businessRisk'][0])
        
        st.divider()
        with st.container():
            st.subheader("Financial Ratios")
            company_fin_ratio.drop(['ticker'], axis=1,inplace=True)
            st.dataframe(company_fin_ratio,use_container_width=True,height=190)
    st.divider()
    
    kpi1,kpi2,kpi3 = st.columns([11,1,11])
    kpi1.header("News and Events Related")

    kpi1.dataframe(company_news_title[['title','publishDate']],use_container_width=True, height=600,hide_index=True)
    
    kpi3.header("Historical exchange")
    time_period_option = kpi3.radio("Select Time Period", ["1 week", "2 weeks", "1 month", "1 year", "all"], horizontal=True, index=2, label_visibility='hidden')
    if time_period_option == "1 week":
    # Create the line chart based on user input
        fig = create_overview_chart(data, 5)
        kpi3.plotly_chart(fig, use_container_width=True)
    elif time_period_option == "2 weeks":
        fig = create_overview_chart(data, 10)
        kpi3.plotly_chart(fig, use_container_width=True)
    elif time_period_option == "1 month":
        fig = create_overview_chart(data, 22)
        kpi3.plotly_chart(fig, use_container_width=True)
    elif time_period_option == "1 year":
        fig = create_overview_chart(data, 225)
        kpi3.plotly_chart(fig, use_container_width=True)
    else:
        fig = create_overview_chart(data)
        kpi3.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Financial Statement
    st.header("Financial Statement")
    report_option = st.selectbox("Choose an income statement", label_visibility="hidden", options=["Balance Sheet", "Income Statement", "Cashflow Statement"])
    report = financial_report(stock, report_option)
    report.reset_index(inplace=True)
    st.dataframe(report,use_container_width=True, height=700, hide_index=True)

def tech_indicators(data, stock):
    st.header('Technical Indicators')
    option = st.radio('Choose a Technical Indicator to Visualize', ['Candlestick chart', 'BB', 'SMA'], horizontal=True)
    
    tech_ana_converter = technical_analysis_indicator(data)
    # MACD
    tech_ana_converter.calculate_macd()
    tech_ana_converter.calculate_obv()
    # RSI
    tech_ana_converter.calculate_rsi()
    tech_ana_converter.calculate_cci()
    tech_ana_converter.calculate_smi()
    tech_ana_converter.calculate_chaikin()
    df = tech_ana_converter.get_data()
    
    if option == 'Candlestick chart':
        st.warning("No more than 3 toggles can be turned on at the same time!")
        
        on_macd = st.toggle('MACD')
        on_volume = st.toggle('Show volume')
        on_rsi = st.toggle('Show RSI')
        on_cci = st.toggle('Show CCI (Commodity Channel Index)')
        on_smi = st.toggle('Show SMI (Stochastic Momentum Indicator)')
        on_obv = st.toggle('Show OBV (On-Balance Volume)')
        st.write("Candlestick chart")
        # Create the candlestick chart
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, row_heights=[0.55, 0.15, 0.15, 0.15])
        candlestick_trace = go.Candlestick(
            x=df['time'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Candlestick',
        )
        fig.add_trace(candlestick_trace, row=1, col=1)
        count_button = sum(1 for var in {on_macd, on_rsi, on_cci, on_smi, on_volume, on_obv} if var)
        i = count_button
        if on_macd:
            add_macd_to_chart(df, fig, row=i+1, col=1)
            i+=1
        if on_smi:
            # on_rsi=False
            add_smi_to_chart(df, fig, row=i+1, col=1)
            i+=1
        if on_cci:
            add_cci_to_chart(df, fig, row=i+1, col=1)
            i+=1
        if on_rsi:
            # on_smi=False
            fig.add_trace(go.Scatter(x=df['time'], y=df['volume'], name="Volume"), row=i+1, col=1)
            i+=1
        if on_volume:
            # on_obv=False
            fig.add_trace(go.Bar(x=df['time'], y=df['volume'], name="Volume"), row=i+1, col=1)
            i+=1
        if on_obv:
            # on_volume=False
            fig.add_trace(go.Bar(x=df['time'], y=df['obv'], name="On-Balance Volume"), row=i+1, col=1)
            i+=1
        print(i)
        if i>=4:
            st.warning("No more than 3 toggles can be turned on at the same time!")
        # Update layout
        fig.update_layout(
            title=f'{stock} - Candlestick Chart',
            xaxis_title='Date',
            yaxis_title="Price",
            hovermode='x',
            yaxis=dict(autorange = True,
                    fixedrange= False),
            font=dict(color='black', size=14),  # Text color
            width=13 * 100,  # Convert short form width to full width
            height=12 * 100  # Convert short form height to full height
        )
        fig.update_xaxes(
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
            row=1, col=1
        )
        fig.update_layout(xaxis_rangeslider_visible=False)
        # df = df.drop(['lowest_low', 'highest_high'], axis=1)
        st.plotly_chart(fig, use_container_width=True)
    elif option == 'BB':
        st.write('BollingerBands')
        st.warning("No more than 3 toggles can be turned on at the same time!")
        
        on_macd = st.toggle('MACD')
        on_rsi = st.toggle('Show RSI')
        on_cci = st.toggle('Show CCI (Commodity Channel Index)')
        on_smi = st.toggle('Show SMI (Stochastic Momentum Indicator)')
        on_volume = st.toggle('Show volume')
        on_obv = st.toggle('Show OBV (On-Balance Volume)')
        st.write("Candlestick chart")
        # Create the candlestick chart
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, row_heights=[0.55, 0.15, 0.15, 0.15])
        df = bollinger_bands(df)
        candlestick_trace = go.Candlestick(
            x=df['time'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Candlestick',
        )

        fig.add_trace(candlestick_trace, row=1, col=1)

        # Create the Bollinger Bands traces
        upper_band_trace = go.Scatter(
            x=df['time'],
            y=df['upper_band'],
            mode='lines',
            line=dict(color='gray'),
            name='Upper Bollinger Band',
        )

        middle_band_trace = go.Scatter(
            x=df['time'],
            y=df['middle_band'],
            mode='lines',
            line=dict(color='orange'),
            name='Middle Bollinger Band',
        )

        lower_band_trace = go.Scatter(
            x=df['time'],
            y=df['lower_band'],
            mode='lines',
            line=dict(color='gray'),
            name='Lower Bollinger Band',
        )

        fig.add_trace(upper_band_trace,row=1,col=1)
        fig.add_trace(middle_band_trace,row=1,col=1)
        fig.add_trace(lower_band_trace,row=1,col=1)

        count_button = sum(1 for var in {on_macd, on_rsi, on_cci, on_smi, on_volume, on_obv} if var)
        i = count_button
        if on_macd:
            add_macd_to_chart(df, fig, row=i+1, col=1)
            i+=1
        if on_smi:
            # on_rsi=False
            add_smi_to_chart(df, fig, row=i+1, col=1)
            i+=1
        if on_cci:
            add_cci_to_chart(df, fig, row=i+1, col=1)
            i+=1
        if on_rsi:
            # on_smi=False
            add_rsi_to_chart(df, fig, row=i+1, col=1)
            i+=1
        if on_volume:
            # on_obv=False
            fig.add_trace(go.Bar(x=df['time'], y=df['volume'], name="Volume"), row=i+1, col=1)
            i+=1
        if on_obv:
            # on_volume=False
            fig.add_trace(go.Bar(x=df['time'], y=df['obv'], name="On-Balance Volume"), row=i+1, col=1)
            i+=1
        if i>4:
            st.error("No more than 3 toggles can be turned on at the same time!")
        print(count_button)
            
                # Update layout
        fig.update_layout(
            title=f'{stock} - Candlestick BB Chart',
            xaxis_title='Date',
            yaxis_title="Price",
            hovermode='x',
            yaxis=dict(autorange = True,
                    fixedrange= False),
            font=dict(color='black', size=14),  # Text color
            width=13 * 100,  # Convert short form width to full width
            height=12 * 100  # Convert short form height to full height
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
            ),
            row=1, col=1
        )
        fig.update_layout(xaxis_rangeslider_visible=False)
        
        df = df.drop(['upper_band', 'lower_band','middle_band'], axis=1)
        st.plotly_chart(fig, use_container_width=True)
    elif option == 'SMA':
        st.write("Candlestick chart with SMA")
        ma_period = st.slider("SMA of how many days?",1,200,value=1,step=1)

        # Create the candlestick chart
        fig = candlestick_chart(df, ma_periods=[ma_period], show_volume=False, reference_period=300,
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
        df = df.drop(['20-day MA', '200-day MA','lowest_low', 'highest_high'], axis=1)
        # Display the chart in the Streamlit app with responsive sizing
        st.plotly_chart(fig, use_container_width=True)

def dataframe():
    st.header('Recent Data')
    st.dataframe(data.tail(10))

def predict():
    model = st.radio('Choose a model', ['RandomForestRegressor', 'ExtraTreesRegressor', 'LSTM', 'XGBoostRegressor', 'CatBoostRegressor'],index=3, horizontal=True)
    num = st.number_input('How many days forecast?', value=5, max_value=10)
    num = int(num)
    
    if "df_w_visualization" not in st.session_state:
        st.session_state["df_w_visualization"] = pd.DataFrame()
    if f"model_{option}_{num}" not in st.session_state:
        st.session_state[f"model_{option}_{num}"] = None
    if "prediction_result" not in st.session_state:
        st.session_state["prediction_result"] = pd.DataFrame()
    
    if st.button('Predict'):
        if model == 'RandomForestRegressor':
            engine = RandomForestRegressor(n_estimators=150, criterion="absolute_error")
        elif model == 'ExtraTreesRegressor':
            engine = ExtraTreesRegressor()
        elif model == 'KNeighborsRegressor':
            engine = KNeighborsRegressor()
        elif model == 'LSTM':
            engine = ExtraTreesRegressor()
        elif model == 'CatBoostRegressor':
            engine = CatBoostRegressor(n_estimators=600, loss_function='RMSE')
        else:
            engine = XGBRegressor(base_score=0.5, booster='gblinear',    
                           n_estimators=900,
                           objective='reg:squarederror',
                           learning_rate=0.03)
        model_engine(engine, num)

def model_preprocess(df, num):
    print(df)
    # Display the updated DataFrame
    df, next_working_days = add_new_working_days(df, num)
    # shifting the closing price based on number of days forecast
    # Calculate the 'return' column
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

def model_engine(model, num):
    # getting only the closing price
    df = data[['time','close','open','high','low','return']]
    df_preprocessed = model_preprocess(df, num)
    
    # scaling the data
    df_available = df_preprocessed[df_preprocessed['isFuture']==False].copy()
    # if it passed 14h30 then apply close price of today, else not
    
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
        time.sleep(0.0000001)
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
    
    # Create a figure
    fig = go.Figure()

    # Add actual and predicted close prices for training data
    fig.add_trace(go.Scatter(x=df_available.index[:int(len(df_available)*0.8)], y=y_train, mode='lines', name='Actual Close (Training)'))
    fig.add_trace(go.Scatter(x=df_available.index[:int(len(df_available)*0.8)], y=model.predict(x_train), mode='lines', name='Predicted Close (Training)'))

    # Add actual and predicted close prices for testing data
    fig.add_trace(go.Scatter(x=df_available.index[int(len(df_available)*0.8):], y=y_test, mode='lines', name='Actual Close (Testing)'))
    fig.add_trace(go.Scatter(x=df_available.index[int(len(df_available)*0.8):], y=preds, mode='lines', name='Predicted Close (Testing)'))

    # Update layout
    fig.update_layout(title='Actual vs Predicted Close Prices Training and Testing dataset',
                    xaxis_title='Time',
                    yaxis_title='Close Price')

    # Show the plot
    st.plotly_chart(fig, use_container_width=True)
    # st.write(f'r2_score: {r2:.2f}% \
    #         \nMAE: {mae:.2f}\
    #         \nRMSE: {rmse:.2f}\
    #         \nMAPE: {mape:.2f}%')
    # predicting stock price based on the number of days
    model.fit(X, y)
    y_new = model.predict(x_forecast)
    # st.session_state.model = model
    df_w_visualization = df_preprocessed[['close', 'isFuture']].copy()
    y_pred_all = model.predict(X)
    print("Length of y_pred_all:", len(y_pred_all))
    print("Number of rows with 'isFuture' False:", len(df_w_visualization[df_w_visualization['isFuture'] == False]))
    # Create a new column 'y_pred_new' and initialize it with NaN
    df_w_visualization['Predicted Close'] = np.nan
    # Update 'y_pred_new' column for rows where 'isFuture' is True
    df_w_visualization['Predicted Close'].iloc[:df_available.shape[0]] = y_pred_all
    
    # Replace 'close' values with predicted values when 'isFuture' is True
    df_w_visualization['Predicted Close'].iloc[df_available.shape[0]:] = y_new
    
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
    col1, col2 = st.columns([3,1])
    col1.dataframe(st.session_state["prediction_result"][-num:])
    excel_file_path = 'prediction_result.xlsx'  # Adjust the file path as needed
    prediction_result.to_excel(excel_file_path)
    col2.download_button(
        label="Download data as Excel",
        data=open(excel_file_path, 'rb').read(),
                        file_name=f"{option}_prediction_{num}_days.xlsx",
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',)


# def existed_model_engine(model, num):
#     # getting only the closing price
#     df = data[['time','close','open','high','low','return']]
#     df_preprocessed = model_preprocess(df, num)
    
#     # scaling the data
#     df_available = df_preprocessed[df_preprocessed['isFuture']==False].copy()
#     # if it passed 14h30 then apply close price of today, else not
#     current_hour = datetime.datetime.now().hour
#     current_minute = datetime.datetime.now().minute
#     if (current_hour < 14 and current_minute < 30)  and (current_hour > 9 and current_minute>15):
#         df_available = df_available[:-1]
#         print(df_available.tail())
#         print(current_hour)
#         print(current_minute)
    
#     TARGET = 'close'
#     X = df_available.drop(columns=[TARGET, 'isFuture']).values
#     # X = df.drop(['time','isFuture','close'], axis=1).values
#     X = scaler.fit_transform(X)
#     # getting the preds column
#     y = df_available[TARGET].values
#     # storing the last num_days data
#     df_new = df_preprocessed[df_preprocessed['isFuture']==True].copy()
#     x_forecast = df_new.drop(['isFuture',TARGET], axis=1).values
#     x_forecast = scaler.transform(x_forecast)

#     # predicting stock price based on the number of days
#     x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=.2, shuffle=False)
    
#     st.success("Model built successful!")
#     preds = model.predict(x_test)
#     col1, col2, col3, col4 = st.columns([1,1,1,1])
#     rmse = np.sqrt(mean_squared_error(y_test, preds))
#     r2 = r2_score(y_test, preds)*100
#     mae = mean_absolute_error(y_test, preds)
#     mape =  np.mean(np.abs((y_test - preds) / y_test))*100
    
#     col1.metric(label="RMSE", value=round(rmse, 2), delta=f"{round(rmse - 0.01 * df_available.close.mean(), 2)}", delta_color="inverse")
#     col2.metric(label="R-squared", value=f"{round(r2, 2)}%", delta=f"{round(r2 - 91, 2)}%")
#     col3.metric(label="MAE", value=round(mae, 2), delta=f"{round(mae - 0.01 * df_available.close.mean(), 2)}", delta_color="inverse")
#     col4.metric(label="MAPE", value=f"{round(mape, 2)}%", delta=f"{round(mape - 1.2, 2)}%", delta_color="inverse")

#     preds = model.predict(x_test)
#     y_new = model.predict(x_forecast)
#     # st.session_state.model = model
#     df_w_visualization = df_preprocessed[['close', 'isFuture']].copy()
#     y_pred_all = model.predict(X)
#     print("Length of y_pred_all:", len(y_pred_all))
#     print("Number of rows with 'isFuture' False:", len(df_w_visualization[df_w_visualization['isFuture'] == False]))
#     # Create a new column 'y_pred_new' and initialize it with NaN
#     df_w_visualization['Predicted Close'] = np.nan
#     # Update 'y_pred_new' column for rows where 'isFuture' is True
#     df_w_visualization['Predicted Close'].iloc[:df_available.shape[0]] = y_pred_all
    
#     # Replace 'close' values with predicted values when 'isFuture' is True
#     df_w_visualization['Predicted Close'].iloc[df_available.shape[0]:] = y_new
    
#     # Create an interactive line chart
#     # st.session_state["df_w_visualization"] = df_w_visualization 
#     fig = create_predicted_chart(st.session_state["df_w_visualization"])
#     st.plotly_chart(fig, use_container_width=True)

#     # Show the plot using Streamlit
#     prediction_result = df_w_visualization.copy()
#     prediction_result['ticker'] = option
#     prediction_result['RMSE'] = rmse
#     prediction_result['MAE'] = mae
#     prediction_result['r-squared'] = r2
#     # st.session_state["prediction_result"] = prediction_result
#     # Create a download button for Excel
#     col1, col2 = st.columns([3,1])
#     col1.dataframe(st.session_state["prediction_result"][-num:])
#     excel_file_path = 'prediction_result.xlsx'  # Adjust the file path as needed
#     prediction_result.to_excel(excel_file_path)
#     col2.download_button(
#         label="Download data as Excel",
#         data=open(excel_file_path, 'rb').read(),
#                         file_name=f"{option}_prediction_{num}_days.xlsx",
#                         mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',)
    
if __name__ == '__main__':
    main()
