import streamlit as st
from st_functions import st_button, load_css
from PIL import Image
from vnstock import * #import all functions
import numpy as np 

st.set_page_config(page_title="Profile", page_icon="📈", layout='wide')
load_css("main.css")

st.title('Stocks Comparison')
st.sidebar.info('Welcome to the Stock Price Prediction App. Choose your options below')
st.sidebar.info("Created and designed by [Dylan Nguyen](https://www.linkedin.com/in/quang-nguyen-4b6a52287/)")

def main():
    sidebar_option = st.sidebar.selectbox('Make a choice', ['Stock Comparison', 'Stock Same Industry Comparison'])
    if sidebar_option == 'Stock Comparison':
        stock_comparison()
    elif sidebar_option == 'Stock Same Industry Comparison':
        stock_same_ind_comparison()

listing_stocks = listing_companies(True)
listing_stocks = listing_stocks.copy()['ticker'].to_list()

def stock_comparison():

    options = st.multiselect("Select some tickers to compare:", listing_stocks, ['FPT', 'VNM', 'TCB','VND'])
    stocks = ','.join(options)
    stocks_sel = price_board(stocks)
    # Set index to the first column and drop the original index
    stocks_sel.set_index(stocks_sel.columns[0], inplace=True, drop=True)
    stocks_sel = stocks_sel.T

    col1, col2 = st.columns([1,1])

    col1.markdown("<h3 style='color: #FF5733;'>Technical Indicators Comparison</h3>", unsafe_allow_html=True)
    col1.dataframe(stocks_sel, use_container_width=True,height=700)

    col2.markdown("<h3 style='color: #C70039;'>Fundamental Indicators Comparison</h3>", unsafe_allow_html=True)
    col2.dataframe(stock_ls_analysis(stocks, lang='vi'), use_container_width=True,height=700)

    st.header("Same Industry Comparison")
    default_ix = listing_stocks.index("VHM")
    option = st.selectbox("Select stock to compare in the same industry:", listing_stocks, default_ix)
    st.dataframe(industry_analysis(option, lang='vi'))


def stock_same_ind_comparison():
    default_ix = listing_stocks.index("VHM")
    option = st.selectbox("Select stock to compare in the same industry:", listing_stocks, default_ix)
    st.dataframe(industry_analysis(option, lang='vi'))


if __name__ == '__main__':
    main()