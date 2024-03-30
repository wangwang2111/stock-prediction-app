import pandas as pd
# import pandas_ta as ta
import holidays

# Assuming df is your DataFrame with the 'return' column added
# ...
def true_range(data):
    high_low = data['High'] - data['Low']
    high_close_prev = abs(data['High'] - data['Close'].shift(1))
    low_close_prev = abs(data['Low'] - data['Close'].shift(1))
    
    true_range = pd.DataFrame({'HL': high_low, 'HC': high_close_prev, 'LC': low_close_prev}).max(axis=1)

    return true_range

# Add a custom method for calculating MAD
def mad_custom(arr):
    """Calculate Mean Absolute Deviation (MAD)"""
    return (arr - arr.mean()).abs().mean()

pd.Series.mad_custom = mad_custom  # Adding custom method to pandas Series

# Function to calculate RSI
class technical_analysis_indicator():
    def __init__(self,data, return_="return"):
        self.original_data = data
        self.data = self.original_data.copy()
        self.return_ = return_ 
    
    
    def calculate_rsi(self, period=14):
        # Calculate daily price changes
        self.data['delta'] = self.data[self.return_].diff()

        # Calculate gains and losses
        self.data['gain'] = self.data['delta'].apply(lambda x: x if x > 0 else 0)
        self.data['loss'] = self.data['delta'].apply(lambda x: abs(x) if x < 0 else 0)

        # Calculate average gains and losses
        avg_gain = self.data['gain'].rolling(window=period, min_periods=1).mean()
        avg_loss = self.data['loss'].rolling(window=period, min_periods=1).mean()

        # Calculate relative strength (RS)
        rs = avg_gain / avg_loss

        # Calculate RSI
        self.data['rsi'] = 100 - (100 / (1 + rs))

        # Drop intermediate columns
        self.data.drop(['delta', 'gain', 'loss'], axis=1, inplace=True)
        return self.data


    def calculate_sma(self, period=7):
        """
        Calculate Simple Moving Average (SMA) for a specified column in a DataFrame.

        Parameters:
        - period: The window size for the SMA (default is 7 days).

        Returns:
        - DataFrame with the original data and an additional column for the SMA.
        """
        self.data['sma'+str(period)] = self.data[self.return_].rolling(window=period, min_periods=period).mean()
        return self.data
    
    
    def calculate_ema(self, period=7):
        """
        Calculate Exponential Moving Average (EMA) for a specified column in a DataFrame.

        Parameters:
        - period: The period parameter for EMA (default is 7 days).

        Returns:
        - DataFrame with the original data and an additional column for the EMA.
        """
        self.data['ema'+str(period)] = self.data[self.return_].ewm(span=period, adjust=False).mean()
        return self.data


    def calculate_macd(self, short_period=12, long_period=26, signal_period=9):
        """
        Calculate Moving Average Convergence Divergence (MACD) for a specified column in a DataFrame.

        Parameters:
        - short_period: Short-term window for the exponential moving average (default is 12 days).
        - long_period: Long-term window for the exponential moving average (default is 26 days).
        - signal_period: Signal line window for the MACD signal line (default is 9 days).

        Returns:
        - DataFrame with the original data and additional columns for MACD, Signal Line, and MACD Histogram.
        """
        # Calculate short-term and long-term exponential moving averages
        self.data['short_ema'] = self.data[self.return_].ewm(span=short_period, adjust=False).mean()
        self.data['long_ema'] = self.data[self.return_].ewm(span=long_period, adjust=False).mean()

        # Calculate MACD line
        self.data['macd'] = self.data['short_ema'] - self.data['long_ema']

        # Calculate Signal Line
        self.data['macd_signal_line'] = self.data['macd'].ewm(span=signal_period, adjust=False).mean()

        # Calculate MACD Histogram
        self.data['macd_histogram'] = self.data['macd'] - self.data['macd_signal_line']

        # Drop intermediate columns
        self.data.drop(['short_ema', 'long_ema'], axis=1, inplace=True)

        return self.data
    
    
    def calculate_bollinger_bands(self, period=20, num_std_dev=2):
        """
        Calculate Bollinger Bands for a specified column in a DataFrame.

        Parameters:
        - period: Window size for the moving average (default is 20).
        - num_std_dev: Number of standard deviations for the upper and lower bands (default is 2).

        Returns:
        - DataFrame with the original data and additional columns for Upper Band, Middle Band, and Lower Band.
        """
        # Calculate the rolling mean and standard deviation
        self.data['rolling_mean'] = self.data[self.return_].rolling(window=period).mean()
        self.data['rolling_std'] = self.data[self.return_].rolling(window=period).std()

        # Calculate Upper and Lower Bollinger Bands
        self.data['upper_bollinger_band'] = self.data['rolling_mean'] + num_std_dev * self.data['rolling_std']
        self.data['lower_bollinger_band'] = self.data['rolling_mean'] - num_std_dev * self.data['rolling_std']

        # Drop intermediate columns
        self.data.drop(['rolling_mean', 'rolling_std'], axis=1, inplace=True)

        return self.data
    
    
    def calculate_vwap(self,low_price, high_price, closing_price, volume_column):
        """
        Calculate Volume Weighted Average Price (VWAP) for a specified column in a DataFrame.

        Parameters:
        - price_column: The name of the column representing the prices.
        - volume_column: The name of the column representing the trading volumes.

        Returns:
        - DataFrame with the original data and an additional column for VWAP.
        """
        # Calculate the cumulative sum of (Price * Volume) and cumulative sum of Volume
        self.data['typical_price'] = (self.data[low_price] + self.data[high_price] + self.data[closing_price])/3
        self.data['cumulative_price_volume'] = (self.data['typical_price'] * self.data[volume_column]).cumsum()
        self.data['cumulative_volume'] = self.data[volume_column].cumsum()

        # Calculate VWAP
        self.data['vwap'] = self.data['cumulative_price_volume'] / self.data['cumulative_volume']

        # Drop intermediate columns
        self.data.drop(['typical_price','cumulative_price_volume', 'cumulative_volume'], axis=1, inplace=True)

        return self.data    

    def create_lag_variables(self, time_column, columns, lag=1, num_of_days_to_lag=1):
        """
        Creates lag variables for specified columns in a DataFrame.

        Parameters:
        - df: DataFrame
        - columns: List of column names for which lag variables will be created
        - lag: Number of lags (default is 1 for lagging by one day)
        - num_of_days_to_lag: a variable can be lag for multiple times. For ex: If num_of_days_to_lag=2 then the function will create column_lag1, column_lag2 and so on.

        Returns:
        - df: DataFrame with lag variables added
        """
        self.data = self.data.sort_values(time_column).copy()
        lag_columns = []
        new_column_names = []
        
        for column in columns:
            for day in list(range(lag, lag + num_of_days_to_lag)):
                new_column_name = f"{column}_lag{day}"
                
                new_column_names.append(new_column_name)
                lag_columns.append(self.data[column].shift(day))

        lag_df = pd.concat(lag_columns, axis=1)
        lag_df.columns = new_column_names
        self.data = pd.concat([self.data, lag_df], axis=1)

        return self.data
    
    # def calculate_adx(self, period=14):
    #     """
    #     Calculates the Average Directional Index (ADX) using the pandas_ta library.

    #     Parameters:
    #         self.data (DataFrame): The DataFrame containing OHLCV data.
    #         period (int): The period for ADX calculation.

    #     Returns:
    #         DataFrame: A new DataFrame containing the ADX values.
    #     """
    #     # Calculate ADX
    #     adx_values = ta.adx(self.data['high'], self.data['low'], self.data['close'], length=period)
        
    #     # Assign ADX values to the DataFrame
    #     self.data['adx'] = adx_values.iloc[:,0]
    #     return self.data
    
    def calculate_pdi(self, period=14):
        """
        Calculates the Positive Directional Index (PDI) for a given DataFrame of OHLCV data.
        
        Args:
        data (DataFrame): DataFrame containing OHLCV (Open, High, Low, Close, Volume) data.
        period (int): Period for calculating the PDI.
        
        Returns:
        pandas.Series: A series containing the calculated PDI values.
        """
        high_diff = self.data['high'] - self.data['high'].shift(1)
        low_diff = self.data['low'].shift(1) - self.data['low']
        
        plus_dm = pd.Series(0, index=self.data.index)
        plus_dm[(high_diff > 0) & (high_diff > low_diff)] = high_diff
        
        tr = true_range(self.data)
        atr = tr.rolling(window=period).mean()
        
        self.data['pdi'] = 100 * (plus_dm.rolling(window=period).sum() / atr)

        return self.data

    
    def calculate_ndi(self, period=14):
        """
        Calculates the Negative Directional Index (NDI) for a given DataFrame of OHLCV data.
        
        Args:
        data (DataFrame): DataFrame containing OHLCV (Open, High, Low, Close, Volume) data.
        period (int): Period for calculating the NDI.
        
        Returns:
        pandas.Series: A series containing the calculated NDI values.
        """
        high_diff = self.data['high'] - self.data['high'].shift(1)
        low_diff = self.data['low'].shift(1) - self.data['low']
        
        minus_dm = pd.Series(0, index=self.data.index)
        minus_dm[(low_diff > 0) & (low_diff > high_diff)] = low_diff
        
        tr = true_range(self.data)
        atr = tr.rolling(window=period).mean()
        
        self.data['ndi'] = 100 * (minus_dm.rolling(window=period).sum() / atr)
        return self.data


    def calculate_atr(self, period=14):
        """
        Calculates the Average True Range (ATR) for a given DataFrame of OHLCV data.
        
        Args:
        self.data (DataFrame): DataFrame containing OHLCV (Open, High, Low, Close, Volume) data.
        period (int): Period for calculating the ATR.
        
        Returns:
        pandas.Series: A series containing the calculated ATR values.
        """
        high_low = self.data['high'] - self.data['low']
        high_close_prev = abs(self.data['high'] - self.data['close'].shift(1))
        low_close_prev = abs(self.data['low'] - self.data['close'].shift(1))
        
        true_range = pd.DataFrame({'HL': high_low, 'HC': high_close_prev, 'LC': low_close_prev}).max(axis=1)
        
        self.data['atr']  = true_range.rolling(window=period).mean()
        return self.data


    def calculate_cci(self, period=20):
        """
        Calculates the Commodity Channel Index (CCI) for a given DataFrame of OHLCV data.
        
        Args:
        self.data (DataFrame): DataFrame containing OHLCV (Open, High, Low, Close, Volume) data.
        period (int): Period for calculating the CCI.
        
        Returns:
        pandas.Series: A series containing the calculated CCI values.
        """
        typical_price = (self.data['high'] + self.data['low'] + self.data['close']) / 3
        mean_deviation = typical_price.rolling(window=period).apply(lambda x: pd.Series(x).mad_custom(), raw=True)
        self.data['cci'] = (typical_price - typical_price.rolling(window=period).mean()) / (0.015 * mean_deviation)

        return self.data


    def calculate_chaikin(self, short_period=3, long_period=10):
        """
        Calculates the Chaikin Oscillator for a given DataFrame of OHLCV data.
        
        Args:
        self.data (DataFrame): DataFrame containing OHLCV (Open, High, Low, Close, Volume) data.
        short_period (int): Short period for the exponential moving average.
        long_period (int): Long period for the exponential moving average.
        
        Returns:
        pandas.Series: A series containing the Chaikin Oscillator values.
        """
        adl = (2 * self.data['close'] - self.data['high'] - self.data['low']) / (self.data['high'] - self.data['low']) * self.data['volume']
        adl_ema_short = adl.ewm(span=short_period, min_periods=short_period).mean()
        adl_ema_long = adl.ewm(span=long_period, min_periods=long_period).mean()
        
        self.data['chaikin'] = adl_ema_short - adl_ema_long

        return self.data

    def calculate_obv(self):
        """
        Calculates the On-Balance Volume (OBV).

        Parameters:
            self.data (DataFrame): The DataFrame containing OHLCV data.

        Returns:
            DataFrame: A new DataFrame containing the On-Balance Volume values.
        """
        # Calculate OBV
        self.data['obv'] = (self.data['close'] > self.data['close'].shift(1)).astype(int)
        self.data['obv'] = self.data['obv'].where(self.data['close'] == self.data['close'].shift(1), -self.data['obv'])
        self.data['obv'] = self.data['obv'] * self.data['volume']
        self.data['obv'] = self.data['obv'].cumsum()

        return self.data

    def calculate_roc(self, period=14):
        """
        Calculates the Rate of Change (ROC).

        Parameters:
            self.data (DataFrame): The DataFrame containing OHLCV data.
            period (int): The period for ROC calculation.

        Returns:
            DataFrame: A new DataFrame containing the ROC values.
        """
        # Calculate ROC
        self.data['roc'] = ((self.data['close'] - self.data['close'].shift(period)) / self.data['close'].shift(period)) * 100

        return self.data

    def calculate_smi(self, period=14, signal_period=3):
        """
        Calculates the Stochastic Momentum Indicator (SMI).

        Parameters:
            self.data (DataFrame): The DataFrame containing OHLCV data.
            period (int): The period for SMI calculation.
            signal_period (int): The period for signal line calculation.

        Returns:
            DataFrame: A new DataFrame containing the SMI and signal line values.
        """
        # Calculate the recent high and low
        self.data['recent_high'] = self.data['high'].rolling(window=period).max()
        self.data['recent_low'] = self.data['low'].rolling(window=period).min()

        # Calculate the raw SMI
        self.data['smi_raw'] = ((self.data['close'] - self.data['recent_low']) / (self.data['recent_high'] - self.data['recent_low'])) * 100

        # Calculate the signal line
        self.data['smi_signal_line'] = self.data['smi_raw'].rolling(window=signal_period).mean()
        
        self.data.drop(['recent_high', 'recent_low'], axis=1, inplace=True)
        return self.data

    def calculate_mfi(self, period=14):
        """
        Calculates the Money Flow Index (MFI)
        Parameters:
            self.data (DataFrame): The DataFrame containing OHLCV data.
            period (int): The period for MFI calculation.

        Returns:
            DataFrame: A new DataFrame containing the MFI values.
        """
        typical_price = (self.data['high'] + self.data['low'] + self.data['close']) / 3
        money_flow = typical_price * self.data['volume']
        positive_flow = money_flow.mask(typical_price.diff() < 0, 0)
        negative_flow = money_flow.mask(typical_price.diff() > 0, 0)
        
        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()
        
        money_flow_ratio = positive_mf / negative_mf
        
        self.data['money_flow_ratio'] = money_flow_ratio  # Add the money_flow_ratio to the DataFrame
        
        # Calculate MFI
        self.data['mfi'] = 100 - (100 / (1 + self.data['money_flow_ratio'].astype(float)))
 

        return self.data

    def get_data(self):
        return self.data
        
def add_date_column(df_name):
    df = pd.read_csv(df_name)
    news_time = df['Published_time'].apply(lambda x: x.split(' ')[0])
    df['Published_date'] = news_time
    df.to_csv(df_name)
    
def take_news_parameter(news_df_list_name, take_vaders_param=False, take_roberta_param=True, lag=0):
    news_df_list = []
    if take_roberta_param:
        usecols = ["Published_date","roberta_neg","roberta_pos", "roberta_neu"]
    elif take_vaders_param:
        usecols = ["Published_date","vader_neg","vader_pos","vader_neu"]
    if len(news_df_list_name) == 1:
        news_df = pd.read_csv(news_df_list_name[0], usecols=usecols)
    elif len(news_df_list_name) > 1:
        for news_df_name in news_df_list_name:
            news_df_list.append(pd.read_csv(news_df_name,usecols=usecols))
        # Concatenate datasets and reset index
        news_df = pd.concat(news_df_list).reset_index(drop=True)
        
    # Create lag vars for news_param
    usecols.remove('Published_date')
    if lag > 0:
        for column in usecols:
            news_df[column] = news_df[column].shift(lag)
        # news_df.drop(usecols, axis=1, inplace=True)
    news_df.rename(columns={'Published_date': 'time'}, inplace=True)
    news_df['time'] = pd.to_datetime(news_df['time']).dt.date
    return news_df[lag:]

def create_seasonality_features(df, time_column='time'):
    """
    Create time series features based on time series index.
    """
    df = df.copy()
    df[time_column] = pd.to_datetime(df[time_column])

    df['dayofweek'] = df[time_column].dt.dayofweek
    df['month'] = df[time_column].dt.month
    df['dayofmonth'] = df[time_column].dt.day
    # df['weekofyear'] = df[time_column].dt.isocalendar().week
    # Convert 'weekofyear' to categorical type
    columns_to_convert = ['dayofweek', 'month', 'dayofmonth']
    df[columns_to_convert] = df[columns_to_convert].astype('category')

    return df

def get_working_days(start_date, end_date, holidays_list):
    all_dates = pd.date_range(start=start_date, end=end_date, freq='B')
    working_days = all_dates[~all_dates.isin(holidays_list) & (all_dates.weekday < 5)]
    return pd.Series(working_days.date.tolist())
def add_new_working_days(df, number_of_new_working_days):
    # Function to get working days between start_date and end_date excluding holidays

    # Define constants
    country_code = 'VN'
    year = 2024

    # Get country holidays
    vietnam_holidays = holidays.country_holidays(country_code, years=year)
    vietnam_holidays.update({'2024-02-08': "New Year's Day"})

    # Get all working days
    all_working_days = get_working_days('2024-01-01', '2024-12-31', vietnam_holidays.keys())

    # Extract latest date from DataFrame
    latest_date = pd.to_datetime(df['time']).dt.date.max()

    # Find the index of the latest date in the working days list
    latest_date_index = all_working_days[all_working_days == latest_date].index[0]
    next_working_days=[]
    for future_day in list(range(1,number_of_new_working_days+1)):
        # Calculate the next working day
        next_working_day_index = latest_date_index + future_day
        next_working_day = all_working_days[next_working_day_index]
        next_working_days.append(next_working_day)

    # Create a new row with the specified conditions
    new_row = pd.DataFrame({'time':next_working_days})
    new_row['isFuture'] = True

    # Reset 'isFuture' for existing rows
    df['isFuture'] = False

    # Concatenate the new row to the original DataFrame
    df = pd.concat([df, new_row], ignore_index=True)
    return df, next_working_days