import numpy as np
from statistics import mean
from scipy.stats import percentileofscore
from scipy.stats import norm
import pandas as pd
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import requests
from st_aggrid import AgGrid, GridUpdateMode, GridOptionsBuilder
from datetime import datetime, timedelta
from PIL import Image
import json
import os

#Gets user input

def app():
    # Page Configuration

    import streamlit as st
    if st.sidebar.button("Refresh"):
    # Use Streamlit's session state to trigger a rerun when the button is clicked
        st.session_state['refresh'] = not st.session_state.get('refresh', False)

    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
    import requests
    import streamlit as st

 
    if "instrument1" not in st.session_state:
        st.session_state.instrument1 = "USD_MXN"
    if "granularity" not in st.session_state:
        st.session_state.granularity = "D"
    if "count1" not in st.session_state:
        st.session_state.count1 = 1000
    if 'window_size' not in st.session_state:
        st.session_state.window_size = max(2, min(10, int(st.session_state.count1 / 2)))  


    frequency_to_periods = {
        'S5': 252 * 24 * 60 * 60 / 5,  # Number of 5-second intervals in a trading year
        'S10': 252 * 24 * 60 * 60 / 10,
        'S15': 252 * 24 * 60 * 60 / 15,
        'S30': 252 * 24 * 60 * 60 / 30,
        'M1': 252 * 24 * 60,  # Number of 1-minute intervals in a trading day times trading days in a year
        'M2': 252 * 24 * 60 / 2,
        'M4': 252 * 24 * 60 / 4,
        'M5': 252 * 24 * 60 / 5,
        'M10': 252 * 24 * 60 / 10,
        'M15': 252 * 24 * 60 / 15,
        'M30': 252 * 24 * 60 / 30,
        'H1': 252 * 24,  # Number of 1-hour intervals in a trading day times trading days in a year
        'H2': 252 * 12,
        'H3': 252 * 8,
        'H4': 252 * 6,
        'H6': 252 * 4,
        'H8': 252 * 3,
        'H12': 252 * 2,
        'D': 252,  # Trading days in a year
        'W': 52,   # Weeks in a year
        'M': 12    # Months in a year
    }

    periods_per_year = frequency_to_periods.get(st.session_state.granularity, 252)  # Default to daily if unknown

    # Calculate the annualization factor
    annualization_factor = np.sqrt(periods_per_year)

        # Dictionary to store data frames for each currency
    data_frames = {}

    def fetch_fx_data(currencies, granularity, count1):
        # OANDA credentials (use environment variables or Streamlit secrets for production)
        bearer_token = os.environ.get('OANDA_API_KEY')
        
        for instrument1 in currencies:
            endpoint = f"https://api-fxpractice.oanda.com/v3/instruments/{instrument1}/candles"
            headers = {'Authorization': f'Bearer {bearer_token}', 'Content-Type': 'application/json'}
            params = {'granularity': st.session_state.granularity, 'count': st.session_state.count1, 'price': 'M'}
            
            response = requests.get(endpoint, headers=headers, params=params)
            if response.status_code != 200:
                st.error(f"Error fetching data from OANDA for {instrument1}: {response.status_code}")
                st.text(f"Response: {response.text}")
                continue

            fx_data = response.json()
            prices = fx_data['candles']
            
            rows = []
            for candle in prices:
                if candle['complete']:
                    row = {
                        'Timestamp': pd.to_datetime(candle['time']),
                        'Open': float(candle['mid']['o']),
                        'High': float(candle['mid']['h']),
                        'Low': float(candle['mid']['l']),
                        'Close': float(candle['mid']['c'])
                    }
                    rows.append(row)

            df = pd.DataFrame(rows)
            df.set_index('Timestamp', inplace=True)
            data_frames[instrument1] = df

    def plot_volatility_charts():
        fig = go.Figure()
        
        for instrument1, df in data_frames.items():
            if not df.empty:
                df['Returns'] = 100 * df['Close'].pct_change().dropna()
                df['Annualized_Vol'] = df['Returns'].rolling(window=st.session_state.window_size).std() * annualization_factor
                df.dropna(inplace=True)
                
                fig.add_trace(go.Scatter(x=df.index, y=df['Annualized_Vol'], mode='lines', name=f'{instrument1} Volatility'))

        fig.update_layout(yaxis_title='Volatility', xaxis_title='Date', title='Annualized Volatility Comparison')
        st.plotly_chart(fig, use_container_width=True)

   # List of currencies
    currencies = st.multiselect("Which Currencies would you like to see?", [
    'EUR_USD', 'GBP_USD', 'USD_MXN', 'USD_CAD', 'USD_JPY', 'AUD_USD', 'USD_CNH', 'NZD_USD',
    'USD_SEK', 'USD_SGD', 'USD_THB', 'USD_TRY', 'USD_ZAR', 'EUR_CHF', 'EUR_DKK', 'EUR_PLN',
    'EUR_SEK', 'GBP_JPY', 'AUD_JPY', 'USD_HKD', 'USD_HUF', 'USD_ILS', 'USD_CZK', 'USD_QAR',
    'EUR_OMR', 'EUR_KWD', 'USD_AED'
    ], default = ['EUR_USD', 'GBP_USD'])

    
    st.session_state.granularity = 'D'
    st.session_state.count1 = 1000
    fetch_fx_data(currencies, st.session_state.granularity, st.session_state.count1)
    plot_volatility_charts()

    # Assuming `data_frames` is a dictionary containing the data frames of currency pairs with 'Close' prices

    # Calculate the annualization factor based on the selected granularity
    periods_per_year = frequency_to_periods.get(st.session_state.granularity, 252)  # Default to daily if unknown
    annualization_factor = np.sqrt(periods_per_year)

    # Calculate Returns for each currency pair and store in the same DataFrame
    for currency, df in data_frames.items():
        df['Returns'] = df['Close'].pct_change()

    # Prepare an empty DataFrame to store portfolio volatility over time
    portfolio_volatility_over_time = pd.DataFrame()

    # Calculate the rolling covariance matrix of returns for the selected window size
    window_size = st.session_state.window_size  # Assuming this is defined in your Streamlit UI
    combined_returns = pd.concat([df['Returns'] for currency, df in data_frames.items()], axis=1, keys=data_frames.keys())
    rolling_covariance = combined_returns.rolling(window=window_size).cov().dropna()

    # Equal weights for each currency pair
    n = len(data_frames)  # Number of currency pairs
    weights = np.ones(n) / n

    # Calculate portfolio volatility for each time point
    for date in rolling_covariance.index.get_level_values(0).unique():
        cov_matrix = rolling_covariance.loc[date].values.reshape(n, n)  # Reshape to ensure matrix is n x n
        portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights.T))) * annualization_factor *100

        # Store the portfolio volatility for the current time point
        portfolio_volatility_over_time.loc[date, 'Portfolio Volatility'] = portfolio_vol

    # Plot Portfolio Volatility Over Time
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=portfolio_volatility_over_time.index, y=portfolio_volatility_over_time['Portfolio Volatility'],
                            mode='lines', name='Portfolio Volatility'))
    fig.update_layout(title='Portfolio Volatility Over Time',
                    xaxis_title='Date', yaxis_title='Portfolio Volatility')
    st.plotly_chart(fig, use_container_width=True)
    # Now use 'window_size' as the default value for the slider
    window_size = st.slider(
        "How many periods would you like to consider for your volatility Window?",
        min_value=2,
        max_value=int(st.session_state.count1 / 2),
        value=st.session_state.window_size,  # Use the session state value as the default
        key="window_size"
    )
    granularity = st.selectbox(
        "Select Granularity",
        options=['S5', 'S10', 'S15', 'S30', 'M1', 'M2', 'M4', 'M5', 'M10', 'M15', 'M30', 'H1', 'H2', 'H3', 'H4', 'H6', 'H8', 'H12', 'D', 'W', 'M'],
        key='granularity',
        index=['S5', 'S10', 'S15', 'S30', 'M1', 'M2', 'M4', 'M5', 'M10', 'M15', 'M30', 'H1', 'H2', 'H3', 'H4', 'H6', 'H8', 'H12', 'D', 'W', 'M'].index(st.session_state.granularity)  # Set default by finding the index
    )

    count1 = st.number_input(
        "Number of Data Points for FX",
        min_value=1,
        value=st.session_state.count1,  # Use session state value as default
        key='count1'
    )


    volcurrency = st.sidebar.selectbox("Which currency would you like to view a vol surface for? A separate window will be generated.", [
        'EUR_USD', 'GBP_USD', 'USD_MXN', 'USD_CAD', 'USD_JPY', 'AUD_USD', 'USD_CNH', 'NZD_USD',
        'USD_SEK', 'USD_SGD', 'USD_THB', 'USD_TRY', 'USD_ZAR', 'EUR_CHF', 'EUR_DKK', 'EUR_PLN',
        'EUR_SEK', 'GBP_JPY', 'AUD_JPY', 'USD_HKD', 'USD_HUF', 'USD_ILS', 'USD_CZK', 'USD_QAR',
        'EUR_OMR', 'EUR_KWD', 'USD_AED'
        ], key="volcurrency")



    st.sidebar.button("Generate Volatility Surface", key='gensurface' )
    if st.session_state.gensurface:


        def fetch_fx_data_single_currency(currency = st.session_state.volcurrency, granularity='D', days=1000):
            """
            Fetch FX data for a single currency over the specified number of days.

            Args:
            currency (str): The currency pair to fetch data for (e.g., 'USD_MXN').
            granularity (str): The granularity of the data ('D' for daily).
            days (int): The number of days of data to fetch.

            Returns:
            pd.DataFrame: DataFrame with the fetched data, indexed by timestamp.
            """
            bearer_token = os.environ.get('OANDA_API_KEY')

            # Calculate the 'to' and 'from' parameters based on the current date and the number of days
            to_date = datetime.utcnow()
            from_date = to_date - timedelta(days=days)

            # Convert dates to the format required by the OANDA API
            to_date_str = to_date.strftime('%Y-%m-%dT%H:%M:%SZ')
            from_date_str = from_date.strftime('%Y-%m-%dT%H:%M:%SZ')

            # OANDA API endpoint for fetching candle data
            endpoint = f"https://api-fxpractice.oanda.com/v3/instruments/{currency}/candles"

            # Set up the headers and parameters for the API request
            headers = {'Authorization': f'Bearer {bearer_token}', 'Content-Type': 'application/json'}
            params = {
                'from': from_date_str,
                'to': to_date_str,
                'granularity': granularity,
                'price': 'M'  # Midpoint prices
            }

            # Make the API request
            response = requests.get(endpoint, headers=headers, params=params)

            # Check for successful response
            if response.status_code == 200:
                fx_data = response.json()
                prices = fx_data['candles']

                # Prepare the data for DataFrame
                rows = []
                for candle in prices:
                    if candle['complete']:
                        row = {
                            'Timestamp': pd.to_datetime(candle['time']),
                            'Open': float(candle['mid']['o']),
                            'High': float(candle['mid']['h']),
                            'Low': float(candle['mid']['l']),
                            'Close': float(candle['mid']['c'])
                        }
                        rows.append(row)

                # Create and return the DataFrame
                df = pd.DataFrame(rows)
                df.set_index('Timestamp', inplace=True)
                return df
            else:
                # Handle errors (e.g., logging, raising an exception)
                raise Exception(f"Error fetching data: {response.status_code}, {response.text}")

        # Example usage
       
        df = fetch_fx_data_single_currency(currency)

        # Display the DataFrame (optional, for verification)
     
        df['Returns'] = df['Close'].pct_change()
        df.dropna(inplace=True)

        # Prepare arrays for 3D plotting
        time = pd.to_datetime(df.index).values
        window_sizes = np.arange(5, 201, 10)  # From 5 to 200 in 10-day increments
        volatility_surface = np.zeros((len(window_sizes), len(df)))

        # Calculate annualized volatility for each window size
        for i, window in enumerate(window_sizes):
            rolling_std = df['Returns'].rolling(window=window).std()
            annualized_vol = rolling_std * np.sqrt(252)  # Assuming 252 trading days in a year
            volatility_surface[i, :] = annualized_vol.values

        # Create mesh grids for plotting
        T, W = np.meshgrid(time, window_sizes)

        # Plotting the 3D volatility surface
        fig = go.Figure(data=[go.Surface(z=volatility_surface, x=T, y=W)])
        fig.update_layout(
            title='3D Volatility Surface',
            scene=dict(
                xaxis_title='Time',
                yaxis_title='Look Back Window Size',
                zaxis_title='Annualized Volatility'
            )
        )

        fig.show()


    
  
    
