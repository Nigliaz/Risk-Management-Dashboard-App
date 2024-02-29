import numpy as np
from statistics import mean
from scipy.stats import percentileofscore
import streamlit as st
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
import requests
from prophet import Prophet



#Gets user input
def fetch_most_recent_close_price(i):
        bearer_token = st.secrets['OANDA_API_KEY'] #os.environ.get('OANDA_API_KEY')
        endpoint = f"https://api-fxpractice.oanda.com/v3/instruments/{i}/candles"
        headers = {'Authorization': f'Bearer {bearer_token}'}
        params = {'granularity': 'M1', 'count': 1, 'price': 'M'}  # 'M' for Midpoint for minimal data

        response = requests.get(endpoint, headers=headers, params=params)

        if response.status_code == 200:
            fx_data = response.json()
            latest_candle = fx_data['candles'][0]['mid']
            most_recent_close_price = latest_candle['c']
            return float(most_recent_close_price)
        else:
            # Handle errors (e.g., by logging or displaying an error message)
            return None  # Indicate failure to fetch price
        pass


def app():
    # Page Configuration
    if st.sidebar.button("Refresh"):
        # Use Streamlit's session state to trigger a rerun when the button is clicked
        st.session_state['refresh'] = not st.session_state.get('refresh', False)



    c1, c2 = st.sidebar.columns(2)

    # st.text_input("Enter Instrument", key="instrument1", value="EUR_USD")
    st.sidebar.selectbox("Which currency would you like to view a vol surface for? A separate window will be generated.", [
        'EUR_USD', 'GBP_USD', 'USD_MXN', 'USD_CAD', 'USD_JPY', 'AUD_USD', 'USD_CNH', 'NZD_USD',
        'USD_SEK', 'USD_SGD', 'USD_THB', 'USD_TRY', 'USD_ZAR', 'EUR_CHF', 'EUR_DKK', 'EUR_PLN',
        'EUR_SEK', 'GBP_JPY', 'AUD_JPY', 'USD_HKD', 'USD_HUF', 'USD_ILS', 'USD_CZK', 'USD_QAR',
        'EUR_OMR', 'EUR_KWD', 'USD_AED'
        ], key="instrument1")

        
   



    if "count1" not in st.session_state:
        st.session_state.count1 = 1000   


    L,M,R  = st.columns((.5, 4.5,.5), gap='large')

    with M:
        
        st.subheader("Time Series Forecasting Module")

        #########VOL MODULE################################################################################
        def fetch_fx_data(i, g, c):
            bearer_token = st.secrets['OANDA_API_KEY'] #os.environ.get('OANDA_API_KEY')
            endpoint = f"https://api-fxpractice.oanda.com/v3/instruments/{i}/candles"
            headers = {
                'Authorization': f'Bearer {bearer_token}',
                'Content-Type': 'application/json'
            }
            params = {
                'granularity': g,
                'count': c,
                'price': 'M'
            }
            response = requests.get(endpoint, headers=headers, params=params)
            if response.status_code != 200:
                st.error(f"Error fetching data from OANDA: {response.status_code}")
                st.text(f"Response: {response.text}")  # This will print the error details
                return pd.DataFrame()  # Return an empty DataFrame on error
            fx_data = response.json()
            prices = fx_data['candles']
            rows = []
            for candle in prices:
                if candle['complete']:  # Only take complete candles
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
            return df

        def plot_candlestick_chart_with_volatility(df, overlay_volatility=False):
            fig = go.Figure(data=[go.Candlestick(x=df.index,
                                                open=df['Open'],
                                                high=df['High'],
                                                low=df['Low'],
                                                close=df['Close'],
                                                name='Candlestick')])

            if overlay_volatility:
                fig.add_trace(go.Scatter(x=df.index, y=df['20_vol'], mode='lines', name='20-day Volatility'))

            fig.update_layout(xaxis_rangeslider_visible=False,  # Hides the range slider
                            yaxis_title='Price',
                            xaxis_title='Date',
                            title=f'{st.session_state.instrument1} Exchange Rate')

            st.plotly_chart(fig, use_container_width=True)



        if "window_size" not in st.session_state:
            st.session_state.window_size = 100
        if "granularity" not in st.session_state:
            st.session_state.granularity = "D"
      


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

        periods_per_year = frequency_to_periods.get(st.session_state.granularity, 252)  
        
        annualization_factor = np.sqrt(periods_per_year)

        # # Fetch and plot
        # df = fetch_fx_data(st.session_state.instrument1, st.session_state.granularity, st.session_state.count1)
        # if not df.empty:
        #     df['Returns'] = 100 * df['Close'].pct_change().dropna()
        #     df['Annualized_Vol'] = df['Returns'].rolling(window=st.session_state.window_size).std() * annualization_factor
        #     df.dropna(inplace=True)
    
        #     #st.dataframe(df[['Open', 'High', 'Low', 'Close', 'Returns']])
        #     plot_candlestick_chart_with_volatility(df)

     

       
        st.sidebar.selectbox("Select Granularity", options=['S5', 'S10', 'S15', 'S30', 'M1', 'M2', 'M4', 'M5', 'M10', 'M15', 'M30', 'H1', 'H2', 'H3', 'H4', 'H6', 'H8', 'H12', 'D', 'W', 'M'], key='granularity')
        st.sidebar.number_input("Number of Data Points for FX", min_value=1, value=500, key = 'count1')

        #########END OF VOL MODULE##################################################################################################################



        def prepare_data_for_prophet(df):
            df_prophet = df.reset_index().rename(columns={'Timestamp': 'ds', 'Close': 'y'})
            df_prophet['ds'] = df_prophet['ds'].dt.tz_localize(None)  # Remove timezone information if present
            return df_prophet
        
        
        # Fetch and process data
        df = fetch_fx_data(st.session_state.instrument1, st.session_state.granularity, st.session_state.count1)

        if not df.empty:
            # Your existing processing
            df['Returns'] = 100 * df['Close'].pct_change().dropna()
            df['Annualized_Vol'] = df['Returns'].rolling(window=st.session_state.window_size).std() * annualization_factor
            df.dropna(inplace=True)
            
            # Prepare data for Prophet
            df_prophet = prepare_data_for_prophet(df)
            
            # Initialize and fit the Prophet model
            model = Prophet()
            model.fit(df_prophet)
            
            # Create a DataFrame for future dates; adjust periods according to your forecasting needs
            future = model.make_future_dataframe(periods=12, freq='M')  # Adjust 'M' as needed for your forecasting frequency
            
            # Make a forecast
            forecast = model.predict(future)
            
            plt.style.use('dark_background')

            fig = model.plot(forecast)
            ax = fig.gca()
            ax.set_facecolor('#202020')  # Set the background to a dark color
            fig.set_facecolor('#202020')  # Set the figure background to a dark color
            plt.title(f'{st.session_state.instrument1} Close Price Forecast', fontsize=16, color='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.tick_params(axis='x', colors='white')
            ax.tick_params(axis='y', colors='white')

            
            plt.show()

            # Display the forecast plot in your tool
            st.pyplot(fig)

            fig_components = model.plot_components(forecast)
            st.pyplot(fig_components)





   
        

     
        

       
    







