import streamlit as st
import landing
import volstudy
from PIL import Image
import cfar
import requests
import streamlit as st
import pandas as pd
import requests
import numpy as np
from datetime import datetime
import yfinance as yf
import time
import plotly.graph_objs as go
from streamlit_autorefresh import st_autorefresh
import json
import News
import os

st.set_page_config(page_title="Speckled Kingsnake", layout="wide")



###### MODULE FOR TICKER TAPE #########

def fetch_fx_data(instrument, granularity='D', count=2):
    # OANDA credentials (use environment variables or Streamlit secrets for production)
    bearer_token = st.secrets['OANDA_API_KEY'] #os.environ.get('OANDA_API_KEY')
    
    # OANDA API endpoint for candlestick data
    endpoint = f"https://api-fxpractice.oanda.com/v3/instruments/{instrument}/candles"
    
    # Set the request headers with the bearer token
    headers = {
        'Authorization': f'Bearer {bearer_token}',
        'Content-Type': 'application/json'
    }

    # Fetch the candlestick data
    params = {
        'granularity': granularity,
        'count': count,
        'price': 'M'
    }
    
    response = requests.get(endpoint, headers=headers, params=params)
    
    if response.status_code != 200:
        st.error(f"Error fetching data from OANDA: {response.status_code}")
        st.text(f"Response: {response.text}")  # This will print the error details
        return pd.DataFrame()  # Return an empty DataFrame on error

    # Process the response JSON and convert to DataFrame
    fx_data = response.json()
    prices = fx_data['candles']
    
    if not prices or len(prices) < 2:
        return pd.DataFrame()  # Return an empty DataFrame if not enough data
    
    # Use the last two complete candles to calculate the return
    prev_close = float(prices[-2]['mid']['c'])
    last_close = float(prices[-1]['mid']['c'])
    
    daily_return = round((last_close - prev_close) / prev_close * 100, 2)
    
    return {
        'Symbol': instrument,
        'Price': last_close,
        'Change': daily_return
    }

def fetch_all_returns():
    pairs = ['EUR_USD', 'GBP_USD', 'USD_MXN', 'USD_CAD', 'USD_JPY', 'USD_CNY', 'AUD_USD']
    results = [fetch_fx_data(pair) for pair in pairs]
    return pd.DataFrame(results)

# Fetch returns and update the ticker
df_returns = fetch_all_returns()



ticker_data = '  •  '.join([
    f"{row.Symbol}: ${row.Price} (<span style=\"color: {'green' if row.Change > 0 else 'red'}\">{'▲' if row.Change > 0 else '▼'}{row.Change}</span>)"
    for index, row in df_returns.iterrows()
])


ticker_html = f"""
<div class="ticker-wrap">
<div class="ticker">
  <div class="ticker__item">{ticker_data}</div>
  <div class="ticker__item">{ticker_data}</div> <!-- Duplicate content for a seamless loop -->
</div>
</div>
<style>
@keyframes ticker {{
  0% {{ transform: translateX(0); }}
  100% {{ transform: translateX(-50%); }} /* Move only half of the total width for looping */
}}
.ticker-wrap {{
  width: 100%;
  overflow: hidden;
  background-color: #333;
  padding: 10px 0;
  color: #FFF;
  font-size: 20px;
  box-sizing: border-box;
  
}}
.ticker {{
  display: flex;
  width: fit-content;
  animation: ticker 60s linear infinite;
}}
.ticker__item {{
  white-space: nowrap;
  padding-right: 50px; /* Space between items */
}}
</style>
"""

st.markdown(ticker_html, unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)  
####### END OF MODULE FOR TICKER TAPE ##############################

image_path = 'media/Globe - WHITE.png'
image = Image.open(image_path)
st.sidebar.image(image)
st.sidebar.markdown("<hr>", unsafe_allow_html=True)

if 'get_started_clicked' not in st.session_state:
    st.session_state['get_started_clicked'] = False


# Initialize the session state to track if the "Get Started" button has been clicked
if 'get_started_clicked' not in st.session_state:
    st.session_state['get_started_clicked'] = False

# Display introductory text and the landing page by default
if not st.session_state['get_started_clicked']:
    st.sidebar.write("Welcome to Speckled Kingsnake - a GPS proprietary risk tool designed for pricing and managing derivatives. Let's get started!")
    st.sidebar.write("- (2-13-2024) Version 1.0 supports TARFs and vanilla structures")
    st.sidebar.write("- (2-14-2024) Version 1.1 added a new cash flow hedging module and streaming news")
    st.sidebar.write("- (2-20-2024) Version 1.1 added a new cash flow hedging module and streaming news")
    landing.app()

    # Place the "Get Started" button after the text and the landing page content
    if st.sidebar.button("Get Started"):
        # Update the session state when the button is clicked
        st.session_state['get_started_clicked'] = True
        # Rerun the app to reflect the changes immediately
        st.experimental_rerun()

# After the "Get Started" button is pressed, display the radio selection
if st.session_state['get_started_clicked']:
    # Define your options for the radio selection
    options = ["CFAR Tool", "Volatiltiy Studies", "News"]
    choice = st.sidebar.radio("Where would you like to be directed?", options=options)
    
    
    if choice == "Volatiltiy Studies":
        volstudy.app()
    
    if choice == "CFAR Tool":
        cfar.app()
   
    if choice == "News":
        News.app()

st.sidebar.markdown("<hr>", unsafe_allow_html=True)

