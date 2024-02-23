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
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        # This will automatically update st.session_state.instrument1
        st.text_input("Enter Instrument", key="instrument1", value="EUR_USD")
    
    with col2:
        start_date1 = st.date_input("Start Date", value='today', key="start_date")

    

    action = st.sidebar.radio("Are you buying or selling the base ccy?", ('Sell', 'Buy'), key="bs")


   
    

    
    st.session_state.rate1 = fetch_most_recent_close_price(st.session_state.instrument1)

   
    st.session_state.strike1 = fetch_most_recent_close_price(st.session_state.instrument1)

    if 'sim_vol1' not in st.session_state:
        st.session_state.sim_vol1 = 0.11



   

    if 'num_steps1' not in st.session_state:
        st.session_state.num_steps1 = 24

    if 'num_iterations1' not in st.session_state:
        st.session_state.num_iterations1 = 100_000


    if 'epy1' not in st.session_state:
        st.session_state.epy1 = 12


    c1, c2 = st.sidebar.columns(2)
   
    st.sidebar.slider("Choose a Volatility for the simulation", min_value=0.0, max_value=1.0, key="sim_vol1")
    # Creating a list of values from 5 to 95 with a step of 5
    selected_value1 = st.sidebar.slider('Select a Value', min_value=5, max_value=95, step=5, value=5)  # Default value is set to 5

    with c1:
        st.number_input("Choose a Budget Rate", min_value=0.0000, max_value=1000.0000, key="strike1")
        st.number_input("Expiries per Year", min_value=0, max_value=365, key="epy1")
        
    with c2:
        st.number_input(f"Amount in {st.session_state.instrument1[4:]}", min_value=0.0, max_value=1000000000.0, value=10000.0, key="notional")
        st.number_input("Expiries", min_value=0, max_value=365, key="num_steps1")

    st.sidebar.number_input("Iterations", min_value=1000, max_value=1000000, key="num_iteration1")

    def update_interest_rate_diff():
        st.session_state.interest_rate_diff1 = st.session_state.quote - st.session_state.base
    
    with c1:

        base_interest_rate = st.number_input('Base Interest Rate', value=0.01, key="base",
                                                on_change=update_interest_rate_diff)

    with c2:
        quote_interest_rate = st.number_input('Quote Interest Rate', value=0.02, key="quote",
                                                on_change=update_interest_rate_diff)

    
    if 'interest_rate_diff1' not in st.session_state:
        st.session_state.interest_rate_diff1 = quote_interest_rate - base_interest_rate

    if "count1" not in st.session_state:
        st.session_state.count1 = 1000   


    L,M,R  = st.columns((2.2, 4.5,2.2), gap='large')

    with R:
        
        st.subheader("Volatility Analytics Module")

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


        def plot_volatility_chart(df):
            fig = go.Figure(data=[go.Scatter(x=df.index, y=df['Annualized_Vol'], mode='lines', name=f'{st.session_state.window_size}-candle Volatility')])

            fig.update_layout(yaxis_title='Volatility',
                            xaxis_title='Date',
                            title=f'Annualized "{st.session_state.granularity}" Volatility on a Rolling {st.session_state.window_size}-candle Window')

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

        # Fetch and plot
        df = fetch_fx_data(st.session_state.instrument1, st.session_state.granularity, st.session_state.count1)
        if not df.empty:
            df['Returns'] = 100 * df['Close'].pct_change().dropna()
            df['Annualized_Vol'] = df['Returns'].rolling(window=st.session_state.window_size).std() * annualization_factor
            df.dropna(inplace=True)
    
            #st.dataframe(df[['Open', 'High', 'Low', 'Close', 'Returns']])
            plot_candlestick_chart_with_volatility(df)

        plot_volatility_chart(df)

        st.slider("How many periods would you like to consider for your volatility Window?", min_value= 2, max_value= int(st.session_state.count1/2), key = "window_size")
        st.selectbox("Select Granularity", options=['S5', 'S10', 'S15', 'S30', 'M1', 'M2', 'M4', 'M5', 'M10', 'M15', 'M30', 'H1', 'H2', 'H3', 'H4', 'H6', 'H8', 'H12', 'D', 'W', 'M'], key='granularity')
        st.number_input("Number of Data Points for FX", min_value=1, value=500, key = 'count1')

        #########END OF VOL MODULE##################################################################################################################

        st.caption("These details represent the budget rate and expected notional. All fields are editable and they affect the calculation. If you selected Buy it is assumed that you are buying the base currency and selling the quote. A simulated rate path above your budget rate is a recorded as a loss. In that world, you were forced to buy higher than you had budgeted. Similarly, simulated paths below your budget rate are in your favor. The market beat your unhedged budget rate.")

        start_date1 = datetime.today()
      
        days_between_expiries = 365 / st.session_state.epy1

        expiry_dates1 = [(start_date1 + timedelta(days=i * days_between_expiries)).strftime('%Y-%m-%d') for i in range(st.session_state.num_steps1+1)]

        def calculate_forward_rates(num_steps, days_between_expiries, quote_interest_rate, base_interest_rate, strike, notional, expiry_dates):
            forward_rates = []
            for i in range(num_steps+1):
                tenor_years = i * days_between_expiries / 365
                forward_rate = (1 + quote_interest_rate * tenor_years) / (1 + base_interest_rate * tenor_years)
                forward_rates.append(forward_rate * strike)

            # Update or create the editable DataFrame in the session state
            if 'editable_df1' not in st.session_state or num_steps != len(st.session_state.editable_df1):
                st.session_state.editable_df1 = pd.DataFrame({
                    'Expiry Date': expiry_dates,
                    'Forward Rate': forward_rates,
                    'Notional': [notional] * len(forward_rates)
                })

            return st.session_state.editable_df1

        

        # Call the function to calculate forward rates and update the session state DataFrame
        editable_df1 = calculate_forward_rates(st.session_state.num_steps1, days_between_expiries, quote_interest_rate, base_interest_rate, st.session_state.strike1, st.session_state.notional, expiry_dates1)
        
        # Display the editable DataFrame using AgGrid
        grid_response1 = AgGrid(editable_df1, editable=True, fit_columns_on_grid_load=True)
        st.session_state.editable_df1 = grid_response1['data']

        


        @st.cache_data
        def run_simulation(r, sv, ns, ni, ird, ep):
            delta_t = 1 / ep
            drift = (ird - 0.5 * sv**2) * delta_t
            samples = np.random.normal(drift, sv * np.sqrt(delta_t), size=(ni, ns))
            cumulative_products = np.cumprod(1 + samples, axis=1)
            starting_rates = np.full((ni, 1), r)
            paths = np.hstack((starting_rates, starting_rates * cumulative_products))
            return paths

        simulation_results1 = run_simulation(
            st.session_state.rate1, 
            st.session_state.sim_vol1, 
            st.session_state.num_steps1, 
            st.session_state.num_iterations1, 
            st.session_state.interest_rate_diff1,
            st.session_state.epy1
            )
        
        

        # Convert to numeric, coercing errors to NaN
        st.session_state.editable_df1['Forward Rate'] = pd.to_numeric(st.session_state.editable_df1['Forward Rate'], errors='coerce')

        strike_prices1 = st.session_state.editable_df1['Forward Rate'].values

        

        if action == 'Buy':
            strike_difference1 = (strike_prices1 - simulation_results1)*st.session_state.notional
        elif action == 'Sell':
            strike_difference1 = (simulation_results1 - strike_prices1)*st.session_state.notional



       
        percentile = np.percentile(strike_difference1, selected_value1, axis=0)
        

       
    with M:
        st.title('Cash Flow at Risk Tool')
        # Plotting
        returns_fig = go.Figure()
        sample_size = 100 
        indices = np.random.choice(simulation_results1.shape[0], sample_size, replace=False)
        for i in indices:
            returns_fig.add_trace(go.Scatter(x=np.arange(st.session_state.num_steps1), y=simulation_results1[i, :], mode='lines', name=f'Path {i+1}'))
        # Update layout
        returns_fig.update_layout(xaxis_title="Time Steps", yaxis_title="Value")
        
        st.plotly_chart(returns_fig, use_container_width=True)





        
        
        percentile_fig = go.Figure()

        # Add bar trace with conditional coloring
        percentile_fig = go.Figure()
        percentile_fig.add_trace(go.Bar(
            x=np.arange(len(percentile)),
            y=percentile,  # Y-axis values
            marker=dict(color=['rgba(255, 87, 51, 0.6)' if val < 0 else 'rgba(144, 238, 144, 0.6)' for val in percentile]),
            
        ))

        # Update layout to match your style preferences
        percentile_fig.update_layout(
            xaxis_title="Simulation Number",
            template="plotly_white",
        )


        st.plotly_chart(percentile_fig, use_container_width=True)


        

        # Function to plot the histogram considering the whole cumulative return array
        def plot_histogram(strike_difference1, sample_size=None):
            # Reshape the 2D total_cumulative array into a 1D array for sampling
            all_values = strike_difference1.reshape(-1)  # This flattens the array
            
            if sample_size is None or sample_size >= all_values.size:
                # Use all data if sample_size is None or larger than the dataset
                sampled_values = all_values
            else:
                # Randomly select indices for the sample if a specific sample size is requested
                sample_indices = np.random.choice(all_values.size, sample_size, replace=False)
                sampled_values = all_values[sample_indices]
            
            # Create the histogram using Plotly
            fig = go.Figure(data=[go.Histogram(x=sampled_values, nbinsx=30, marker_color='blue')])
            
            # Update the layout
            fig.update_layout(
                xaxis_title='Cumulative Return Values',
                yaxis_title='Frequency',
                bargap=0.1,  # gap between bars of adjacent location coordinates
                template='plotly_white'
            )
            
            return fig


        # Checkbox to toggle full population
        use_full_population = st.sidebar.checkbox('Use Full Population', value=False)

        # Slider for sample size, shown only when not using the full population
        if not use_full_population:
            sample_size = st.sidebar.slider('Sample Size', min_value=100, max_value=strike_difference1.size, value=min(1000, strike_difference1.size), step=100)
        else:
            sample_size = None  # Indicates the full population should be used

        hist_fig = plot_histogram(strike_difference1, sample_size=sample_size)

        st.plotly_chart(hist_fig, use_container_width=True)





    with L:
        st.markdown("<br><br><br><hr>", unsafe_allow_html=True) 
        st.subheader("Vectorized Montecarlo")   
        #st.markdown("<hr>", unsafe_allow_html=True)
        st.caption(f"The plot below is a sampling of a vectorized montecarlo simulation. This approach is commonly used in exotic financial derivatives that exhibit path-dependent behaviors. Our use case is distinct, we are using the simulation to determine the probability of gains and losses agasint our budget rate under current and hypothetical market scenarios. Please use the volatility tool and interest rates to stress test various scenarios. Your current scenario settings have created an average terminal rate of {np.mean(simulation_results1):.2f}. This represents the mean future value of {st.session_state.instrument1} and should be carefully considered.")
        
        
        st.markdown("<hr><br><br><br><br><hr>", unsafe_allow_html=True)
        st.subheader(f"{selected_value1}th Percentile Gains/Losses at each Expiry")   
        #st.markdown("<hr>", unsafe_allow_html=True)
        st.caption(f"The bar chart below depicts the {selected_value1}th percentile gain/loss of each expiry along the {st.session_state.num_steps1} selected expiries. Feel free to use the percentile slider in the sidebar to examine the universe of possibilities. Percentile calculations are sensitive to you current volatiltiy of {st.session_state.sim_vol1} and the interest rates. ")
        
        st.markdown("<hr><br><br><br><br><br><br><hr>", unsafe_allow_html=True) 
        st.subheader(f"Histogram of Returns at {st.session_state.num_steps1}th Expiry")   
        #st.markdown("<hr>", unsafe_allow_html=True)  
        st.caption(f"The histogram is alternative way of considering the risk. It provides a more holistiv view of the shape of the distribution. This can be extremely helpful when considerin the real risks of going unhedged.")
        st.markdown("<hr>", unsafe_allow_html=True)  

        # st.write(st.session_state)




#Notes #########################
        # use historical USDMXN returns for 2000 for vol

        # add cost to hedge with FWD

        # add percent of notional hedged
