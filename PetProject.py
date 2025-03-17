import yfinance as yf
import numpy as np
import scipy.stats as sp
import matplotlib.pyplot as plt
import mplfinance as mpf
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

def get_options_data(ticker_symbol, target_date):
    company = yf.Ticker(ticker_symbol)
    options_dates = company.options  # list of expiration dates
    today = datetime.today().date()
    # Find closest expiration date to today
    best_date = min(
        options_dates,
        key=lambda d: abs(datetime.strptime(d, "%Y-%m-%d").date() - target_date)
    )
    
    # Recalculate back to the acetual time
    actual_exp_date = datetime.strptime(best_date, "%Y-%m-%d").date()
    T_actual = (actual_exp_date - today).days / 365.0
    
    options_data = company.option_chain(best_date)
    return options_data.calls, options_data.puts, T_actual

#Fetch stock data of the past year's prices
def get_stock_data(ticker_symbol):
    data = yf.download(ticker_symbol, period = '1y')
    plt.figure(figsize=(12, 6))
    plt.plot(data['Close'])
    plt.title(f'{ticker_symbol} Historical Stock Price')
    plt.ylabel('Stock Price (USD)')
    plt.xlabel('Date')
    plt.grid(True)
    returns = np.log(data['Close'] / data['Close'].shift(1)) #takes the logarithm of today's stock/ yesterday's for all stock data
    volatility = np.sqrt(252) * returns.std()
    current_price = data['Close'].iloc[0][0]#standardizes it from daily to annual
    return volatility, current_price

#BSM
class BlackScholesModel:
    def __init__(self, S, K, T, r, sigma):
        self.S = S        # Underlying price
        self.K = K        # Strike price
        self.T = T        # Time to expiration in years
        self.r = r        # Risk-free interest rate
        self.sigma = sigma  # Volatility 

    def d1(self):
        return (np.log(self.S/self.K) + self.T * (self.r + (self.sigma ** 2)/2)) / (self.sigma * np.sqrt(self.T))
    def d2(self):
        return self.d1() - self.sigma * np.sqrt(self.T)
    def call_price(self):
        return (self.S * sp.norm.cdf(self.d1(), 0, 1)) - (self.K * np.exp(-self.r * self.T) * sp.norm.cdf(self.d2(), 0, 1))
    def put_price(self):
        return (self.K * np.exp(-self.r * self.T) * sp.norm.cdf(-self.d2(), 0, 1)) - (self.S * sp.norm.cdf(-self.d1(), 0 , 1))

    ###GREEKS
    def call_delta(self):
        return sp.norm.cdf(self.d1(), 0, 1)
    def put_delta(self):
        return -1 * sp.norm.cdf(self.d1() * -1, 0, 1)    
    def gamma(self):
        return sp.norm.pdf(self.d1(), 0, 1)/ (self.S * self.sigma * np.sqrt(self.T))
    def vega(self):
        return self.S * sp.norm.pdf(self.d1(), 0, 1) *np.sqrt(self.T)
    def call_theta(self):
        return (-self.S * sp.norm.pdf(self.d1(), 0, 1) * self.sigma / (2*np.sqrt(self.T)) - (self.r * self.K * np.exp(-self.r * self.T) * sp.norm.cdf(self.d2(), 0, 1)))
    def put_theta(self):
        return (-self.S * sp.norm.pdf(self.d1(), 0, 1) * self.sigma / (2*np.sqrt(self.T)) + (self.r * self.K * np.exp(-self.r * self.T) * sp.norm.cdf(-self.d2(), 0, 1)))
    def call_rho(self):
        return self.K * self.T * np.exp(-self.r * self.T) * sp.norm.cdf(self.d2(),0,1)
    def put_rho(self):
        return -self.K * self.T * np.exp(-self.r * self.T) * sp.norm.cdf(-self.d2(),0,1)
    
    def monte_carlo_antithetic(self, option_type = ''):
        #np.random.seed(42)  # for reproducibility
        Z = np.random.normal(size=500000)  # Generate half the paths
        antithetic_Z = -Z  # Create antithetic counterpart

        # Stock prices for Z and -Z
        S_T = self.S * np.exp((self.r - 0.5 * self.sigma**2) * self.T + self.sigma * np.sqrt(self.T) * Z)
        S_T_anti = self.S * np.exp((self.r - 0.5 * self.sigma**2) * self.T + self.sigma * np.sqrt(self.T) * antithetic_Z)

        # Payoffs for Z and -Z
        if option_type == 'call':
            payoff_Z = np.maximum(S_T - self.K, 0)
            payoff_anti_Z = np.maximum(S_T_anti - self.K, 0)
        else:
            payoff_Z = np.maximum(self.K - S_T, 0)
            payoff_anti_Z = np.maximum(self.K - S_T_anti, 0)

        # Average the payoffs
        payoff = (payoff_Z + payoff_anti_Z) / 2
        discounted_payoff = np.exp(-self.r * self.T) * payoff

        return np.mean(discounted_payoff)
    
    
def compute_option_metrics(row, S, T, r, sigma, option_type):
#given our inputs for a single row, we will return a dict that can be appended to that row
    K = row['strike']
    model = BlackScholesModel(S, K, T, r, sigma)

    if option_type == 'call':
        bsm_price = model.call_price()
        delta = model.call_delta()
        theta = model.call_theta()
        rho = model.call_rho()
        mc_price = model.monte_carlo_antithetic(option_type='call')
    else:
        bsm_price = model.put_price()
        delta = model.put_delta()
        theta = model.put_theta()
        rho = model.put_rho()
        mc_price = model.monte_carlo_antithetic(option_type='put')

    return {
        "BSM_Price": bsm_price,
        "MC_Price": mc_price,
        "Delta": delta,
        "Gamma": model.gamma(),
        "Vega": model.vega(),
        "Theta": theta,
        "Rho":   rho
    }


#Portion of my app that's in streamlit    
if "tickers" not in st.session_state: #this will create the tabs based off the user's input of Tickers
    st.session_state["tickers"] = []

def tab_add():
    desired_ticker = st.session_state['Ticker']
    if desired_ticker not in st.session_state["tickers"]:
        st.session_state["tickers"].append(desired_ticker.upper())

with st.sidebar:
    st.title("Black-Scholes and Monte Carlo Option Pricer")
    st.write("Ryan Har")
    st.markdown("[LinkedIn](https://www.linkedin.com/in/ryan-har/)")

    st.text_input(
        "Enter Ticker:",
        key = 'Ticker', on_change = tab_add)
    
# 3. Create tabs in the main area for each ticker in our session list

st. title("Black Scholes and Monte Carlo Option Pricer")
if st.session_state["tickers"]:
    tabs = st.tabs(st.session_state["tickers"])
    for i, t in enumerate(st.session_state["tickers"]):
        with tabs[i]:
            a_volatility, current_price = get_stock_data(t)
            st.subheader(f'Current Price: {current_price}')
            st.subheader(f'Annualized Volatility: {a_volatility}')
            st.subheader('Filters and Inputs (T and r)')
            today = datetime.today().date()

            # Let the user select an expiry date. Default set to 180 days from today.
            selected_date = st.date_input("Select Expiry Date", value=today + timedelta(days=30))

            # Calculate time to expiry in years
            time_to_expiry = (selected_date - today).days / 365.0
            st.write(f"Time to expiry: {time_to_expiry:.5f} years")

            user_R = st.slider(
            "Risk-Free Rate",
            min_value=0.0,
            max_value=1.0,
            key = f"{t} Rate",
            value=0.05,
            step=0.001
            )

            calls, puts, actual_T = get_options_data(t, selected_date)
            max_strike = max(calls['strike'].max(),puts['strike'].max())
            max_IV = max(calls['impliedVolatility'].max(),puts['impliedVolatility'].max())
            st.write(actual_T)    
            st.write(a_volatility)        
            strike_range = st.slider(
                'Strike Price Range',
                min_value=0.0,
                max_value = max_strike,
                key = f"{t} Strike",
                value = ((max_strike / 3), (max_strike * 2/3)),
                step = 0.5    
            )
            iv_range = st.slider(
            "IV",
            min_value=0.01,
            max_value=max_IV,
            key = f"{t} Volatility",
            value=(max_IV/3, max_IV * 2/3),
            step=0.01
            )

            # 5c. Filter calls & puts to the user’s chosen strike range
            calls_filtered = calls[
                (calls["strike"] >= strike_range[0]) &
                (calls["strike"] <= strike_range[1])
            ].copy()

            puts_filtered = puts[
                (puts["strike"] >= strike_range[0]) &
                (puts["strike"] <= strike_range[1])
            ].copy()

            # 5d. For each row, compute BSM & MC
            # We'll add new columns to calls_filtered & puts_filtered.
            # For calls:
            def compute_call(row):
                return compute_option_metrics(
                    row,
                    S=current_price,
                    T=actual_T,
                    r=user_R,         # risk-free rate hardcoded for demo
                    sigma=a_volatility, # from slider
                    option_type="call"
                )

            calls_metrics = calls_filtered.apply(compute_call, axis=1)
            # calls_metrics is a Series of dicts, we can convert to DataFrame
            calls_metrics_df = pd.DataFrame(list(calls_metrics))
            calls_filtered = pd.concat([calls_filtered.reset_index(drop=True),
                                        calls_metrics_df.reset_index(drop=True)], axis=1)
            
            # For puts:
            def compute_put(row):
                return compute_option_metrics(
                    row,
                    S=current_price,
                    T=actual_T,
                    r=user_R,
                    sigma=a_volatility,
                    option_type="put"
                )
            puts_metrics = puts_filtered.apply(compute_put, axis=1)
            puts_metrics_df = pd.DataFrame(list(puts_metrics))
            puts_filtered = pd.concat([puts_filtered.reset_index(drop=True),
                                    puts_metrics_df.reset_index(drop=True)], axis=1)

            # 5e. Display
            st.subheader(f"{t} Calls (filtered)")
            st.dataframe(calls_filtered)

            st.subheader(f"{t} Puts (filtered)")
            st.dataframe(puts_filtered)
else:
    st.info("No ticker selected yet. Please enter a ticker in the sidebar.")

