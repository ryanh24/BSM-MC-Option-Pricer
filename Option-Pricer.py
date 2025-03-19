import yfinance as yf
import numpy as np
import scipy.stats as sp
import matplotlib.pyplot as plt
import mplfinance as mpf
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

st.set_page_config(layout="wide")

rename_dict = {
                "contractSymbol": "Option Symbol",
                #"lastTradeDate": "Last Trade Date",
                "strike": "Strike",
                "lastPrice": "Last Price",
                "bid": "Bid",
                "ask": "Ask",
                #"change": "Price Change",
                #"percentChange": "% Change",
                "volume": "Volume",
                "openInterest": "Open Interest",
                "impliedVolatility": "IV",
                "inTheMoney": "ITM",
                #"currency": "Currency",
                #"contractSize": "Contract Size"
            }


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
    actual_time_to_exp = (actual_exp_date - today).days / 365.0
    
    options_data = company.option_chain(best_date)
    calls = options_data.calls.drop(columns = ["change", "percentChange", "lastTradeDate", "currency", "contractSize"])
    calls = calls.rename(columns = rename_dict)
    puts = options_data.puts.drop(columns = ["change", "percentChange", "lastTradeDate", "currency", "contractSize"])
    puts = puts.rename(columns = rename_dict)
    return calls, puts, actual_time_to_exp

#Fetch stock data of the past year's prices
def get_stock_data(ticker_symbol):
    data = yf.download(ticker_symbol, period = '1y')
    plt.figure(figsize=(12, 6))
    plt.plot(data['Close'])
    plt.title(f'{ticker_symbol} Historical Stock Price')
    plt.ylabel('Stock Price (USD)')
    plt.xlabel('Date')
    plt.grid(True)
    returns = np.log(data['Close'][ticker_symbol] / data['Close'][ticker_symbol].shift(1)) #takes the logarithm of today's stock/ yesterday's for all stock data
    volatility = np.sqrt(252) * returns.std()
    current_price = data['Close'][ticker_symbol][-1]
    return volatility, current_price

#BSM
class Option_Pricer:
    def __init__(self, S, K, T, r, sigma):
        self.S = S        # Underlying price
        self.K = K        # Strike price
        self.T = T        # Time to expiration in years
        self.r = r        # Risk-free interest rate
        self.sigma = sigma  # Volatility 


    ## BSM
    def d1(self):
        return (np.log(self.S/self.K) + self.T * (self.r + (self.sigma ** 2)/2)) / (self.sigma * np.sqrt(self.T))
    def d2(self):
        return self.d1() - self.sigma * np.sqrt(self.T)
    def call_price_BSM(self):
        return (self.S * sp.norm.cdf(self.d1(), 0, 1)) - (self.K * np.exp(-self.r * self.T) * sp.norm.cdf(self.d2(), 0, 1))
    def put_price_BSM(self):
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
    

    ## Monte Carlo
    def monte_carlo(self, option_type = ''):
        Z = np.random.normal(size=500000)  # Generating the paths
        antithetic_Z = -Z  # Decided to try an antithetic since it's a simple way to produce better accuracy

        # Stock prices for Z the antithetic_Z
        S_T = self.S * np.exp((self.r - 0.5 * self.sigma**2) * self.T + self.sigma * np.sqrt(self.T) * Z)
        S_T_anti = self.S * np.exp((self.r - 0.5 * self.sigma**2) * self.T + self.sigma * np.sqrt(self.T) * antithetic_Z)

        # Payoffs
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
    K = row['Strike']
    model = Option_Pricer(S, K, T, r, sigma)

    if option_type == 'call':
        bsm_price = model.call_price_BSM()
        delta = model.call_delta()
        theta = model.call_theta()
        rho = model.call_rho()
        mc_price = model.monte_carlo(option_type='call')
    else:
        bsm_price = model.put_price_BSM()
        delta = model.put_delta()
        theta = model.put_theta()
        rho = model.put_rho()
        mc_price = model.monte_carlo(option_type='put')

    return {
        "BSM Price": bsm_price,
        "MC Price": mc_price,
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

def tab_remove():
    desired_ticker = st.session_state['Ticker']
    if desired_ticker in st.session_state["tickers"]:
        st.session_state["tickers"].remove(desired_ticker.upper())


with st.sidebar:
    st.title("Black-Scholes and Monte Carlo Option Pricer")
    st.write("Ryan Har")
    st.markdown("[LinkedIn](https://www.linkedin.com/in/ryan-har/)")

    st.text_input(
        "Enter Ticker:", key = "Ticker"
        )
    if st.button("Add Ticker"):
        tab_add()
    if st.button("Remove Ticker"):
        tab_remove()
    
    
    
# Creating the tabs in the main area for each ticker in our session list

st. title("Black Scholes and Monte Carlo Option Pricer")
if st.session_state["tickers"]:
    tabs = st.tabs(st.session_state["tickers"])
    for i, t in enumerate(st.session_state["tickers"]):
        with tabs[i]:
            a_volatility, current_price = get_stock_data(t)
            st.subheader(f'Current Price: {current_price}')
            st.subheader(f'Annualized Volatility: {a_volatility}')
            st.subheader('Filter by Strike Price/Date, Input (r, IV)')
            today = datetime.today().date()


            col1, col2 = st.columns(2)

            with col1:
                selected_date = st.date_input("Select Expiry Date (Rounded to nearest Expiry Date)", value=today + timedelta(days=30), key=f"expiry_date_{t}")
                calls, puts, actual_T = get_options_data(t, selected_date)
                max_IV = max(calls['IV'].max(),puts['IV'].max())
                # Calculate time to expiry in years
                st.write(f"Time to expiry for the nearest date: {actual_T:.5f} years")

                max_strike = max(calls['Strike'].max(),puts['Strike'].max())
                strike_range = st.slider(
                    'Strike Price Range',
                    min_value=0.0,
                    max_value = max_strike,
                    key = f"{t} Strike",
                    value = ((max_strike / 3), (max_strike * 2/3)),
                    step = 0.5    
                )        

            with col2:
                user_R = st.slider(
                "Risk-Free Rate",
                min_value=0.0,
                max_value=1.0,
                key = f"{t} Rate",
                value=0.05,
                step=0.001
                )

                iv_range = st.slider(
                "IV",
                min_value=0.01,
                max_value=max_IV,
                key = f"{t} Volatility",
                value=(a_volatility),
                step=0.01
                )    

            st.divider()
           #my filter for strikes
            calls_filtered = calls[
                (calls["Strike"] >= strike_range[0]) &
                (calls["Strike"] <= strike_range[1])
            ].copy()

            puts_filtered = puts[
                (puts["Strike"] >= strike_range[0]) &
                (puts["Strike"] <= strike_range[1])
            ].copy()
            # add the new columns
            def compute_call(row):
                return compute_option_metrics(
                    row,
                    S=current_price,
                    T=actual_T,
                    r=user_R,         
                    sigma=iv_range, 
                    option_type="call"
                )
            
            calls_metrics = calls_filtered.apply(compute_call, axis=1)
            # calls_metrics we convert from series to DF
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
                    sigma=iv_range,
                    option_type="put"
                )
            puts_metrics = puts_filtered.apply(compute_put, axis=1)
            puts_metrics_df = pd.DataFrame(list(puts_metrics))
            puts_filtered = pd.concat([puts_filtered.reset_index(drop=True),
                                    puts_metrics_df.reset_index(drop=True)], axis=1)

            # 5e. Display
            st.subheader(f"{t} Calls")
            st.dataframe(calls_filtered)

            st.subheader(f"{t} Puts")
            st.dataframe(puts_filtered)
else:
    st.info("No ticker selected yet. Please enter a ticker in the sidebar.")

