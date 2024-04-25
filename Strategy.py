import ta
import pandas_ta as taP
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np
from binance.client import Client
from dotenv import load_dotenv
import os

# Coinbase API

api_key = os.environ.get("api_key")
api_secret = os.environ.get("api_secret")

client = Client(api_key, api_secret)

symbol = "BTCUSDT"
interval = Client.KLINE_INTERVAL_1DAY
start_str = "2018-01-01"
end_str = "2099-01-01"

klines = client.get_historical_klines(symbol, interval, start_str, end_str)

df = pd.DataFrame(
    klines,
    columns=[
        "Timestamp",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Close time",
        "Quote asset volume",
        "Number of trades",
        "Taker buy base asset volume",
        "Taker buy quote asset volume",
        "Ignore",
    ],
)

df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="ms")
df.set_index("Timestamp", inplace=True)
df["Close"] = df["Close"].astype(float)
df["High"] = df["High"].astype(float)
df["Low"] = df["Low"].astype(float)
close = df['Close'].astype(float)
high = df["High"].astype(float)
low = df["Low"].astype(float)

# Start of Indicators

# Start of Universal 1
# Weighted Closing Price for Loop + Vii Stop


def system(subject, a, b):
    total = np.zeros_like(subject)
    for i in range(a, b + 1):
        shifted_subject = subject.shift(i)
        total += np.where(subject > shifted_subject, 1, -1)
    return total

df['subject'] = ((df['Close'] * 0.9) + (df['Close'].shift(1) * 0.1)) / 2
df['Score'] = system(df['subject'], 1, 47)


def vstop(close, high, low, atrlen, atrfactor):
    # Calculate ATR using pandas_ta with the taP alias
    atr = taP.atr(high=high, low=low, close=close, length=atrlen) * atrfactor
    atr.fillna(method='ffill', inplace=True)  # Forward fill to handle initial NaNs

    stop = np.full_like(close, fill_value=np.nan)
    uptrend = np.full_like(close, fill_value=True, dtype=bool)
    max_val = close.copy()
    min_val = close.copy()

    for i in range(1, len(close)):
        atrM = atr[i]
        max_val[i] = max(max_val[i-1], close[i])
        min_val[i] = min(min_val[i-1], close[i])
        if uptrend[i-1]:
            stop[i] = max(stop[i-1], max_val[i] - atrM)
        else:
            stop[i] = min(stop[i-1], min_val[i] + atrM)
        uptrend[i] = close[i] - stop[i] >= 0
        if uptrend[i] != uptrend[i-1]:
            max_val[i] = close[i]
            min_val[i] = close[i]
            stop[i] = max_val[i] - atrM if uptrend[i] else min_val[i] + atrM

    return stop, uptrend


atrlen = 49
mul = 6.4
close = df['Close']
high = df['High']
low = df['Low']

vStop, uptrend = vstop(close, high, low, atrlen, mul)

df['vstopl'] = uptrend
df['vstops'] = ~df['vstopl']

score = system(df['subject'], 1, 47)
df['Score'] = score  # Optionally store scores in DataFrame
WCP_long = df['Score'] > 40
WCP_short = df['Score'] < -10

L1 = WCP_long       # A Series where each value is True if the long condition is met
S1 = uptrend & df['vstops']        # A Series where each value is True if the stop condition is met

excL1 = L1 & ~S1
excS1 = S1 & ~L1

Coff1 = excL1.astype(float) - excS1.astype(float)  # Converts to 1, -1, 0


# End of Universal 1


# Start of Universal 2
# Median Supertrend + WCPFL

def vii_supertrend(df, atr_period=15, multiplier=5.5, percentile=50, percentile_length=6):
    # Calculate the Percentile Nearest Rank
    STsrc = df['Close'].rolling(window=percentile_length).apply(lambda x: np.percentile(x, percentile), raw=True)
    
    # Calculate the ATR
    atr = taP.atr(high=df['High'], low=df['Low'], close=df['Close'], length=atr_period)
    
    # Calculate the upper and lower bands
    u = STsrc + multiplier * atr
    l = STsrc - multiplier * atr
    
    # Initialize the variables
    df['d'] = np.nan
    df['st'] = np.nan
    
    # Loop through the DataFrame to simulate the `nz` functionality of Pine Script
    for i in range(1, len(df)):
        # Set the previous values of lower and upper
        pl = l.iloc[i - 1] if not np.isnan(l.iloc[i - 1]) else l.iloc[i]
        pu = u.iloc[i - 1] if not np.isnan(u.iloc[i - 1]) else u.iloc[i]
        
        # Determine the value for the current lower and upper
        l.iloc[i] = l.iloc[i] if l.iloc[i] > pl or df['Close'].iloc[i - 1] < pl else pl
        u.iloc[i] = u.iloc[i] if u.iloc[i] < pu or df['Close'].iloc[i - 1] > pu else pu
        
        # Determine the direction 'd' and 'st' values
        if i == 1 or np.isnan(atr.iloc[i - 1]):
            df['d'].iloc[i] = 1
        elif df['st'].iloc[i - 1] == pu:
            df['d'].iloc[i] = -1 if df['Close'].iloc[i] > u.iloc[i] else 1
        else:
            df['d'].iloc[i] = 1 if df['Close'].iloc[i] < l.iloc[i] else -1
        
        df['st'].iloc[i] = l.iloc[i] if df['d'].iloc[i] == -1 else u.iloc[i]
        
    return df['st'], df['d']


df['st'], df['d'] = vii_supertrend(df)

df['stl'] = df['d'] < 0
df['sts'] = df['d'] > 0

score = system(df['subject'], 1, 47)
df['Score'] = score  # Optionally store scores in DataFrame
WCP_long = df['Score'] > 40
WCP_short = df['Score'] < -10

L2 = df['stl'] & WCP_long
S2 = df['sts'] | WCP_short

excL2 = L2 & ~S2
excS2 = S2 & ~L2

Coff2 = excL2.astype(float) - excS2.astype(float)  # Converts to 1, -1, 0


# End of Universal 2
# RSI 

# Calculate DEMA
dema_close = taP.dema(df['Close'], length=140)

# Calculate RSI of DEMA
rsi_dema = taP.rsi(dema_close, length=1)

# Calculate standard deviation of DEMA
std_dema = taP.stdev(dema_close, length=140)

# Define long and short conditions
RSI_long = (rsi_dema > 70) & ~(df['Close'] < (dema_close + std_dema))
RSI_short = rsi_dema < 55

L3 = RSI_long
S3 = RSI_short

excL3 = L3 & ~S3
excS3 = S3 & ~L3

Coff3 = excL3.astype(float) - excS3.astype(float)  # Converts to 1, -1, 0


# End of Universal 3


# Start of Universal 4
# III EMA

# Calculate HMA
hma = taP.hma(df['Close'], length=93)
df['hmal'] = hma > hma.shift(1)
df['hmas'] = hma < hma.shift(1)

# Calculate DEMA for high and low
df['demal'] = taP.dema(df['High'], length=7)
df['demas'] = taP.dema(df['Low'], length=7)

# Calculate SMA
sma = taP.sma(df['Close'], length=60)

# Define main long and short conditions based on DEMA and SMA
df['mainl'] = df['demal'] > sma
df['mains'] = df['demas'] < sma

# Calculate standard deviation of SMA
sd = taP.stdev(sma, length=45) # Adjust length according to your condition
df['sd_upper'] = sma + sd
df['sd_lower'] = sma - sd

# Define additional conditions
df['invert_l'] = df['Close'] > df['sd_upper']

# Calculate ATR
df['atr'] = taP.atr(df['High'], df['Low'], df['Close'], length=14)
df['u'] = sma + df['atr']
df['atrs'] = df['Close'] < df['u']

# Combine conditions for final long and short signals
df['smal'] = df['mainl'] & df['invert_l']
df['smas'] = df['mains'] & df['atrs']

# Linear Regression (LSMA) condition
lsma = taP.linreg(df['Close'], length=77, offset=0)
df['lsmal'] = (lsma > lsma.shift(1)) & (df['Close'] > lsma)
df['lsmas'] = (lsma < lsma.shift(1)) & (df['Close'] < lsma)

# Determine final long and short position flags
L4 = df['hmal'] & df['smal'] & df['lsmal']
S4 = df['hmas'] & df['smas'] & df['lsmas']

excL4 = L4 & ~S4
excS4 = S4 & ~L4

Coff4 = excL4.astype(float) - excS4.astype(float)  # Converts to 1, -1, 0

# End of Universal 4


# Start of Universal 5
# Median For Loop

df['hlc3'] = (df['High'] + df['Low'] + df['Close']) / 3

def median_for_loop(df):
    # Initialize a score series with zeros
    score = pd.Series(np.zeros(len(df)), index=df.index)
    # Calculate the median
    median_series = df['hlc3'].rolling(window=2).median()
    
    # Loop through the DataFrame starting from the 60th element to the end
    for i in range(60, len(df)):
        total = 0
        # Compare the current median to the previous medians within the range 10 to 60
        for j in range(10, 61):
            if i - j >= 0:  # Check to avoid negative indexing
                # Increment or decrement total based on the comparison
                total += 1 if median_series.iloc[i] > median_series.iloc[i - j] else -1
        # Assign the total to the score series
        score.iloc[i] = total
    return score

# Apply the function to the dataframe to calculate scores
df['score'] = median_for_loop(df)

# Determine the long and short conditions
L5 = df['score'] > 40
S5 = df['score'] < 15

excL5 = L5
excS5 = S5

Coff5 = excL5.astype(float) - excS5.astype(float)  # Converts to 1, -1, 0

# End of Universal 5



# Long Short Conditions + Plotting

# Replace 0 with NaN so that they can be forward filled
Coff5.replace(to_replace=0.0, value=np.nan, inplace=True)

# Forward fill NaN values with the last valid observation
Coff5.fillna(method='ffill', inplace=True)

# Finally, re-convert NaN values to 0 if there are any left at the start of the series
Coff5.fillna(value=0.0, inplace=True)

# conditions
avg = (Coff1 + Coff2 + Coff3 + Coff4 + Coff5) / 5

long = avg > 0
short = avg  < 0

# Plotting 

cInterval = "1D"


plt.figure(figsize=(14, 7))
plt.plot(df.index, df['Close'], color="black")  # Ensure 'Close' is from df
plt.yscale("log")

# Adding custom title and legend
plt.title(f"{symbol} ({cInterval}) from {start_str}")
plt.legend(title="Legend")

# Add x and y labels
plt.xlabel("Date")
plt.ylabel("Price")

plt.fill_between(df.index, df['Close'], where=long, color="#00FFA6", alpha=0.3)
plt.fill_between(df.index, df['Close'], where=short, color="#FF0070", alpha=0.3)


# plt.show()
""" Strategy Plot """
st.pyplot(plt)  # streamlit plot