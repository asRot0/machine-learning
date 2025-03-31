import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import datetime as dt
import pandas as pd

# Load the stock data
stock_data = pd.read_csv('../datasets/stocks/TSLA/tesla_small.csv')
# stock_data = pd.read_csv('../datasets/HistoricalPrices (1).csv')

stock_data = stock_data.rename(columns={' Open': 'Open', ' High': 'High', ' Low': 'Low', ' Close': 'Close'})

# Convert the 'Date' column to datetime
stock_data['Date'] = pd.to_datetime(stock_data['Date'])

# Remove '$' and convert columns to numeric
for column in ['Close/Last', 'Open', 'High', 'Low']:
    stock_data[column] = stock_data[column].replace({'\$': '', ',': ''}, regex=True).astype(float)

# Print the first few rows to check
print(stock_data.head())

# Plotting the data
plt.figure(figsize=(15, 10))

# Format the x-axis to show date
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=30))

# Use the 'Date' column for x-axis and plot the 'High' and 'Low' stock prices
plt.plot(stock_data['Date'], stock_data['High'], label='High')
plt.plot(stock_data['Date'], stock_data['Low'], label='Low')

# Auto format date on x-axis
plt.gcf().autofmt_xdate()

# Add title and labels
plt.title("Tesla Stock Price")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()

# Show the plot
plt.show()
