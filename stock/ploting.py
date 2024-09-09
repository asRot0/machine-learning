import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load the dataset into a Pandas DataFrame
df = pd.read_csv('../datasets/HistoricalPrices (1).csv')

# Rename the column to remove an additional space
df = df.rename(columns = {' Open': 'Open', ' High': 'High', ' Low': 'Low', ' Close': 'Close'})

# Convert the date column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Sort the dataset in the ascending order of date
df = df.sort_values(by = 'Date')

print(df.head())
# Extract the date and close price columns
dates = df['Date']
closing_price = df['Low']

# Create a line plot
plt.plot(dates, closing_price)

# Show the plot
plt.show()