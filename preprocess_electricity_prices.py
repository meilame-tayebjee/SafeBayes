# To preprocess electricity prices and charges, dowloaded from:
# https://energy-charts.info/charts/price_spot_market/chart.htm?l=fr&c=FR&interval=year&year=2019&legendItems=00000110

import pandas as pd

def calculate_day_average(csv_file):
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Manually convert the 'Date (TC+1)' column to datetime format
    df['Date'] = pd.to_datetime(df['Date (TC+1)'], format='%Y-%m-%dT%H:%M%z',utc=True).dt.tz_convert("CET").dt.date

    # Group by date and calculate the mean for 'Charge' and 'Day Ahead Auction' columns
    day_avg_charge = df.groupby('Date')['Charge'].mean()
    day_avg_auction = df.groupby('Date')['Day Ahead Auction'].mean()
    
    # Time variable between 0 and 1
    time = (pd.to_datetime(day_avg_charge.index).dayofyear - 1) / 365

    # Create new DataFrame
    result_df = pd.DataFrame({
        'Date': day_avg_charge.index,
        'time' : time.values,
        'charge': day_avg_charge.values,
        'price': day_avg_auction.values
    })

    return result_df

csv_file = 'data/raw_price_elec.csv' 
csv_file_output = 'data/avg_price_elec.csv'
result_df = calculate_day_average(csv_file)
result_df.to_csv(csv_file_output)