import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm


as_of_date = datetime(2025, 2, 17)

delivery_date = as_of_date
delivery_date_format_1 = delivery_date.strftime('%Y-%m-%d')
delivery_date_format_2 = delivery_date.strftime('%d/%m/%Y')

trading_date = (delivery_date - timedelta(days=1)).strftime('%Y-%m-%d')


headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:58.0) Gecko/20100101 Firefox/58.0'}


base_url = 'https://dataportal-api.nordpoolgroup.com/api/DayAheadVolumes/multiple'

market_areas = ['EE', 'LT', 'LV',
               'AT', 'BE', 'FR', 'GER', 'NL', 'PL',
               'DK1', 'DK2',
               'FI',
               'NO1', 'NO2', 'NO3', 'NO4', 'NO5',
               'SE1', 'SE2', 'SE3', 'SE4',
               'TEL']

query_params = {
                'date': delivery_date_format_1,
                'market': 'DayAhead',
                'deliveryAreas': ','.join(market_areas)
                }      


# Send a GET request to the URL with query parameters
response = requests.get(base_url, headers = headers, params=query_params)

json_data = None

if response.status_code == 200:
    json_data = response.json()
else:
    print(f"Request failed with status code: {response.status_code}")
    raise requests.ConnectionError 
    
    
    
volumes_per_hour = json_data['multiAreaEntries']
volumes_per_hour = sorted(volumes_per_hour, key=lambda x: pd.to_datetime(x['deliveryStart']))


rows = []

for hour_info in volumes_per_hour:
    start_str = hour_info['deliveryStart']
    end_str   = hour_info['deliveryEnd']
    
    #Convert to date time
    start_time = pd.to_datetime(start_str) + timedelta(hours=1)
    end_time   = pd.to_datetime(end_str) + timedelta(hours=1)
    delivery_period = f"{start_time.strftime('%H:%M')} - {end_time.strftime('%H:%M')}"
    
    # Prepare a row dict
    row = {"Delivery period (CET)": delivery_period}


    for area, vals in hour_info["entryPerArea"].items():
        buy_col = f"{area} Buy (MW)"
        sell_col = f"{area} Sell (MW)"
        row[buy_col] = vals["buy"]
        row[sell_col] = vals["sell"]
        

    rows.append(row)

# Convert the list of row dicts to a DataFrame
df = pd.DataFrame(rows)


#print(df.head())

    
    
    
