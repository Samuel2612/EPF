import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm
import pytz


start_date = datetime(2025, 2, 17)
end_date   = datetime(2025, 2, 17)

date_list = [start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)]

for as_of_date in tqdm(date_list):

    #as_of_date = datetime(2025, 2, 18)
    
    trade_date = as_of_date
    trade_date_format_1 = trade_date.strftime('%Y-%m-%d')
    trade_date_format_2 = trade_date.strftime('%d/%m/%Y')
    

    
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:58.0) Gecko/20100101 Firefox/58.0'}
    
    
    base_url = 'https://dataportal-api.nordpoolgroup.com/api/IntradayMarketStatistics'
    
    market_areas = ['EE', 'LT', 'LV',
                   'AT', 'BE', 'FR', 'GER', 'NL', 'PL',
                   'DK1', 'DK2',
                   'FI',
                   'NO1', 'NO2', 'NO3', 'NO4', 'NO5',
                   'SE1', 'SE2', 'SE3', 'SE4',
                   'TEL']
    
    for mkt_area in ['NL']:
    
        query_params = {
                        'date': trade_date_format_1,
                        'deliveryArea': mkt_area
                        }      
    
    
        # Send a GET request to the URL with query parameters
        response = requests.get(base_url, headers = headers, params=query_params)
    
    json_data = None
    
    if response.status_code == 200:
        json_data = response.json()
    else:
        print(f"Request failed with status code: {response.status_code}")
        raise requests.ConnectionError 
        
        
        

    
    
utc = pytz.UTC
cet = pytz.timezone("CET")

contract_records = []

for c in json_data["contracts"]:
    market = "Local" if c["isLocalContract"] else "SIDC"
    
    start_utc = pd.to_datetime(c["deliveryStart"], utc=True)
    end_utc   = pd.to_datetime(c["deliveryEnd"],   utc=True)
    start_cet = start_utc.tz_convert(cet)
    end_cet   = end_utc.tz_convert(cet)
    delivery_period_cet = f"{start_cet.strftime('%H:%M')} – {end_cet.strftime('%H:%M')}"

    resolution_minutes = int((end_cet - start_cet).total_seconds() / 60)
    
    contract_open_cet = pd.to_datetime(c["contractOpenTime"], utc=True).tz_convert(cet)
    contract_close_cet = pd.to_datetime(c["contractCloseTime"], utc=True).tz_convert(cet)
    
    row = {
        "Delivery period (CET)": delivery_period_cet,
        "Market": market,
        "Resolution in minutes": resolution_minutes,
        "High": c["highPrice"],
        "Low": c["lowPrice"],
        "VWAP": c["averagePrice"],
        "VWAP3H": c["averagePriceLast3H"],
        "VWAP1H": c["averagePriceLast1H"],
        "Open": c["openPrice"],
        "Close": c["closePrice"],
        "Buy Volume (MW)": c["buyVolume"],
        "Sell Volume (MW)": c["sellVolume"],
        "Transaction Volume (MW)": c["volume"],
        "Contract open time (CET)": contract_open_cet.strftime("%Y-%m-%d %H:%M:%S"),
        "Contract close time (CET)": contract_close_cet.strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    contract_records.append(row)


df = pd.DataFrame(contract_records)


#print(df.head())

    
    
    
