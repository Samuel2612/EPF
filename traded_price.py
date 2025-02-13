
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import re
import zipfile
import os
from pathlib import Path


year = 2024


df_da_price = pd.read_csv(os.path.join(Path(r'C:\Users\samue\Documents\XS'),
                                     r"auction_spot_prices_netherlands_2024.csv"),
                       skiprows=1
                       )
df_da_price.rename({'Hour 3A': 'Hour 3'}, axis=1, inplace=True)

df_da_price['Delivery day'] = df_da_price['Delivery day'].map(lambda date_str: datetime.strptime(date_str, "%d/%m/%Y"))

df_da_price.set_index(['Delivery day'], inplace=True)

da_cols = ['Hour ' + str(k) for k in range(1, 25)]
df_da_price = df_da_price.loc[:, da_cols]


data_dir = Path(os.path.abspath(r'C:\Users\samue\Documents\XS\Continuous_Trades-NL-2024'))

list_zip = sorted(os.listdir(data_dir))

dates = []

for zip_filename in list_zip[-1:]:
    
    match = re.search(r'\d{8}-\d{8}', zip_filename)

    if match:
        date_range = match.group()
        d1, d2 = date_range.split('-')
        dates.append(d1)
        
    else:
        raise ValueError
        


    # # Path to the ZIP file
    zip_file_path = Path(os.path.join(data_dir, zip_filename))
    
    # # Name of the CSV file inside the ZIP file
    csv_file_name = zip_filename.removesuffix('.zip')
    
    list_trade_by_products = []
    
    # # Open the ZIP file
    with zipfile.ZipFile(zip_file_path, 'r') as zip_file:
        # Open the CSV file inside the ZIP file
        with zip_file.open(csv_file_name) as csv_file:
            # Load the CSV file into a Pandas DataFrame
            df = pd.read_csv(csv_file, skiprows=1)
            
            df = df[df['TradePhase']=='CONT']
            
            df['DeliveryStart'] = pd.to_datetime(df['DeliveryStart']) 
            df['DeliveryEnd']   = pd.to_datetime(df['DeliveryEnd'])
            df['Tenor']         = df['DeliveryEnd'] - df['DeliveryStart']
            df                  = df[df['Tenor']==pd.Timedelta(hours=1)]
            
            # id_1hr_contracts =  set(df['DeliveryStart'])
            print(f'Total buy volume is {sum(df.loc[df["Side"]=="BUY", "Volume"])}')
            print(f'Total sell volume is {sum(df.loc[df["Side"]=="SELL", "Volume"])}')
            
            df['ExecutionTime'] = pd.to_datetime(df['ExecutionTime'])
            
            delivery_start_times = sorted(list(set(df['DeliveryStart'] )))
            
            
            
            
            for j, t in enumerate(delivery_start_times):
                t_date = t.date()
                t_hour = t.hour
                
                t_da_price = df_da_price.loc[pd.to_datetime(t_date), 'Hour ' + str(t_hour + 1)]
                
                assert(abs(t_da_price) >=0)
                
                
                sub_df = df[df['DeliveryStart']==t]
            
                trade_times = sorted(list(set(sub_df['ExecutionTime'])))
                
                traded_prices = np.zeros((len(trade_times), ))
                
                for k, ts in enumerate(trade_times):
                    trades = sub_df[sub_df['ExecutionTime']==ts]
                    
                    if len(set(trades['Price'])) > 1:
                        volumes = trades['Volume'].values
                        prices  = trades['Price'].values
                        
                        volume_weighted_price = np.sum(volumes * prices)/ np.sum(volumes)
                        traded_prices[k]      = volume_weighted_price
                    else:
                        price = trades['Price'].values[0]
                        traded_prices[k] = price
                        
                        
                # Create a DataFrame
                df_trade_times = pd.DataFrame(trade_times, columns=['trade_time'])

                # Group by hour and count the number of timestamps in each hour
                hourly_counts = df_trade_times.groupby(pd.Grouper(key='trade_time', freq='h')).size().reset_index(name='count')
                
                if True:
                    print(hourly_counts)
                
                if True:
                    
                    alpha, beta = 1.3, 0.7
                            
                    plt.figure(figsize=(12, 6))
                    plt.plot(trade_times, traded_prices, label=f'DeliveryStart:{t_date.strftime("%Y-%m-%d")}: {t.hour}-hour')
                    plt.axhline(y=t_da_price, color='r', linestyle='--', linewidth=2, label='DA price')
                    plt.axhline(y=t_da_price * alpha, color='limegreen', linestyle='--', linewidth=2, label=f'DA price multiplied by {alpha}')
                    plt.axhline(y=t_da_price * beta,  color='orange', linestyle='--', linewidth=2, label=f'DA price multiplied by {beta}')
                    plt.xticks(trade_times, pd.Series(data=trade_times).dt.strftime("%h:%M"), fontsize=6, rotation=90)  # Rotate for better readability
                    plt.legend()
                    plt.tight_layout()
                    plt.show()
                    
                    
                list_trade_by_products.append( pd.Series(data=traded_prices, index=trade_times))
                
    if True:
        for j in range(len(list_trade_by_products) - 1):
            
            df_this, df_next = list_trade_by_products[j], list_trade_by_products[j+1]
            
            plt.figure(figsize=(12, 6))
            
            t = delivery_start_times[j]
            t_date = t.date()
            t_hour = t.hour
            
            plt.plot(df_this.index, df_this.values, label=f'This:DeliveryStart:{t_date.strftime("%Y-%m-%d")}: {t.hour}-hour',
                     color='r', marker='o')
            
            t = delivery_start_times[j+1]
            t_date = t.date()
            t_hour = t.hour
            
            plt.plot(df_next.index, df_next.values, label=f'Next:DeliveryStart:{t_date.strftime("%Y-%m-%d")}: {t.hour}-hour',color='b', marker='s')
            
            plt.xlabel('Trade time')
            plt.ylabel('Price')
            plt.title('Two neighbouring contracts')
            plt.legend()
            plt.grid()
            plt.xticks(rotation=45)  # Rotate x-ticks for better readability
            plt.tight_layout()
            plt.show()
    
                    
                    
                        
    # # Now `df` contains the data from the CSV file
    # print(df.head())