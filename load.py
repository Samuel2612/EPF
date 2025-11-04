from entsoe import EntsoePandasClient          
import pandas as pd


client = EntsoePandasClient(     
    api_key="58d8fa63-472d-4254-ace5-52aff14928d4"   
)

country_code = "NL"               
tz = "Europe/Amsterdam"           

start = pd.Timestamp("2024-01-01 00:00", tz=tz)
end   = pd.Timestamp("2025-01-01 00:00", tz=tz)   


# load = client.query_generation(
#     country_code=country_code,
#     start=start,
#     end=end
# )


# gen_actual = client.query_generation(
#     country_code=country_code,
#     start=start,
#     end=end
# )                             


gen_forecast = client.query_generation_forecast(
    country_code=country_code,
    start=start,
    end=end
)                             



gen_forecast.to_csv("netherlands_generation_2024_forecast.csv")

print(gen_forecast.head())        
