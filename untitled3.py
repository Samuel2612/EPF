import pandas as pd
from entsoe import EntsoePandasClient

def _fetch_day_ahead_prices_nl_2021(api_key: str) -> pd.Series:
    """
    Fetch day-ahead prices for 2021 for the Netherlands from the ENTSO-E Transparency Platform.

    Parameters:
        api_key (str): Your ENTSO-E Transparency API key.
    
    Returns:
        pd.Series: A pandas Series of day-ahead prices indexed by hourly timestamps.
    """
    client = EntsoePandasClient(api_key=api_key)

    # Define the period for 2021 using correct timestamps.
    start = pd.Timestamp('2021-01-02 00:00:00', tz='Europe/Brussels')
    end = pd.Timestamp('2021-01-04 00:00:00', tz='Europe/Brussels')
    
    # Query day-ahead prices. The country code for the Netherlands is "NL".
    prices = client.query_day_ahead_prices('NL', start=start, end=end)
    return prices

# Example usage:
if __name__ == "__main__":
    my_api_key = "196a8ab2-b40d-4ac9-aac0-afa6a6c32f2a"
    prices_2021 = _fetch_day_ahead_prices_nl_2021(my_api_key)
    print(prices_2021.head())