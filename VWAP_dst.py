import datetime as _dt
from zoneinfo import ZoneInfo

import pandas as pd
from zipfile import ZipFile
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy import stats
import seaborn as sns
import types


_AMS_TZ = "Europe/Amsterdam"
_AMS_ZONE = ZoneInfo(_AMS_TZ)


def dst_days(year, tz_name=_AMS_TZ):
    tz = ZoneInfo(tz_name)
    dst_changes = []
    d = _dt.date(year, 1, 1)
    while d.year == year:
        midnight = _dt.datetime(d.year, d.month, d.day, tzinfo=tz)
        next_midnight = midnight + _dt.timedelta(days=1)
        if midnight.utcoffset() != next_midnight.utcoffset():
            dst_changes.append(d)
        d += _dt.timedelta(days=1)
    return dst_changes


def _is_spring_transition(day: _dt.date, tz=_AMS_ZONE) -> bool:
    """True if offset increases at the transition (CET->CEST)."""
    m0 = _dt.datetime.combine(day, _dt.time(0, 0), tzinfo=tz)
    m1 = m0 + _dt.timedelta(days=1)
    return (m1.utcoffset() or _dt.timedelta(0)) > (m0.utcoffset() or _dt.timedelta(0))


def _is_autumn_transition(day: _dt.date, tz=_AMS_ZONE) -> bool:
    """True if offset decreases at the transition (CEST->CET)."""
    m0 = _dt.datetime.combine(day, _dt.time(0, 0), tzinfo=tz)
    m1 = m0 + _dt.timedelta(days=1)
    return (m1.utcoffset() or _dt.timedelta(0)) < (m0.utcoffset() or _dt.timedelta(0))


def _effective_local_ts(day: _dt.date, hour: int) -> pd.Timestamp:
    """
    Map wall-clock hour (0..23) under the 'no-clock-change' rule to an actual tz-aware timestamp:
      - Spring forward day & hour==2: use 03:00 (duplicate 3 as surrogate for 2).
      - Autumn back day & hour==2: choose the FIRST 02:00 (CEST) -> ambiguous=True.
      - Otherwise: localize with ambiguous=False.
    """
    assert 0 <= hour <= 23
    if _is_spring_transition(day) and hour == 2:
        ts = pd.Timestamp(_dt.datetime.combine(day, _dt.time(hour=3)))
        return ts.tz_localize(_AMS_TZ, ambiguous=False, nonexistent="shift_forward")

    # Normal construction at requested hour
    ts = pd.Timestamp(_dt.datetime.combine(day, _dt.time(hour=hour)))


    if _is_autumn_transition(day) and hour == 2:
        return ts.tz_localize(_AMS_TZ, ambiguous=True, nonexistent="shift_forward")

    return ts.tz_localize(_AMS_TZ, ambiguous=False, nonexistent="shift_forward")


def _wall_index(delivery_date: _dt.date, delivery_hour: int,
                start_hours: float, time_interval: str) -> pd.DatetimeIndex:
    """
    Build a naive wall-clock index ending at (delivery_date, delivery_hour) with no DST jumps.
    """
    end_wall = pd.Timestamp(_dt.datetime.combine(delivery_date, _dt.time(hour=delivery_hour)))
    start_wall = end_wall - pd.Timedelta(hours=start_hours)
    step = pd.Timedelta(time_interval)
    periods = int(round(start_hours * 3600 / step.total_seconds()))
    return pd.date_range(start=start_wall, periods=periods, freq=time_interval) 

class LimitOrderBookProcessor:
    def __init__(self, 
                 zip_file, 
                 time_interval="15min",
                 start_hours=5,
                 auction_spot_csv="auction_spot_prices_netherlands_2021.csv",
                 aggregator_zip="auction_aggregated_curves_netherlands_2021.zip"):
        """
        Parameters:
            zip_file (str): Path to the zip file containing CSV files of trades.
            time_interval (str): Resampling interval (e.g., '5min', '15min', '30min').
            start_hours (int/float): Look-back window size before delivery (in hours).
            auction_spot_csv (str): Path to the CSV file containing auction spot prices.
        """
        self.zip_file = zip_file
        self.time_interval = time_interval
        self.start_hours = start_hours

        if "2021" in zip_file:
            self.target_year = 2021
        elif "2022" in zip_file:
            self.target_year = 2022
        else:
            # Default: infer from file / fallback to current year
            self.target_year = pd.Timestamp.today(tz=_AMS_TZ).year

        self.data = self._load_data()
        self._preprocess_data()

        # Day-ahead prices with DST rule baked in (per-row)
        self.auction_spot_prices = self._load_auction_spot_csv(auction_spot_csv)

        # Build per-(date,hour) bin data using the configured window (DST-safe)
        self._all_data_dict = self.extract_all_data(product_type='hourly', start_hours=self.start_hours)

        self.aggregator_zip = aggregator_zip
        self.aggregated_curves = self._load_aggregator_curves()

        # Precompute merit-order slopes aligned to the configured bins (DST-safe)
        self.compute_and_store_all_mo_slopes()

    # -----------------------------
    # Loading & preprocessing trades
    # -----------------------------
    def _load_data(self):
        df_list = []
        rename_dict_2022 = {
            "DeliveryEnd"  : "Delivery End",
            "DeliveryStart": "Delivery Start",
            "ExecutionTime": "Execution time",
            "Volume"       : "Quantity (MW)",
            "TradeId"      : "TradeID",
            "SelfTrade"    : "Is Self Trade",
        }

        with ZipFile(self.zip_file, 'r') as top_zip:
            nested_zips = [f for f in top_zip.namelist() if f.lower().endswith('.zip')]
            if nested_zips:
                for nested_zip_name in nested_zips:
                    with ZipFile(top_zip.open(nested_zip_name)) as nested_zip:
                        csv_files = [x for x in nested_zip.namelist() if x.lower().endswith('.csv')]
                        for csv_file in csv_files:
                            tmp_df = pd.read_csv(nested_zip.open(csv_file), skiprows=1)
                            if any(c in tmp_df.columns for c in rename_dict_2022):
                                tmp_df.rename(columns=rename_dict_2022, inplace=True)
                            df_list.append(tmp_df)
            else:
                csv_files = [f for f in top_zip.namelist() if f.lower().endswith('.csv')]
                for csv_file in csv_files:
                    tmp_df = pd.read_csv(top_zip.open(csv_file))
                    if any(c in tmp_df.columns for c in rename_dict_2022):
                        tmp_df.rename(columns=rename_dict_2022, inplace=True)
                    df_list.append(tmp_df)

        if not df_list:
            raise ValueError("No CSV files found in the zip (or nested zips).")

        combined = pd.concat(df_list, ignore_index=True)
        return combined

    def _preprocess_data(self):
        # All times to Europe/Amsterdam (CET/CEST), keeping actual offsets.
        # NOTE: we do NOT drop hour==2 anywhere; DST handled later by rule.
        self.data['Delivery Start'] = pd.to_datetime(
            self.data['Delivery Start'], format='ISO8601', utc=True
        ).dt.tz_convert(_AMS_TZ)
        self.data['Delivery End'] = pd.to_datetime(
            self.data['Delivery End'], format='ISO8601', utc=True
        ).dt.tz_convert(_AMS_TZ)
        self.data['Execution time'] = pd.to_datetime(
            self.data['Execution time'], format='ISO8601', utc=True
        ).dt.tz_convert(_AMS_TZ)

        if 'TradePhase' in self.data.columns:
            self.data = self.data[self.data['TradePhase'] == 'CONT']
        if 'Is Self Trade' in self.data.columns:
            self.data = self.data[self.data['Is Self Trade'] != 'Y']
        self.data = self.data.drop_duplicates(subset='TradeID', keep='first')

        self.data['ProductDuration'] = (
            self.data['Delivery End'] - self.data['Delivery Start']
        ).dt.total_seconds() / 60

        self.data = self.data[self.data['Delivery Start'].dt.year == self.target_year]

    # ----------------------------------
    # Day-ahead (auction) prices (DST OK)
    # ----------------------------------
    def _load_auction_spot_csv(self, csv_file: str) -> pd.DataFrame:
        """
        Returns a per-day table with columns '0'..'23' (local wall-clock hours).
        Implements the no-clock-change rule:
          - Autumn: if both 3A and 3B exist, keep 3A (first occurrence), drop 3B.
          - Spring: if hour '2' missing, fill it by duplicating hour '3'.
        Robustly handles 'Delivery day' / 'Delivery Date' / 'Date' as the date column.
        """
        df = pd.read_csv(csv_file, skiprows=1)
        df = df.iloc[:, :26].copy()
    

        date_col = df.columns[0]

    
        # Parse and localize dates to Europe/Amsterdam (CET/CEST)
        dt = pd.to_datetime(df[date_col], format="%d/%m/%Y", errors="coerce", dayfirst=True)
        if dt.isna().all():
            # fallback if format differs slightly
            dt = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
        if dt.isna().any():
            # drop impossible rows (rare)
            df = df.loc[~dt.isna()].copy()
            dt = dt.loc[~dt.isna()]
        df["Delivery Date"] = dt.dt.tz_localize(_AMS_TZ)
    
        # --- Normalize hour columns per row into 0..23 under the rule ---
        def normalize_hours(row: pd.Series) -> pd.Series:
            day = row["Delivery Date"].date()
            # Build raw mapping from available hour-like columns
            vals = {}
    
            for c in row.index:
                if c == "Delivery Date" or c == date_col:
                    continue
                name = str(c).strip()
                lname = name.lower().replace("hour", "").strip()  # strip "Hour"
                lname = lname.replace(" ", "")
                # Handle 3A/3B explicitly
                if lname == "3a":
                    vals[3] = row[c]  # prefer 3A (first)
                    continue
                if lname == "3b":
                    # discard (second occurrence)
                    continue
                # Numeric hour?
                try:
                    h = int(lname)
                    if 0 <= h <= 23:
                        vals[h] = row[c]
                except Exception:
                    pass
    
            # Spring forward: if hour 2 missing, duplicate hour 3
            if _is_spring_transition(day) and 2 not in vals:
                if 3 in vals and not pd.isna(vals[3]):
                    vals[2] = vals[3]
    
            # Construct complete 0..23 vector and interpolate within the day
            out = pd.Series(np.nan, index=[str(i) for i in range(24)], dtype="float64")
            for h, v in vals.items():
                out[str(h)] = v
    
            out = out.astype(float).interpolate(method="linear", limit_direction="both", axis=0)
            return out
    
        hours_df = df.apply(normalize_hours, axis=1)
        out = pd.concat([df[["Delivery Date"]], hours_df], axis=1)
        out.set_index("Delivery Date", inplace=True)
        return out  # index tz-aware; columns "0".."23"


    def _get_auction_spot_price(self, delivery_date: _dt.date, delivery_hour: int) -> float:
        day_ts = pd.Timestamp(_dt.datetime.combine(delivery_date, _dt.time(0, 0))).tz_localize(_AMS_TZ)
        h = delivery_hour
        # Spring rule: if hour 2 missing, use 3
        if _is_spring_transition(delivery_date) and h == 2:
            h = 3
        try:
            return float(self.auction_spot_prices.loc[day_ts].loc[str(h)])
        except KeyError:
            return np.nan


    def _compute_vwap(self, group: pd.DataFrame):
        total_qty = group['Quantity (MW)'].sum()
        if total_qty > 0:
            return (group['Price'] * group['Quantity (MW)']).sum() / total_qty
        return None

    def extract_all_data(self, product_type='hourly', start_hours=None):
        """
        Build a fixed grid of bins in the last `start_hours` before delivery using `self.time_interval`,
        and compute per-bin VWAP, ΔVWAP, and alpha=1{trades>0}. Fallback to DA price where needed.

        Keys are (delivery_date, delivery_hour) to avoid DST collisions.
        """
        if start_hours is None:
            start_hours = self.start_hours

        duration_map = {'hourly': 60, 'half_hourly': 30, 'quarterly': 15}
        if product_type not in duration_map:
            raise ValueError("Invalid product_type. Choose from 'hourly', 'half_hourly', or 'quarterly'.")
        target_duration = duration_map[product_type]

        df_filtered = self.data[np.isclose(self.data['ProductDuration'], target_duration, atol=1e-6)].copy()

        # Generate every day of the target year
        start_date = _dt.date(self.target_year, 1, 1)
        end_date = _dt.date(self.target_year, 12, 31)

        interval_td = pd.Timedelta(self.time_interval)
        interval_minutes = interval_td.total_seconds() / 60.0
        num_intervals = int(round(start_hours * 60 / interval_minutes))

        results = {}
        d = start_date
        while d <= end_date:
            for hour in range(24):
                # effective tz-aware delivery start (honors your DST rule)
                eff_ts = _effective_local_ts(d, hour)
                window_start_real = eff_ts - pd.Timedelta(hours=start_hours)
                full_index_real = pd.date_range(start=window_start_real, periods=num_intervals,
                                                freq=self.time_interval, tz=_AMS_TZ)
        
                # NEW: wall-clock (naive) bins that pretend no DST jump
                full_index_wall = _wall_index(d, hour, start_hours, self.time_interval)
        
                df_sub = df_filtered[df_filtered['Delivery Start'] == eff_ts].copy()
        
                if df_sub.empty:
                    fallback_price = self._get_auction_spot_price(d, hour)
                    if np.isnan(fallback_price):
                        fallback_price = -9999
                    df_result = pd.DataFrame({
                        'vwap': np.full(len(full_index_wall), fallback_price, dtype=float),
                        'vwap_changes': np.zeros(len(full_index_wall), dtype=float),
                        'alpha': np.zeros(len(full_index_wall), dtype=int),
                        'bin_timestamp_real': full_index_real  # keep the actual tz-aware timeline
                    }, index=full_index_wall)
                else:
                    df_sub.set_index('Execution time', inplace=True)
                    df_sub.sort_index(inplace=True)
        
                    vwap_series  = df_sub.resample(self.time_interval).apply(self._compute_vwap)
                    alpha_series = df_sub.resample(self.time_interval).size().apply(lambda x: 1 if x > 0 else 0)
        
                    # Align to the REAL tz-aware grid
                    vwap_series  = vwap_series.reindex(full_index_real)
                    alpha_series = alpha_series.reindex(full_index_real, fill_value=0)
        
                    if vwap_series.isna().all():
                        fallback_price = self._get_auction_spot_price(d, hour)
                        vwap_series = pd.Series(fallback_price, index=full_index_real)
                        alpha_series = pd.Series(0, index=full_index_real)
        
                    if pd.isna(vwap_series.iloc[0]):
                        vwap_series.iloc[0] = self._get_auction_spot_price(d, hour)
        
                    vwap_series  = vwap_series.ffill()
                    vwap_changes = vwap_series.diff().fillna(0)
        
                    # IMPORTANT: write values onto the WALL timeline (same length/order)
                    df_result = pd.DataFrame({
                        'vwap': vwap_series.values,
                        'vwap_changes': vwap_changes.values,
                        'alpha': alpha_series.values,
                        'bin_timestamp_real': full_index_real
                    }, index=full_index_wall)
        
                results[(d, hour)] = (eff_ts, df_result)
        
            d += _dt.timedelta(days=1)

        return results

    def aggregate_extracted_data(self, product_type='hourly', start_hours=None):
        data_dict = self.extract_all_data(product_type, start_hours)
        dfs = []
        for (d, h), (eff_ts, df) in data_dict.items():
            tmp = df.copy()
            tmp['delivery_date'] = pd.Timestamp(_dt.datetime.combine(d, _dt.time(0,0))).tz_localize(_AMS_TZ)
            tmp['delivery_hour'] = h
            tmp['delivery_start_effective'] = eff_ts
            # also keep the wall-clock delivery start (naive)
            tmp['delivery_start_wall'] = pd.Timestamp(_dt.datetime.combine(d, _dt.time(h)))
            # index is WALL clock -> expose as bin_timestamp (your forecaster uses this)
            tmp = tmp.reset_index().rename(columns={'index': 'bin_timestamp'})
            dfs.append(tmp)
        return pd.concat(dfs, ignore_index=True)

    # -----------------------------------------------
    # Aggregated (merit-order) curves (DST rule aware)
    # -----------------------------------------------
    def _load_aggregator_curves(self):
        """
        Reads aggregator files and returns a single DataFrame with columns:
            Date (tz-naive), Hour (0..23), Sale/Purchase, Price, Volume, ...
        DST handling:
          - Drop any '3B' rows (second occurrence) if present in source.
          - Keep all hours 0..23; we do NOT drop '2'. Spring day's hour 2 might be absent.
        """
        df_list = []
        with ZipFile(self.aggregator_zip, 'r') as z:
            csv_files = [f for f in z.namelist() if f.lower().endswith('.csv')]
            for csv_file in csv_files:
                agg_df = pd.read_csv(z.open(csv_file), skiprows=1)

                # Drop second occurrence labels like '3B' if present
                if 'Hour' in agg_df.columns:
                    mask_3B = agg_df['Hour'].astype(str).str.lower().str.contains('3b')
                    agg_df = agg_df[~mask_3B].copy()
                    # Normalize Hour to int 0..23: many sources are 1..24 -> shift down by 1
                    # Keep within [0,23]
                    try:
                        # If there are '3A' strings, coerce them to 3 first
                        agg_df['Hour'] = agg_df['Hour'].astype(str).str.replace('A','', regex=False)
                        agg_df['Hour'] = agg_df['Hour'].astype(int)
                    except Exception:
                        pass
                    agg_df["Hour"] = agg_df["Hour"] - 1
                    agg_df = agg_df[(agg_df["Hour"] >= 0) & (agg_df["Hour"] <= 23)].copy()

                # Normalize date
                if 'Date' in agg_df.columns:
                    agg_df['Date'] = pd.to_datetime(agg_df['Date'], dayfirst=True).dt.date

                df_list.append(agg_df)

        if not df_list:
            raise ValueError(f"No aggregator CSV found in {self.aggregator_zip}")

        combined = pd.concat(df_list, ignore_index=True)
        combined.sort_values(['Date','Hour','Sale/Purchase','Price'], inplace=True)
        return combined

    def transform_supply_curve(self, df_hour, p_min=-500.0):
        supply = df_hour[df_hour["Sale/Purchase"].str.lower() == "sell"].sort_values(by="Price")
        demand = df_hour[df_hour["Sale/Purchase"].str.lower().isin(["buy","purchase"])].sort_values(by="Price")

        def supply_inv(p): return np.interp(p, supply["Price"], supply["Volume"], left=0, right=0)
        def demand_inv(p): return np.interp(p, demand["Price"], demand["Volume"], left=0, right=0)

        inelastic_dmd = demand_inv(p_min)
        def transformed_supply_inv(p):
            return supply_inv(p) + inelastic_dmd - demand_inv(p)

        price_range = np.linspace(supply["Price"].min(), supply["Price"].max(), 500)
        transf_vols = np.array([transformed_supply_inv(px) for px in price_range])
        return price_range, transf_vols

    def compute_merit_order_slope(self, price_range, transformed_volumes, p_id0, q_values=[500,1000,2000]):
        def supply_func(vol):
            return np.interp(vol, transformed_volumes, price_range,
                             left=price_range[0], right=price_range[-1])
        def supply_func_inv(price_val):
            return np.interp(price_val, price_range, transformed_volumes,
                             left=transformed_volumes[0], right=transformed_volumes[-1])

        dem_implied = supply_func_inv(p_id0)

        mo_dict = {}
        for q in q_values:
            vol_plus  = dem_implied + q
            vol_minus = max(dem_implied - q, 0)
            p_plus  = supply_func(vol_plus)
            p_minus = supply_func(vol_minus)
            mo_dict[q] = (p_plus - p_minus)/(2.0*q)
        return mo_dict

    def compute_and_store_all_mo_slopes(self, p_min=-500.0):
        """
        For each (delivery_date, delivery_hour), compute merit-order slopes at each bin
        using the carry-forward VWAP for that bin (aligned to `self.time_interval`),
        honoring the DST 'no-clock-change' rule.
        """
        # Build (date,hour) -> (price_range, transf_vols)
        supply_cache = {}

        # First pass: what hours exist in aggregator file per day
        for (d, h), df_hour in self.aggregated_curves.groupby(['Date', 'Hour']):
            eff_ts = _effective_local_ts(d, h)  # not strictly needed, but ensures we're in scope
            supply_cache[(d, h)] = self.transform_supply_curve(df_hour, p_min=p_min)

        # Spring days may have no hour 2 -> duplicate hour 3
        for d in set(self.aggregated_curves['Date']):
            if _is_spring_transition(d) and (d, 2) not in supply_cache and (d, 3) in supply_cache:
                supply_cache[(d, 2)] = supply_cache[(d, 3)]

        self.merit_order_slopes = {}
        for (d, h), (eff_ts, df_bins) in self._all_data_dict.items():
            price_range, transf_vols = supply_cache.get((d, h), (None, None))

            # If hour 2 missing on spring day -> borrow hour 3, else fallback None
            if price_range is None:
                if _is_spring_transition(d) and h == 2 and (d, 3) in supply_cache:
                    price_range, transf_vols = supply_cache[(d, 3)]
                else:
                    # no supply; skip
                    records = []
                    for ts in df_bins.index:
                        records.append({'bin_timestamp': ts,
                                        'mo_slope_500': np.nan,
                                        'mo_slope_1000': np.nan,
                                        'mo_slope_2000': np.nan})
                    self.merit_order_slopes[(d, h)] = pd.DataFrame(records).set_index('bin_timestamp')
                    continue

            vwap_ff = df_bins['vwap'].ffill()
            if vwap_ff.isna().all():
                fallback_price = self._get_auction_spot_price(d, h)
                vwap_ff = pd.Series(fallback_price, index=df_bins.index)

            records = []
            prev_price = None
            prev_mo = None
            for ts, price in vwap_ff.items():
                if (prev_price is not None) and np.isclose(price, prev_price):
                    mo = prev_mo
                else:
                    mo = self.compute_merit_order_slope(price_range, transf_vols, price)
                    prev_price = price
                    prev_mo = mo
                records.append(
                    {
                        'bin_timestamp': ts,
                        'mo_slope_500':  mo[500],
                        'mo_slope_1000': mo[1000],
                        'mo_slope_2000': mo[2000],
                    }
                )
            self.merit_order_slopes[(d, h)] = pd.DataFrame(records).set_index('bin_timestamp')

    def merge_merit_order_slopes(self, df):
        flat_frames = []
        for (d, h), df_slope in self.merit_order_slopes.items():
            tmp = df_slope.reset_index()
            tmp['delivery_date'] = pd.Timestamp(_dt.datetime.combine(d, _dt.time(0,0))).tz_localize(_AMS_TZ)
            tmp['delivery_hour'] = h
            flat_frames.append(tmp)
        df_mo = pd.concat(flat_frames, ignore_index=True)
        return df.merge(df_mo, on=['delivery_date', 'delivery_hour', 'bin_timestamp'], how='left')

    # -----------------------------------
    # RES forecasts (DST rule integrated)
    # -----------------------------------
    def _res_forecast_table(self, res_csv, res_type):
        """
        Read RES CSV with 'Date (CET)' timestamps, localize to Europe/Amsterdam, and
        produce a DST-safe table over (delivery_date, delivery_hour) keeping FIRST 02:00
        on autumn day and duplicating 03:00 to fill missing 02:00 on spring day.
        """
        col_name = f'{res_type.upper()} DA FORECAST (TENNET)'
        res_df = pd.read_csv(res_csv, skiprows=[1])
        if "Date (CET)" not in res_df.columns:
            # sometimes brackets exist, keep compatible with original code
            cands = [c for c in res_df.columns if "Date" in c]
            if not cands:
                raise ValueError(f"Cannot find time column in {res_csv}")
            time_col = cands[0]
        else:
            time_col = "Date (CET)"

        res_df[time_col] = res_df[time_col].astype(str).str.strip("[]")
        idx = pd.to_datetime(res_df[time_col], format="%d/%m/%Y %H:%M", errors="coerce")
        idx = idx.dt.tz_localize(_AMS_TZ, ambiguous="infer")
        res_df.index = idx
        res_df = res_df[(res_df.index.year == self.target_year)]
        # Build (date,hour) + dst flag
        tmp = pd.DataFrame({
            'delivery_date': res_df.index.normalize(),
            'delivery_hour': res_df.index.hour,
            'dst_flag': res_df.index.map(lambda t: (t.dst() or _dt.timedelta(0)) != _dt.timedelta(0)),
            res_type.lower() + '_forecast': res_df[col_name].values
        })

        # AUTUMN: if duplicate hour==2, keep the first (dst_flag True)
        tmp = (tmp.sort_values(['delivery_date', 'delivery_hour', 'dst_flag'], ascending=[True, True, False])
                 .drop_duplicates(['delivery_date', 'delivery_hour'], keep='first'))

        # SPRING: ensure hour 2 exists by duplicating hour 3
        mask_spring = tmp['delivery_date'].map(lambda d: _is_spring_transition(d.date()))
        spring_days = tmp.loc[mask_spring, 'delivery_date'].unique()
        add_rows = []
        for d in spring_days:
            day_rows = tmp[tmp['delivery_date'] == d]
            if not (day_rows['delivery_hour'] == 2).any():
                row3 = day_rows[day_rows['delivery_hour'] == 3]
                if not row3.empty:
                    r = row3.iloc[0].copy()
                    r['delivery_hour'] = 2
                    add_rows.append(r)
        if add_rows:
            tmp = pd.concat([tmp, pd.DataFrame(add_rows)], ignore_index=True)

        return tmp[['delivery_date', 'delivery_hour', res_type.lower() + '_forecast']]

    def merge_res_forecasts(self, df, res_csv, res_type):
        res_tab = self._res_forecast_table(res_csv, res_type)
        merged_df = df.merge(res_tab, on=["delivery_date", "delivery_hour"], how="left")
        return merged_df

    # --------------------------------
    # Helpers for features / model I/O
    # --------------------------------
    @staticmethod
    def create_weekday_dummies(delivery_start_times):
        weekdays = delivery_start_times.dt.weekday
        return {
            'MON': (weekdays == 0).astype(int),
            'SAT': (weekdays == 5).astype(int),
            'SUN': (weekdays == 6).astype(int)
        }

    @staticmethod
    def prepare_hourly_arrays(df):
        hourly_data = {}
        for hour in range(24):
            hour_df = df[df['delivery_hour'] == hour]
            dtype = [
                ('vwap_changes', 'float64'),
                ('alpha', 'int64'),
                ('MON', 'int64'),
                ('SAT', 'int64'), 
                ('SUN', 'int64'),
                ('f_TTD', 'float64'),
            ]
            hourly_data[hour] = np.array(
                [tuple(x) for x in hour_df[['vwap_changes','alpha','MON','SAT','SUN','f_TTD']].values],
                dtype=dtype
            )
        return hourly_data

    def prepare_gamlss_data(self, product_type='hourly', start_hours=None, 
                            wind_csv='wind-renewables-elec.csv',
                            solar_csv='solar-renewables-elec.csv'):
        if start_hours is None:
            start_hours = self.start_hours

        # Aggregate using requested window
        agg_df = self.aggregate_extracted_data(product_type, start_hours)

        # Time-to-delivery bins relative to the effective (tz-aware) delivery start
        interval_td = pd.Timedelta(self.time_interval)
        interval_seconds = interval_td.total_seconds()
        agg_df['TTD_bins'] = (
            (agg_df['delivery_start_wall'] - agg_df['bin_timestamp']).dt.total_seconds() / interval_seconds
        )
        agg_df['f_TTD'] = 1.0 / np.sqrt(1.0 + agg_df['TTD_bins'])

        # Weekday dummies computed from delivery_date
        weekdays = agg_df['delivery_date'].dt.weekday
        agg_df['MON'] = (weekdays == 0).astype(int)
        agg_df['SAT'] = (weekdays == 5).astype(int)
        agg_df['SUN'] = (weekdays == 6).astype(int)

        # DA price per (date,hour) with DST rule
        agg_df['da_price'] = agg_df.apply(
            lambda r: self._get_auction_spot_price(r['delivery_date'].date(), int(r['delivery_hour'])),
            axis=1
        )

        # Merit-order slopes
        merged_df = self.merge_merit_order_slopes(agg_df)

        # RES forecasts
        merged_df = self.merge_res_forecasts(merged_df, wind_csv, 'wind')
        merged_df = self.merge_res_forecasts(merged_df, solar_csv, 'solar')

        merged_df['delivery_hour'] = merged_df['delivery_hour'].astype(int)

        # Absolute value columns & lags
        merged_df = self.add_absolute_value_columns(merged_df, ['vwap_changes'])
        merged_df = self.add_lagged_variables(
            merged_df,
            {
                'vwap': [1, 2, 3], 
                'vwap_changes': [1, 2, 3], 
                'alpha': list(range(1, 13)),
                'abs_vwap_changes': list(range(1, 13))
            }
        )

        # Spread features using effective delivery start
        merged_df["da-id"] = merged_df['da_price'] - merged_df['vwap']
        merged_df["da-id_lag1"] = merged_df['da_price'] - merged_df.get('vwap_lag1')
        merged_df["da-id_lag2"] = merged_df['da_price'] - merged_df.get('vwap_lag2')

        # Keep only rows within the configured window
        max_bins = int(round(start_hours * 3600 / interval_seconds))
        merged_df = merged_df[merged_df['TTD_bins'] <= max_bins].copy()

        # Create integer bin dummies
        merged_df['TTD_bins_int'] = np.floor(merged_df['TTD_bins']).astype(int)
        ttd_dummies = pd.get_dummies(merged_df['TTD_bins_int'], prefix='TTD_bin').astype(int)
        merged_df = pd.concat([merged_df, ttd_dummies], axis=1)

        # Convenience: also expose a tz-aware "representative" delivery_start (no uniqueness guaranteed)
        merged_df['delivery_start'] = merged_df['delivery_start_effective']
        dropped_cols = ['delivery_start_effective', 'TTD_bins_int']
        merged_df = merged_df.drop(columns=dropped_cols)

        return merged_df

    @staticmethod
    def add_lagged_variables(df, lag_dict):
        # IMPORTANT: group by (delivery_date, delivery_hour) to avoid collisions on DST days
        df = df.sort_values(by=['delivery_date', 'delivery_hour', 'bin_timestamp']).copy()
        group_keys = ['delivery_date', 'delivery_hour']
        for col in list(df.columns):
            if col in lag_dict:
                count_inserted = 0
                for lag in lag_dict[col]:
                    new_col_name = f"{col}_lag{lag}"
                    insertion_index = df.columns.get_loc(col) + 1 + count_inserted
                    new_series = df.groupby(group_keys)[col].shift(lag)
                    if new_col_name in df.columns:
                        df.drop(columns=[new_col_name], inplace=True)
                    df.insert(insertion_index, new_col_name, new_series)
                    count_inserted += 1
        return df

    @staticmethod
    def add_absolute_value_columns(df, columns):
        for col in columns:
            if col in df.columns:
                insertion_index = df.columns.get_loc(col) + 1
                new_col = f"abs_{col}"
                if new_col in df.columns:
                    df.drop(columns=[new_col], inplace=True)
                df.insert(insertion_index, new_col, df[col].abs())
        return df


    def plot_price_changes_histogram(
        self,
        product_type='hourly',
        start_hours=None,
        delivery_hour=None,
        bins=51,
        xlim=(-50, 50)                 # <— NEW: plotting window
    ):
        if start_hours is None:
            start_hours = self.start_hours
        if bins % 2 == 0:
            bins += 1
    
        agg_df = self.aggregate_extracted_data(product_type, start_hours)
        if delivery_hour is not None:
            agg_df = agg_df[agg_df['delivery_hour'] == delivery_hour]
    
        df_changes = agg_df[['vwap_changes', 'alpha']].dropna()
        changes = df_changes['vwap_changes'].to_numpy()
        alphas  = df_changes['alpha'].to_numpy()
    
        # Keep only data inside xlim for the histogram + axis
        lo, hi = xlim
        mask = (changes >= lo) & (changes <= hi)
        changes = changes[mask]
        alphas  = alphas[mask]
    
        bin_edges   = np.linspace(lo, hi, bins + 1)
        counts, _   = np.histogram(changes, bins=bin_edges)
        counts_a0,_ = np.histogram(changes[alphas == 0], bins=bin_edges)
    
        bin_width   = bin_edges[1] - bin_edges[0]
        bin_centers = bin_edges[:-1] + bin_width / 2
    
        # Find the zero bin index (the bin that contains 0)
        zero_bin_index = np.searchsorted(bin_edges, 0, side='right') - 1
        zero_bin_index = int(np.clip(zero_bin_index, 0, len(counts) - 1))
    
        base_color   = '#4C78A8'   # one uniform color for all bars
        special_color= '#E45756'   # only for the alpha=0 portion in the zero bin
    
        plt.figure(figsize=(10, 6))
        for i, c in enumerate(counts):
            if i == zero_bin_index:
                c_a0   = counts_a0[i]
                c_rest = c - c_a0
                # Base bar (non-alpha-0 portion) in the uniform color
                if c_rest > 0:
                    plt.bar(bin_centers[i], c_rest, width=bin_width,
                            edgecolor='black', align='center', color=base_color)
                # Overlay alpha=0 portion in the special color
                if c_a0 > 0:
                    plt.bar(bin_centers[i], c_a0, width=bin_width,
                            edgecolor='black', align='center', bottom=c_rest, color=special_color)
            else:
                plt.bar(bin_centers[i], c, width=bin_width,
                        edgecolor='black', align='center', color=base_color)
    
        plt.xlabel("VWAP Price Changes")
        plt.ylabel("Frequency")
        title = f"Histogram of VWAP Price Changes (Last {start_hours}h before delivery)"
        if delivery_hour is not None:
            title += f"\nfor Delivery Hour = {delivery_hour:02d}:00"
        plt.title(title)
        plt.xlim(lo, hi)
        plt.grid(True)
    
        legend_main = mpatches.Patch(color=base_color,    label='Intervals (α ≠ 0) + non-α=0 mass in zero bin')
        legend_zero = mpatches.Patch(color=special_color, label='α = 0 (zero bin only)')
        plt.legend(handles=[legend_main, legend_zero])
        plt.tight_layout()
        plt.show()


    def fit_distributions_to_price_changes(
        self,
        product_type='hourly',
        start_hours=None,
        delivery_hour=None,
        bins=51,
        plot=True,
        xlim=(-50, 50),              # <— NEW: plotting window
        trim_data=True               # <— NEW: optionally trim data before fitting/plotting
    ):
        if start_hours is None:
            start_hours = self.start_hours
    
        agg_df = self.aggregate_extracted_data(product_type, start_hours)
        if delivery_hour is not None:
            agg_df = agg_df[agg_df['delivery_hour'] == delivery_hour]
    
        valid_df = agg_df[agg_df['alpha'] != 0]
        changes_full = valid_df['vwap_changes'].dropna().to_numpy()
    
        # Optionally trim to focus on central mass in both fit & plot
        if trim_data:
            mask = (changes_full >= xlim[0]) & (changes_full <= xlim[1])
            changes = changes_full[mask]
        else:
            changes = changes_full
    
        # — Fit
        params_johnsonsu = stats.johnsonsu.fit(changes)
        params_t         = stats.t.fit(changes)
        params_norm      = stats.norm.fit(changes)
    
        # — KS (on the data you fitted)
        ks_johnsonsu = stats.kstest(changes, 'johnsonsu', args=params_johnsonsu)
        ks_t         = stats.kstest(changes, 't',         args=params_t)
        ks_norm      = stats.kstest(changes, 'norm',      args=params_norm)
    
        if plot:
            lo, hi = xlim
            x = np.linspace(lo, hi, 1000)
            pdf_johnsonsu = stats.johnsonsu.pdf(x, *params_johnsonsu)
            pdf_t         = stats.t.pdf(x, *params_t)
            pdf_norm      = stats.norm.pdf(x, *params_norm)
    
            # Histogram only in [lo, hi]
            bin_edges = np.linspace(lo, hi, bins + 1)
    
            plt.figure(figsize=(10, 6))
            plt.hist(changes, bins=bin_edges, density=True,
                     edgecolor='black', alpha=0.6, label='Data histogram')
            plt.plot(x, pdf_johnsonsu, '-', lw=2, label="Johnson's SU")
            plt.plot(x, pdf_t,         '-', lw=2, label="Student's t")
            plt.plot(x, pdf_norm,      '-', lw=2, label="Normal")
            plt.xlim(lo, hi)
            plt.xlabel("VWAP Price Changes")
            plt.ylabel("Density")
            title = "Fitted Distributions for VWAP Price Changes (alpha != 0)"
            if delivery_hour is not None:
                title += f"\nfor Delivery Hour = {delivery_hour:02d}:00"
            plt.title(title)
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
    
        return {
            'johnsonsu': {'params': params_johnsonsu, 'ks_statistic': ks_johnsonsu.statistic, 'p_value': ks_johnsonsu.pvalue},
            'student_t': {'params': params_t,         'ks_statistic': ks_t.statistic,         'p_value': ks_t.pvalue},
            'normal':    {'params': params_norm,      'ks_statistic': ks_norm.statistic,      'p_value': ks_norm.pvalue},
        }


    def plot_data(self, delivery_start=None, delivery_date=None, delivery_hour=None,
                  product_type='hourly', start_hours=None):
        """
        Backwards compatible:
          - If delivery_start (tz-aware ts) is given, infer date/hour and plot.
          - Otherwise, use (delivery_date, delivery_hour).
        """
        if start_hours is None:
            start_hours = self.start_hours

        if delivery_start is not None:
            delivery_date = delivery_start.normalize().date()
            delivery_hour = int(delivery_start.hour)

        if delivery_date is None or delivery_hour is None:
            raise ValueError("Provide either delivery_start or (delivery_date, delivery_hour).")

        data_dict = self.extract_all_data(product_type, start_hours)
        key = (delivery_date if isinstance(delivery_date, _dt.date) else delivery_date.date(), int(delivery_hour))
        if key not in data_dict:
            print(f"No data found for Delivery: {key}")
            return
        eff_ts, df_result = data_dict[key]

        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        axes[0].step(df_result.index, df_result['vwap'], where='post', linestyle='-')
        axes[0].set_title("VWAP")
        axes[0].set_ylabel("Price")
        axes[0].grid(True)

        axes[1].step(df_result.index, df_result['vwap_changes'], where='post', linestyle='-', color='orange')
        axes[1].set_title("VWAP Price Changes")
        axes[1].set_ylabel("Price Change")
        axes[1].grid(True)

        axes[2].step(df_result.index, df_result['alpha'], where='post', color='green')
        axes[2].set_title("Alpha (Trade Indicator)")
        axes[2].set_ylabel("Alpha")
        axes[2].set_xlabel("Execution Time")
        axes[2].grid(True)

        plt.tight_layout()
        plt.show()

    def plot_total_traded_volume_per_delivery_hour(self):
        df = self.data.copy()
        tol = 1e-6
        conditions = [
            np.abs(df['ProductDuration'] - 60) < tol,
            np.abs(df['ProductDuration'] - 30) < tol,
            np.abs(df['ProductDuration'] - 15) < tol
        ]
        choices = ['hourly', 'half_hourly', 'quarterly']
        df['product_type'] = np.select(conditions, choices, default='other')
        df = df[df['product_type'] != 'other']
        df['delivery_hour'] = df['Delivery Start'].dt.hour
        volume_pivot = df.groupby(['delivery_hour', 'product_type'])['Quantity (MW)'].sum().unstack()

        volume_pivot.plot(kind='bar', figsize=(10, 6))
        plt.xlabel("Delivery Hour")
        plt.ylabel("Total Traded Volume (MW)")
        plt.title("Total Traded Volume per Delivery Hour by Product Type")
        plt.legend(title="Product Type")
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.show()


# === Example Usage ===
if __name__ == '__main__':
    # 6-hour window, 15-minute bins
    lob_2021 = LimitOrderBookProcessor(
        zip_file="Continuous_Trades_NL_2021.zip",
        time_interval="15min",
        start_hours=4,
        auction_spot_csv='auction_spot_prices_netherlands_2021.csv',
        aggregator_zip="auction_aggregated_curves_netherlands_2021.zip"
    )
    

    lob_2021.fit_distributions_to_price_changes(start_hours = 4, delivery_hour = 12, xlim=(-50,50))
    lob_2021.plot_price_changes_histogram(start_hours = 4, xlim=(-100,100))
    lob_2021.plot_price_changes_histogram(start_hours = 4, delivery_hour = 12, xlim=(-50,50))
    lob_2021.plot_price_changes_histogram(start_hours = 4, delivery_hour = 8, xlim=(-50,50))
    lob_2021.plot_price_changes_histogram(start_hours = 4, delivery_hour = 1, xlim=(-50,50))
    lob_2021.plot_price_changes_histogram(start_hours = 4, delivery_hour = 17, xlim=(-50,50))
    
    delivery_start = pd.Timestamp("2021-10-05 03:00", tz="Europe/Amsterdam")


    lob_2021.plot_data(delivery_start=delivery_start, start_hours = 4)
    lob_2021.plot_total_traded_volume_per_delivery_hour()

    # datas_2021 = lob_2021.prepare_gamlss_data()  
    
    # lob_2022 = LimitOrderBookProcessor(
    #     zip_file="Continuous_Trades_NL_2022.zip",
    #     time_interval="15min",
    #     start_hours=4,
    #     auction_spot_csv='auction_spot_prices_netherlands_2022.csv',
    #     aggregator_zip="auction_aggregated_curves_netherlands_2022.zip"
    # )
    # datas_2022 = lob_2022.prepare_gamlss_data()

    # datas_2021.to_csv("df_2021_MVT_dst_10.csv", index=False)
    # datas_2022.to_csv("df_2022_MVT_dst_10.csv", index=False)
