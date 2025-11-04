import pandas as pd
import numpy as np




df_api = pd.read_csv(
    "netherlands_load_2024.csv",
    skiprows= 1,                             
    names=["DateTime", "Load_api"],       
    usecols=[0, 1]                            
)

df_sftp = pd.read_csv("2024_01_ActualTotalLoad_6.1.A.csv", sep="\t")


for df, col in [(df_api, "DateTime"), (df_sftp, "DateTime")]:
    df[col] = (pd.to_datetime(df[col].str.strip(), utc=True,errors="coerce").dt.tz_localize(None))


df_nl = (
    df_sftp.loc[df_sftp["MapCode"].eq("NL"), ["DateTime", "TotalLoadValue"]]
           .drop_duplicates("DateTime")            
           .rename(columns={"TotalLoadValue": "Load_sftp"})
)


merged = (
    df_api.merge(df_nl, on="DateTime", how="inner")
          .sort_values("DateTime")
          .assign(Deviation   = lambda d: d["Load_api"] - d["Load_sftp"], AbsDeviation= lambda d: (d["Load_api"] - d["Load_sftp"]).abs())
)


largest_dev = merged.loc[merged["AbsDeviation"].idxmax(), ["DateTime", "Load_api", "Load_sftp", "Deviation"]]


monthly_dev = (
    merged.assign(Month=merged["DateTime"].dt.to_period("M"))
          .groupby("Month", sort=False)["AbsDeviation"].sum()
          .reset_index(name="Total Absolute Deviation")
)


hourly_dev = (
    merged.assign(Hour=merged["DateTime"].dt.hour)
          .groupby("Hour", sort=True)["Deviation"].sum()
          .reset_index(name="Total Deviation")
)


print("\nLargest devation:")
print(largest_dev)

print("\nTotal absolute deviation per month:")
print(monthly_dev.to_string(index=False))

print("\nTotal deviaion per hour")
print(hourly_dev.to_string(index=False))
