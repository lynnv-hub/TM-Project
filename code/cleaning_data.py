import pandas as pd
import glob
import os

files = glob.glob(os.path.join("data\\", "*.csv"))

for f in files:
    df = pd.read_csv(f, dtype=str)
    rows_start = len(df)
    #removing arabic cnn
    df = df[~df["SOURCEURL"].str.contains("arabic.cnn", na=False)] 
    #removing duplicates
    df = df.drop_duplicates(subset=["SOURCEURL"])
    rows_end = len(df)
    base = os.path.splitext(os.path.basename(f))[0]
    out_path = os.path.join("data\\", f"{base}_cleaned.csv")
    df.to_csv(out_path, index=False)
    print(f"{base}: from {rows_start} rows to {rows_end} rows")

df = pd.read_csv("data\\US2016.csv", dtype=str)
rows_start = len(df)
#removing arabic cnn
df = df[~df["SOURCEURL"].str.contains("arabic.cnn", na=False)] 
#removing duplicates
df = df.drop_duplicates(subset=["SOURCEURL"])
rows_end = len(df)
out_path = os.path.join("data\\", f"US2016_cleaned.csv")
df.to_csv(out_path, index=False)
print(f"US2016: from {rows_start} rows to {rows_end} rows")
