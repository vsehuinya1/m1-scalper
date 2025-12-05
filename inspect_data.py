import pandas as pd
import glob
files = glob.glob('data/binance_1m/*.parquet')
print('Files:', files)
for f in files[:2]:
    df = pd.read_parquet(f)
    print(f, 'rows:', len(df))
    if len(df) > 0:
        print('timestamp range:', df['timestamp'].min(), 'to', df['timestamp'].max())
        print('columns:', df.columns.tolist())
