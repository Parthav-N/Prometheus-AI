import pandas as pd

df = pd.read_csv('data/xgboost_training_labels.csv')

# Remove Type B negatives — keep only same-radius samples (0-50km)
df_clean = df[df['min_dist_km'] <= 50].copy()

print(f'Before : {len(df):,}')
print(f'After  : {len(df_clean):,}')
print(f'Positive: {(df_clean["outage_label"]==1).sum():,}')
print(f'Negative: {(df_clean["outage_label"]==0).sum():,}')
print(f'Ratio   : {df_clean["outage_label"].mean():.2%}')

print('\nn_fires_50km by label (after fix):')
print(df_clean.groupby('outage_label')['n_fires_50km'].describe().round(2))

print('\nmin_dist_km by label (after fix):')
print(df_clean.groupby('outage_label')['min_dist_km'].describe().round(2))

df_clean.to_csv('data/xgboost_training_labels.csv', index=False)
print('\nSaved.')