import pandas as pd

df = pd.read_csv("observations-660880.csv")

df_filtered = df.groupby("scientific_name").filter(lambda x: len(x) >= 3)

df_filtered.to_csv("observations_filtered.csv", index=False)

print("Было объектов:", len(df))
print("Стало объектов:", len(df_filtered))
print("Осталось классов:", df_filtered["scientific_name"].nunique())
print("\nРаспределение по классам:")
print(df_filtered["scientific_name"].value_counts())
