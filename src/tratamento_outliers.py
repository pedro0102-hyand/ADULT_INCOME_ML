import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

train_df = pd.read_csv("data/train_clean.csv")
validation_df = pd.read_csv("data/validation_clean.csv")
test_df = pd.read_csv("data/test_clean.csv")

variaveis_numericas = ['age', 'hours-per-week', 'capital-gain', 'capital-loss']

os.makedirs("relatorio/graficos", exist_ok=True)

for var in variaveis_numericas:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=train_df[var], color="skyblue")
    plt.title(f"Boxplot - {var}")
    plt.savefig(f"relatorio/graficos/boxplot_{var}.png")
    plt.close()

    plt.figure(figsize=(6,4))
    sns.histplot(train_df[var], bins=30, kde=True, color="orange")
    plt.title(f"Histograma - {var}")
    plt.savefig(f"relatorio/graficos/histograma_{var}.png")
    plt.close()

print("\nBoxplots e histogramas salvos em 'relatorio/graficos/'.")


limites = {
    'age': (17, 78),
    'hours-per-week': (10, 80),
    'capital-gain': (0, 40000),
    'capital-loss': (0, 2000)
}


def tratar_outliers(df):
    for col, (li, ls) in limites.items():
        df[col] = df[col].clip(lower=li, upper=ls)
    return df

train_df = tratar_outliers(train_df)
validation_df = tratar_outliers(validation_df)
test_df = tratar_outliers(test_df)


train_df.to_csv("data/train_clean_outliers.csv", index=False)
validation_df.to_csv("data/validation_clean_outliers.csv", index=False)
test_df.to_csv("data/test_clean_outliers.csv", index=False)

print("\nTratamento de outliers concluído. Novos datasets salvos.")
