import pandas as pd

train_df = pd.read_csv("data/train_clean.csv")

#  Função para detectar outliers usando IQR
def detectar_outliers(df, coluna):
    Q1 = df[coluna].quantile(0.25)
    Q3 = df[coluna].quantile(0.75)
    IQR = Q3 - Q1
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR
    outliers = df[(df[coluna] < limite_inferior) | (df[coluna] > limite_superior)]
    return len(outliers), limite_inferior, limite_superior


variaveis_numericas = ['age', 'hours-per-week', 'capital-gain', 'capital-loss']

#  Detectar outliers em cada variável
for var in variaveis_numericas:
    qtd, li, ls = detectar_outliers(train_df, var)
    print(f"Outliers na variável {var}: {qtd} (Limites: {li:.2f} a {ls:.2f})")

print("\nEstatísticas gerais das variáveis analisadas:")
print(train_df[variaveis_numericas].describe())
