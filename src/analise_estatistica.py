import pandas as pd 
import matplotlib.pyplot as plt 

#  Carregar dataset de treino
train_df = pd.read_csv("data/train.csv")

print("\n--- Etapa 3: Análise Estatística e EDA ---\n")

# exibir estatísticas padrao
print("Estatísticas das variáveis numéricas:\n")
print(train_df.describe(), "\n")

#  Histogramas das variáveis numéricas
numeric_cols = train_df.select_dtypes(include=['int64', 'float64']).columns
print(f"Variáveis numéricas: {list(numeric_cols)}\n")

for col in numeric_cols:
    plt.figure(figsize=(6,4))
    plt.hist(train_df[col], bins=30, color='skyblue', edgecolor='black')
    plt.title(f'Distribuição da variável: {col}')
    plt.xlabel(col)
    plt.ylabel('Frequência')
    plt.tight_layout()
    plt.show()

# boxplot
for col in numeric_cols:
    plt.figure(figsize=(6,4))
    plt.boxplot(train_df[col], vert=False)
    plt.title(f'Boxplot da variável: {col}')
    plt.xlabel(col)
    plt.tight_layout()
    plt.show()

# Contagem das variáveis categóricas
categorical_cols = train_df.select_dtypes(include=['object']).columns
print(f"Variáveis categóricas: {list(categorical_cols)}\n")

for col in categorical_cols:
    print(f"Distribuição de {col}:\n")
    print(train_df[col].value_counts(), "\n")

    plt.figure(figsize=(8,4))
    train_df[col].value_counts().plot(kind='bar', color='orange')
    plt.title(f'Distribuição da variável: {col}')
    plt.xlabel(col)
    plt.ylabel('Contagem')
    plt.tight_layout()
    plt.show()