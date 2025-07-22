
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # heatmap de correlação

train_df = pd.read_csv("data/train.csv")

print("\n--- Etapa 4: Correlações e Relações com a Variável Alvo ---\n")

#  Matriz de correlação para variáveis numéricas
numeric_cols = train_df.select_dtypes(include=['int64', 'float64']).columns
correlation_matrix = train_df[numeric_cols].corr()

print("Matriz de correlação (variáveis numéricas):\n")
print(correlation_matrix, "\n")

#  heatmap de correlação
plt.figure(figsize=(8,6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title('Matriz de Correlação - Variáveis Numéricas')
plt.tight_layout()
plt.show()

#  Analisar a relação entre variáveis categóricas e income
categorical_cols = train_df.select_dtypes(include=['object']).columns
categorical_cols = [col for col in categorical_cols if col != 'income']

for col in categorical_cols:
    print(f"\nDistribuição de {col} por classe (income):\n")
    print(train_df.groupby([col, 'income']).size().unstack(fill_value=0))

    plt.figure(figsize=(8,4))
    sns.countplot(data=train_df, x=col, hue='income')
    plt.title(f'Relação entre {col} e income')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

#  Distribuições numéricas separadas por classe
for col in numeric_cols:
    plt.figure(figsize=(6,4))
    sns.kdeplot(data=train_df, x=col, hue="income", fill=True)
    plt.title(f'Distribuição de {col} por classe (income)')
    plt.tight_layout()
    plt.show()
