import pandas as pd #manipulacao de dados
import matplotlib.pyplot as plt #gerar gráficos

#leitura dos arquivos e transformacao em dataframe
train_df = pd.read_csv("data/train.csv")
validation_df = pd.read_csv("data/validation.csv")
test_df = pd.read_csv("data/test.csv")

print("Primeiras linhas do train.csv:")
print(train_df.head(), "\n")

# Contagem da variável 
target_counts = train_df['income'].value_counts()
target_percentage = (target_counts / len(train_df)) * 100

# Criar DataFrame com contagem e porcentagem
target_info = pd.DataFrame({
    'Contagem': target_counts,
    'Porcentagem (%)': target_percentage.round(2)
})

print("Distribuição da variável alvo (income):")
print(target_info, "\n")

#  Plotar gráfico de barras
plt.figure(figsize=(6,4))
train_df['income'].value_counts().plot(kind='bar', color=['skyblue', 'orange'])
plt.title('Distribuição da variável alvo (income) - Train Dataset')
plt.xlabel('Classe de Renda')
plt.ylabel('Contagem')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()