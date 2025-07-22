import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import os

# =========================
# FUNÇÃO AUXILIAR PARA SALVAR GRÁFICOS
# =========================
def salvar_grafico(fig, nome_arquivo):
    caminho = f"relatorio/graficos/{nome_arquivo}.png"
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    fig.savefig(caminho, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return caminho

# =========================
# 1. CARREGAR DATASETS
# =========================
train_df = pd.read_csv("data/train.csv")

# =========================
# 2. ANÁLISES
# =========================
# Variável alvo
target_counts = train_df['income'].value_counts()
target_percent = (target_counts / len(train_df)) * 100

# Criar gráfico da variável alvo
fig, ax = plt.subplots(figsize=(6,4))
target_counts.plot(kind='bar', color=['skyblue', 'orange'], ax=ax)
ax.set_title('Distribuição da variável alvo (income)')
ax.set_xlabel('Classe')
ax.set_ylabel('Contagem')
grafico_target = salvar_grafico(fig, "variavel_alvo")

# Heatmap de correlação
numeric_cols = train_df.select_dtypes(include=['int64', 'float64']).columns
corr_matrix = train_df[numeric_cols].corr()
fig, ax = plt.subplots(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
ax.set_title("Matriz de Correlação")
grafico_corr = salvar_grafico(fig, "heatmap_correlacao")

# Boxplot de algumas variáveis
outlier_plots = []
for col in ['age', 'hours-per-week', 'capital-gain', 'capital-loss']:
    fig, ax = plt.subplots(figsize=(6,3))
    sns.boxplot(x=train_df[col], ax=ax)
    ax.set_title(f'Boxplot - {col}')
    path = salvar_grafico(fig, f"boxplot_{col}")
    outlier_plots.append(path)


