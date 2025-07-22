import pandas as pd 

# 1. Carregar os datasets
train_df = pd.read_csv("data/train.csv")
validation_df = pd.read_csv("data/validation.csv")
test_df = pd.read_csv("data/test.csv")

print("\n--- Etapa 2: Verificação de Valores Ausentes e Inconsistências ---\n")

#  Verificar valores ausentes 
print("Valores ausentes (NaN) em train.csv:\n")
print(train_df.isnull().sum(), "\n")

print("Valores ausentes (NaN) em validation.csv:\n")
print(validation_df.isnull().sum(), "\n")

print("Valores ausentes (NaN) em test.csv:\n")
print(test_df.isnull().sum(), "\n")

# Verificar valores inválidos representados por '?'
print("Valores '?' (desconhecidos) em train.csv:\n")
for col in train_df.columns:
    qtd = (train_df[col] == '?').sum()
    if qtd > 0:
        print(f"{col}: {qtd} valores")
print()

print("Valores '?' (desconhecidos) em validation.csv:\n")
for col in validation_df.columns:
    qtd = (validation_df[col] == '?').sum()
    if qtd > 0:
        print(f"{col}: {qtd} valores")
print()

print("Valores '?' (desconhecidos) em test.csv:\n")
for col in test_df.columns:
    qtd = (test_df[col] == '?').sum()
    if qtd > 0:
        print(f"{col}: {qtd} valores")
print()

print("Informações gerais do train.csv:\n")
print(train_df.info())
print("\nDescrição estatística das variáveis numéricas do train.csv:\n")
print(train_df.describe())