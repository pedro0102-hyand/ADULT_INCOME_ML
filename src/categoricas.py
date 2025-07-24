import pandas as pd

# =========================
# 1. Carregar datasets tratados (com outliers ajustados)
# =========================
train_df = pd.read_csv("data/train_clean_outliers.csv")
validation_df = pd.read_csv("data/validation_clean_outliers.csv")
test_df = pd.read_csv("data/test_clean_outliers.csv")

print("\n--- Etapa 2.3: One-Hot Encoding ---\n")

# =========================
# 2. Identificar variáveis categóricas (removendo a variável alvo)
# =========================
categorical_cols = train_df.select_dtypes(include=["object"]).columns
categorical_cols = [col for col in categorical_cols if col != "income"]
print(f"Variáveis categóricas: {list(categorical_cols)}")

# =========================
# 3. Separar X (features) e y (alvo)
# =========================
target_col = "income"
y_train = train_df[target_col]
y_val = validation_df[target_col]
y_test = test_df[target_col]

X_train = train_df.drop(columns=[target_col])
X_val = validation_df.drop(columns=[target_col])
X_test = test_df.drop(columns=[target_col])

X_train = pd.get_dummies(X_train, columns=categorical_cols)
X_val = pd.get_dummies(X_val, columns=categorical_cols)
X_test = pd.get_dummies(X_test, columns=categorical_cols)

X_val = X_val.reindex(columns=X_train.columns, fill_value=0)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

X_train.to_csv("data/train_encoded.csv", index=False)
X_val.to_csv("data/validation_encoded.csv", index=False)
X_test.to_csv("data/test_encoded.csv", index=False)

y_train.to_csv("data/y_train.csv", index=False)
y_val.to_csv("data/y_val.csv", index=False)
y_test.to_csv("data/y_test.csv", index=False)

print("One-Hot Encoding concluído. Arquivos salvos em 'data/'.")

