import pandas as pd
from sklearn.preprocessing import StandardScaler

X_train = pd.read_csv("data/train_encoded.csv")
X_val = pd.read_csv("data/validation_encoded.csv")
X_test = pd.read_csv("data/test_encoded.csv")

y_train = pd.read_csv("data/y_train.csv")
y_val = pd.read_csv("data/y_val.csv")
y_test = pd.read_csv("data/y_test.csv")

print("\n--- Etapa 2.4: Escalonamento das Variáveis Numéricas ---\n")

numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
print(f"Variáveis numéricas: {list(numeric_cols)}")

scaler = StandardScaler()
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_val[numeric_cols] = scaler.transform(X_val[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

X_train.to_csv("data/train_ready.csv", index=False)
X_val.to_csv("data/validation_ready.csv", index=False)
X_test.to_csv("data/test_ready.csv", index=False)

y_train.to_csv("data/y_train.csv", index=False)
y_val.to_csv("data/y_val.csv", index=False)
y_test.to_csv("data/y_test.csv", index=False)

print("Escalonamento concluído. Arquivos prontos para modelagem salvos em 'data/'.")
