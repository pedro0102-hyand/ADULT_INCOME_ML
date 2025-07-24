import pandas as pd

# =========================
# 1. Carregar datasets prontos (após escalonamento)
# =========================
X_train = pd.read_csv("data/train_ready.csv")
X_val = pd.read_csv("data/validation_ready.csv")
X_test = pd.read_csv("data/test_ready.csv")

y_train = pd.read_csv("data/y_train.csv")
y_val = pd.read_csv("data/y_val.csv")
y_test = pd.read_csv("data/y_test.csv")

print("\n--- Etapa 2.5: Verificação Final dos Dados ---\n")

# =========================
# 2. Verificar shapes
# =========================
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}\n")

# =========================
# 3. Verificar se as colunas dos datasets são consistentes
# =========================
if list(X_train.columns) == list(X_val.columns) == list(X_test.columns):
    print("As colunas de X_train, X_val e X_test estão consistentes.\n")
else:
    print("As colunas NÃO estão consistentes. Verifique o One-Hot Encoding.\n")

# =========================
# 4. Verificar primeiras linhas (opcional, apenas inspeção)
# =========================
print("Primeiras linhas de X_train:")
print(X_train.head(3))
print("\nPrimeiras linhas de y_train:")
print(y_train.head(3))

# =========================
# 5. Mensagem final
# =========================
print("\nPreparação final concluída. Dados prontos para a Fase 3.")
