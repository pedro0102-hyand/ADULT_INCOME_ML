import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, precision_score, recall_score, make_scorer
import joblib
import os

# =========================
# 1. Carregar datasets
# =========================
X_train = pd.read_csv("data/train_ready.csv")
X_val = pd.read_csv("data/validation_ready.csv")
y_train = pd.read_csv("data/y_train.csv").values.ravel()
y_val = pd.read_csv("data/y_val.csv").values.ravel()

print("\n--- Fase 4.4: Tuning Rápido do Gradient Boosting ---\n")

# =========================
# 2. Modelo base
# =========================
gb = GradientBoostingClassifier(random_state=42)

# =========================
# 3. Espaço de busca reduzido
# =========================
param_dist = {
    'n_estimators': [100, 150, 200],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'subsample': [0.9, 1.0],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Scorer principal
scorer = make_scorer(f1_score, pos_label=">50K")

# =========================
# 4. Randomized Search rápido
# =========================
random_search = RandomizedSearchCV(
    estimator=gb,
    param_distributions=param_dist,
    n_iter=10,        # apenas 10 combinações
    scoring=scorer,
    cv=2,             # 2-fold CV (mais rápido)
    random_state=42,
    verbose=2,
    n_jobs=-1
)

random_search.fit(X_train, y_train)

# =========================
# 5. Avaliar modelo
# =========================
print("\nMelhores parâmetros encontrados:")
print(random_search.best_params_)
print(f"Melhor F1-score médio (cross-validation): {random_search.best_score_:.4f}")

best_gb = random_search.best_estimator_
y_pred = best_gb.predict(X_val)

f1 = f1_score(y_val, y_pred, pos_label=">50K")
accuracy = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred, pos_label=">50K")
recall = recall_score(y_val, y_pred, pos_label=">50K")

print("\nDesempenho no conjunto de validação:")
print(classification_report(y_val, y_pred))
print(f"F1-Score (classe '>50K'): {f1:.4f}")
print(f"Acurácia: {accuracy:.4f}")
print(f"Precisão: {precision:.4f}")
print(f"Recall: {recall:.4f}")

print("\nMatriz de Confusão:")
print(confusion_matrix(y_val, y_pred))

# =========================
# 6. Salvar resultados
# =========================
os.makedirs("resultados", exist_ok=True)
with open("resultados/gradient_boosting_tuning_fast.txt", "w") as f:
    f.write("=== Gradient Boosting Tuning Rápido ===\n")
    f.write(f"Melhores parâmetros: {random_search.best_params_}\n")
    f.write(f"F1-score de validação: {f1:.4f}\n")
    f.write("\n=== Classification Report ===\n")
    f.write(classification_report(y_val, y_pred))

joblib.dump(best_gb, "resultados/modelo_gradient_boosting_tuning_fast.pkl")
print("\nModelo ajustado salvo em 'resultados/modelo_gradient_boosting_tuning_fast.pkl'.")
