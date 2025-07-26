import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score, make_scorer, classification_report, confusion_matrix
import joblib
import os


X_train = pd.read_csv("data/train_ready.csv")
X_val = pd.read_csv("data/validation_ready.csv")
y_train = pd.read_csv("data/y_train.csv").values.ravel()
y_val = pd.read_csv("data/y_val.csv").values.ravel()

print("\n--- Fase 4.2 (Rápida): Tunning do Random Forest com RandomizedSearchCV ---\n")


rf = RandomForestClassifier(random_state=42, n_jobs=-1)

param_dist = {
    'n_estimators': [80, 100, 120, 150, 200],
    'max_depth': [None, 10, 20, 30, 40],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4, 6]
}


scorer = make_scorer(f1_score, pos_label=">50K")


random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=30,             
    scoring=scorer,
    cv=3,
    n_jobs=-1,
    verbose=2,
    random_state=42
)

random_search.fit(X_train, y_train)

print("\nMelhores parâmetros encontrados:")
print(random_search.best_params_)
print(f"Melhor F1-score médio (cross-validation): {random_search.best_score_:.4f}")


best_rf = random_search.best_estimator_
y_pred = best_rf.predict(X_val)

f1 = f1_score(y_val, y_pred, pos_label=">50K")
print("\nDesempenho no conjunto de validação:")
print(classification_report(y_val, y_pred))
print(f"F1-Score (classe '>50K'): {f1:.4f}")
print("\nMatriz de Confusão:")
print(confusion_matrix(y_val, y_pred))


os.makedirs("resultados", exist_ok=True)
with open("resultados/random_forest_randomized_metrics.txt", "w") as f:
    f.write("=== Random Forest RandomizedSearchCV ===\n")
    f.write(f"Melhores parâmetros: {random_search.best_params_}\n")
    f.write(f"F1-score de validação: {f1:.4f}\n")
    f.write("\n=== Classification Report ===\n")
    f.write(classification_report(y_val, y_pred))

joblib.dump(best_rf, "resultados/modelo_random_forest_randomized.pkl")
print("\nModelo ajustado salvo em 'resultados/modelo_random_forest_randomized.pkl'.")

