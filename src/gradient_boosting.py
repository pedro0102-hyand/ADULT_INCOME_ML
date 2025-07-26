import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, precision_score, recall_score
import joblib
import os


X_train = pd.read_csv("data/train_ready.csv")
X_val = pd.read_csv("data/validation_ready.csv")
y_train = pd.read_csv("data/y_train.csv").values.ravel()
y_val = pd.read_csv("data/y_val.csv").values.ravel()

print("\n--- Fase 4.3: Gradient Boosting Classifier ---\n")


gb_model = GradientBoostingClassifier(
    n_estimators=200,     # número de árvores
    learning_rate=0.1,    # taxa de aprendizado
    max_depth=3,          # profundidade das árvores
    random_state=42
)

gb_model.fit(X_train, y_train)


y_pred = gb_model.predict(X_val)

f1 = f1_score(y_val, y_pred, pos_label=">50K")
accuracy = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred, pos_label=">50K")
recall = recall_score(y_val, y_pred, pos_label=">50K")

print("Relatório de classificação (Validation Set):")
print(classification_report(y_val, y_pred))
print(f"F1-Score (classe '>50K'): {f1:.4f}")
print(f"Acurácia: {accuracy:.4f}")
print(f"Precisão: {precision:.4f}")
print(f"Recall: {recall:.4f}")

print("\nMatriz de Confusão:")
print(confusion_matrix(y_val, y_pred))


os.makedirs("resultados", exist_ok=True)
with open("resultados/gradient_boosting_metrics.txt", "w") as f:
    f.write("=== Modelo Gradient Boosting ===\n")
    f.write(f"F1-Score (>50K): {f1:.4f}\n")
    f.write(f"Acurácia: {accuracy:.4f}\n")
    f.write(f"Precisão (>50K): {precision:.4f}\n")
    f.write(f"Recall (>50K): {recall:.4f}\n\n")
    f.write("=== Classification Report ===\n")
    f.write(classification_report(y_val, y_pred))

joblib.dump(gb_model, "resultados/modelo_gradient_boosting.pkl")
print("\nModelo Gradient Boosting salvo em 'resultados/modelo_gradient_boosting.pkl'.")
