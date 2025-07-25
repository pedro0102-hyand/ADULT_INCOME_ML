import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, precision_score, recall_score
import joblib #salvar e carregar modelos treinados
import os #criar diretorios e trabalhar com arquivos

X_train = pd.read_csv("data/train_ready.csv")
X_val = pd.read_csv("data/validation_ready.csv")
# transformacao da coluna alvo em array 1D
y_train = pd.read_csv("data/y_train.csv").values.ravel()
y_val = pd.read_csv("data/y_val.csv").values.ravel()

print("\n--- Fase 3: Modelo Baseline (Logistic Regression) ---\n")

#instancia do modelo de regressao logistica
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train) #ajuste do modelo de acordo com os dados

y_pred = model.predict(X_val) #modelo preditivo para dados avaliativos

f1 = f1_score(y_val, y_pred, pos_label=">50K") #equilibrio entre precisao e recall
accuracy = accuracy_score(y_val, y_pred) #percentual de acertos
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
with open("resultados/baseline_metrics.txt", "w") as f:
    f.write("=== Modelo Baseline (Logistic Regression) ===\n")
    f.write(f"F1-Score (>50K): {f1:.4f}\n")
    f.write(f"Acurácia: {accuracy:.4f}\n")
    f.write(f"Precisão (>50K): {precision:.4f}\n")
    f.write(f"Recall (>50K): {recall:.4f}\n\n")
    f.write("=== Classification Report ===\n")
    f.write(classification_report(y_val, y_pred))

print("\nMétricas salvas em 'resultados/baseline_metrics.txt'.")

joblib.dump(model, "resultados/modelo_baseline.pkl")
print("Modelo baseline salvo em 'resultados/modelo_baseline.pkl'.")
