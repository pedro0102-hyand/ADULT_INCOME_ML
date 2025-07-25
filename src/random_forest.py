import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, precision_score, recall_score
import joblib #salvar os modelos
import os #criar diretorios

#leitura dos datasets
X_train = pd.read_csv("data/train_ready.csv")
X_val = pd.read_csv("data/validation_ready.csv")
y_train = pd.read_csv("data/y_train.csv").values.ravel()
y_val = pd.read_csv("data/y_val.csv").values.ravel()

print("\n--- Fase 4: Random Forest Classifier ---\n")

#criacao do modelo
rf_model = RandomForestClassifier(
    n_estimators=100, #numero de arvores
    max_depth=None, #sem limite de profundidade para as arvores
    random_state=42,
    n_jobs=-1 #uso de nucleos do processador
)

#treinamento dos modelos com os dados
rf_model.fit(X_train, y_train)

#estabelecimento das previsoes
y_pred = rf_model.predict(X_val)

f1 = f1_score(y_val, y_pred, pos_label=">50K") #equilibrio entre precisao e recall
accuracy = accuracy_score(y_val, y_pred) #proporcao de acertos
precision = precision_score(y_val, y_pred, pos_label=">50K") #veracidade dos acertos
recall = recall_score(y_val, y_pred, pos_label=">50K") #deteccao dos acertos

print("Relatório de classificação (Validation Set):")
print(classification_report(y_val, y_pred))
print(f"F1-Score (classe '>50K'): {f1:.4f}")
print(f"Acurácia: {accuracy:.4f}")
print(f"Precisão: {precision:.4f}")
print(f"Recall: {recall:.4f}")

print("\nMatriz de Confusão:")
print(confusion_matrix(y_val, y_pred))

os.makedirs("resultados", exist_ok=True)
with open("resultados/random_forest_metrics.txt", "w") as f:
    f.write("=== Modelo Random Forest ===\n")
    f.write(f"F1-Score (>50K): {f1:.4f}\n")
    f.write(f"Acurácia: {accuracy:.4f}\n")
    f.write(f"Precisão (>50K): {precision:.4f}\n")
    f.write(f"Recall (>50K): {recall:.4f}\n\n")
    f.write("=== Classification Report ===\n")
    f.write(classification_report(y_val, y_pred))

print("\nMétricas salvas em 'resultados/random_forest_metrics.txt'.")

joblib.dump(rf_model, "resultados/modelo_random_forest.pkl")
print("Modelo Random Forest salvo em 'resultados/modelo_random_forest.pkl'.")
