import os
import pandas as pd

# Caminho da pasta onde estão as métricas
resultados_path = "resultados"

# Arquivos de métricas esperados
arquivos_metricas = {
    "Baseline (Logistic Regression)": "baseline_metrics.txt",
    "Random Forest": "random_forest_metrics.txt",
    "Random Forest Tuning": "random_forest_randomized_metrics.txt",
    "Gradient Boosting": "gradient_boosting_metrics.txt",
    "Gradient Boosting Tuning": "gradient_boosting_tuning_fast.txt"
}

def extrair_f1_score(caminho):
    """Extrai o F1-Score (>50K) de um arquivo de métricas."""
    if not os.path.exists(caminho):
        return None
    with open(caminho, "r") as f:
        for linha in f:
            if "F1-Score" in linha or "F1-score" in linha:
                try:
                    valor = float(linha.split(":")[-1].strip())
                    return round(valor, 4)
                except:
                    pass
    return None

def comparar_modelos():
    dados = []
    for modelo, arquivo in arquivos_metricas.items():
        caminho = os.path.join(resultados_path, arquivo)
        f1 = extrair_f1_score(caminho)
        dados.append({"Modelo": modelo, "F1-Score (>50K)": f1})

    df_resultados = pd.DataFrame(dados)
    df_resultados = df_resultados.sort_values(by="F1-Score (>50K)", ascending=False)

    print("\n=== Comparação de Modelos ===")
    print(df_resultados.to_string(index=False))

    # Salvar comparação em CSV
    comparacao_csv = os.path.join(resultados_path, "comparacao_modelos.csv")
    df_resultados.to_csv(comparacao_csv, index=False)
    print(f"\nTabela resumo salva em '{comparacao_csv}'.")

if __name__ == "__main__":
    comparar_modelos()
 