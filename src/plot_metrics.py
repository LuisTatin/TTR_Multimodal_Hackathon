import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from pathlib import Path

# Configuração de caminhos
BASE_DIR = Path(__file__).resolve().parent.parent
TEST_PATH = BASE_DIR / "data" / "processed" / "test.parquet"
MODEL_SLA_PATH = BASE_DIR / "models" / "modelo_sla.json"


def gerar_graficos():
    print("Iniciando a geração de gráficos para a apresentação...")

    # 1. Carregar o Motor 1 (Classificador de SLA)
    model_sla = xgb.XGBClassifier()
    model_sla.load_model(MODEL_SLA_PATH)

    # ==========================================
    # GRÁFICO 1: FEATURE IMPORTANCE (XAI)
    # ==========================================
    plt.figure(figsize=(10, 6))
    # O 'gain' mede a melhoria na precisão que cada variável traz para o modelo
    ax = xgb.plot_importance(model_sla, importance_type='gain', max_num_features=10, show_values=False, color='#ff4b4b')
    plt.title("O que mais causa atrasos de SLA? (Impacto na Decisão)", fontsize=14, pad=20)
    plt.ylabel("Variáveis do Chamado")
    plt.xlabel("Grau de Importância (Fator de Ganho)")
    plt.tight_layout()

    caminho_feat = BASE_DIR / "feature_importance.png"
    plt.savefig(caminho_feat, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Gráfico 1 salvo: {caminho_feat}")

    # ==========================================
    # GRÁFICO 2: MATRIZ DE CONFUSÃO
    # ==========================================
    # Carregar dados de teste e separar features do target
    test_df = pd.read_parquet(TEST_PATH)
    colunas_alvo = ['estoura_sla', 'time_to_complete_min']
    X_test = test_df.drop(columns=[col for col in colunas_alvo if col in test_df.columns])
    y_test = test_df['estoura_sla']

    # O modelo faz a prova de fogo (predição) contra dados que nunca viu
    preds = model_sla.predict(X_test)

    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_test, preds,
        display_labels=['No Prazo', 'Vai Atrasar'],
        cmap='Blues',
        ax=ax,
        colorbar=False
    )
    plt.title("Matriz de Confusão: Desempenho Real na Operação", fontsize=14, pad=20)
    plt.tight_layout()

    caminho_matriz = BASE_DIR / "matriz_confusao.png"
    plt.savefig(caminho_matriz, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Gráfico 2 salvo: {caminho_matriz}")


if __name__ == "__main__":
    gerar_graficos()