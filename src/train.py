import pandas as pd
import numpy as np
import xgboost as xgb
import mlflow
import mlflow.xgboost
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, mean_absolute_error, r2_score
from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_PATH = BASE_DIR / "data" / "processed" / "train.parquet"
TEST_PATH = BASE_DIR / "data" / "processed" / "test.parquet"
MODELS_DIR = BASE_DIR / "models"

os.environ['MLFLOW_S3_ENDPOINT_URL'] = "http://localhost:9000"
os.environ['AWS_ACCESS_KEY_ID'] = "admin"
os.environ['AWS_SECRET_ACCESS_KEY'] = "password123"

mlflow.set_tracking_uri("http://localhost:5001")
mlflow.set_experiment("Infraspeak_SLA_Prediction")


def train():
    print("🚀 Iniciando Treinamento da Arquitetura Dual-Model...")
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    # Separando as Features dos Targets (Garantindo que removemos as duas colunas alvo do X)
    colunas_alvo = ['estoura_sla', 'time_to_complete_min']
    X_train = train_df.drop(columns=[col for col in colunas_alvo if col in train_df.columns])
    X_test = test_df.drop(columns=[col for col in colunas_alvo if col in test_df.columns])

    # ==========================================
    # MOTOR 1: CLASSIFICADOR DE RISCO (SLA)
    # ==========================================
    y_train_class = train_df['estoura_sla']
    y_test_class = test_df['estoura_sla']

    print("\n" + "=" * 40)
    print("TREINANDO MOTOR 1: Classificador de SLA")
    print("=" * 40)

    with mlflow.start_run(run_name="XGB_SLA_Classifier"):
        params_class = {
            "objective": "binary:logistic",
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "random_state": 42
        }
        mlflow.log_params(params_class)

        model_class = xgb.XGBClassifier(**params_class)
        model_class.fit(X_train, y_train_class)

        # Salva o modelo físico para a API
        model_class.save_model(MODELS_DIR / "modelo_sla.json")

        preds_class = model_class.predict(X_test)
        prob_preds = model_class.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test_class, preds_class)
        auc = roc_auc_score(y_test_class, prob_preds)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("roc_auc", auc)

        print(f"ACURÁCIA: {acc:.2%}")
        print(f"ROC AUC: {auc:.4f}")

        mlflow.xgboost.log_model(xgb_model=model_class, artifact_path="model_sla")

    # ==========================================
    # MOTOR 2: REGRESSOR DE TEMPO (ESTIMATIVA)
    # ==========================================
    print("\n" + "=" * 40)
    print("TREINANDO MOTOR 2: Estimador de Tempo")
    print("=" * 40)

    # Usamos o Log da variável de tempo para lidar com a longa cauda de outliers
    y_train_reg = np.log1p(train_df['time_to_complete_min'])
    y_test_reg = np.log1p(test_df['time_to_complete_min'])

    with mlflow.start_run(run_name="XGB_Tempo_Regressor"):
        params_reg = {
            "objective": "reg:squarederror",
            "n_estimators": 500,
            "max_depth": 8,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "random_state": 42
        }
        mlflow.log_params(params_reg)

        model_reg = xgb.XGBRegressor(**params_reg)
        model_reg.fit(X_train, y_train_reg)

        # Salva o segundo modelo físico para a API
        model_reg.save_model(MODELS_DIR / "modelo_tempo.json")

        preds_log = model_reg.predict(X_test)

        # Revertemos o Log para calcular o Erro Médio Absoluto (MAE) em minutos reais
        preds_reais = np.expm1(preds_log)
        y_test_reais = np.expm1(y_test_reg)

        mae = mean_absolute_error(y_test_reais, preds_reais)
        r2 = r2_score(y_test_reg, preds_log)

        mlflow.log_metric("mae_minutos", mae)
        mlflow.log_metric("r2_score", r2)

        print(f"ERRO MÉDIO ABSOLUTO (MAE): {mae:.2f} minutos")
        print(f"R² SCORE (Capacidade Explicativa): {r2:.4f}")

        mlflow.xgboost.log_model(xgb_model=model_reg, artifact_path="model_tempo")

    print("\nAmbos os modelos foram treinados e salvos com sucesso no MLflow e localmente!")


if __name__ == "__main__":
    train()