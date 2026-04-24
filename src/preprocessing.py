import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import joblib
import os

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = BASE_DIR / "data" / "raw" / "tickets_data.parquet"
OUTPUT_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"


def preprocess():
    print("🚀 Iniciando pré-processamento para ARQUITETURA DUAL (Classificação + Regressão)...")
    df = pd.read_parquet(INPUT_PATH)

    # 1. TRANSFORMAÇÃO DE TEMPO E FILTRO DE ZUMBIS (Protege o Regressor)
    df['time_to_complete_min'] = df['Tempo para fechar (segundos)'] / 60

    # Removemos nulos e tickets com tempos absurdos (erros de sistema)
    # Mantemos apenas chamados que duraram entre 1 minuto e 7 dias
    df = df.dropna(subset=['time_to_complete_min'])
    df = df[(df['time_to_complete_min'] >= 1) & (df['time_to_complete_min'] <= 10080)]

    # 2. CRIAR O TARGET DE NEGÓCIO (SLA para o Classificador)
    LIMITE_SLA_MINUTOS = 240
    df['estoura_sla'] = (df['time_to_complete_min'] > LIMITE_SLA_MINUTOS).astype(int)

    print(f"📊 Distribuição do SLA após limpeza: \n{df['estoura_sla'].value_counts(normalize=True) * 100}")

    # 3. EXTRAÇÃO TEMPORAL
    df['Data de reporte'] = pd.to_datetime(df['Data de reporte'])
    df['hora_abertura'] = df['Data de reporte'].dt.hour
    df['dia_semana'] = df['Data de reporte'].dt.dayofweek
    df['eh_fim_de_semana'] = (df['dia_semana'] >= 5).astype(int)

    # 4. FEATURE ENGINEERING DO MINUTO ZERO
    encoders = {}
    cat_features = ['Reportado por', 'Tipo', 'Área', 'Edifício']

    for col in cat_features:
        if col in df.columns:
            le = LabelEncoder()
            df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(encoders, MODELS_DIR / "encoders_dict.joblib")

    # 5. DEFINIÇÃO FINAL (Features + 2 Targets)
    features = [f'{col}_encoded' for col in cat_features] + \
               ['Prioridade.1', 'hora_abertura', 'dia_semana', 'eh_fim_de_semana']

    targets = ['estoura_sla', 'time_to_complete_min']  # Agora passamos as duas colunas!

    X = df[features]
    Y = df[targets]

    # 6. SPLIT E SALVAR
    # Continuamos estratificando pelo 'estoura_sla' para manter o classificador equilibrado
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y['estoura_sla'])

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Juntando tudo para o parquet
    pd.concat([X_train, Y_train], axis=1).to_parquet(OUTPUT_DIR / "train.parquet", index=False)
    pd.concat([X_test, Y_test], axis=1).to_parquet(OUTPUT_DIR / "test.parquet", index=False)

    print(f"Preprocessamento concluído! Targets configurados: {targets}")


if __name__ == "__main__":
    preprocess()