from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from pathlib import Path
import logging

# Configuração de Logs para Produção
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurações de Caminhos
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
ENCODERS_PATH = MODELS_DIR / "encoders_dict.joblib"

# Caminhos dos dois modelos
MODEL_SLA_PATH = MODELS_DIR / "modelo_sla.json"
MODEL_TEMPO_PATH = MODELS_DIR / "modelo_tempo.json"


# ==========================================
# SCHEMAS DE DADOS (Pydantic) - Validação Rigorosa
# ==========================================
class TicketRequest(BaseModel):
    reportado_por: str = Field(..., example="User_123")
    tipo: str = Field(..., example="Vazamento")
    area: str = Field(..., example="Águas/Esgotos")
    edificio: str = Field(..., example="Sede Principal")
    prioridade: int = Field(..., ge=1, le=4, example=2)
    hora_abertura: int = Field(..., ge=0, le=23, example=14)
    dia_semana: int = Field(..., ge=0, le=6, example=2)  # 0=Seg, 6=Dom


# Sub-schemas para organizar a resposta de forma limpa e profissional
class RiscoSLA(BaseModel):
    estoura_sla: bool
    probabilidade_atraso: float
    nivel_risco: str


class EstimativaTempo(BaseModel):
    minutos_estimados: int
    texto_formatado: str


class PredictionResponse(BaseModel):
    risco_sla: RiscoSLA
    estimativa_tempo: EstimativaTempo
    mensagem: str


# ==========================================
# INICIALIZAÇÃO DA API E MODELOS
# ==========================================
app = FastAPI(
    title="Infraspeak Dual-AI Prediction API",
    description="API provendo análise de risco de SLA e estimativa de tempo de resolução.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variáveis globais para os dois motores
modelo_classificador = None
modelo_regressor = None
encoders = None


@app.on_event("startup")
async def carregar_modelos():
    """Carrega os motores e encoders na memória."""
    global modelo_classificador, modelo_regressor, encoders
    try:
        logger.info("Carregando encoders...")
        encoders = joblib.load(ENCODERS_PATH)

        logger.info("Carregando Motor 1 (Classificador de SLA)...")
        modelo_classificador = xgb.XGBClassifier()
        modelo_classificador.load_model(MODEL_SLA_PATH)

        logger.info("Carregando Motor 2 (Regressor de Tempo)...")
        modelo_regressor = xgb.XGBRegressor()
        modelo_regressor.load_model(MODEL_TEMPO_PATH)

        logger.info("✅ Arquitetura Dual-AI pronta para requisições!")
    except Exception as e:
        logger.error(f"Erro ao carregar modelos. Verifique os caminhos: {e}")


# ==========================================
# FUNÇÕES AUXILIARES
# ==========================================
def formatar_tempo(minutos: int) -> str:
    """Converte minutos inteiros para uma string legível para o usuário (ex: 5h 15m)."""
    horas = minutos // 60
    mins = minutos % 60
    if horas > 0:
        return f"{horas}h {mins}m"
    return f"{mins}m"


# ==========================================
# ENDPOINTS
# ==========================================
@app.get("/health", tags=["Monitoramento"])
def health_check():
    status_classificador = "Carregado" if modelo_classificador is not None else "Erro"
    status_regressor = "Carregado" if modelo_regressor is not None else "Erro"
    return {
        "status": "online",
        "motor_classificador": status_classificador,
        "motor_regressor": status_regressor
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Machine Learning"])
def prever_chamado(ticket: TicketRequest):
    if modelo_classificador is None or modelo_regressor is None or encoders is None:
        raise HTTPException(status_code=503, detail="A IA ainda não foi totalmente carregada no servidor.")

    try:
        # 1. Transformar o request
        df_input = pd.DataFrame([{
            'Reportado por': ticket.reportado_por,
            'Tipo': ticket.tipo,
            'Área': ticket.area,
            'Edifício': ticket.edificio,
            'Prioridade.1': ticket.prioridade,
            'hora_abertura': ticket.hora_abertura,
            'dia_semana': ticket.dia_semana,
            'eh_fim_de_semana': 1 if ticket.dia_semana >= 5 else 0
        }])

        # 2. Aplicar Encoders de forma segura
        cat_features = ['Reportado por', 'Tipo', 'Área', 'Edifício']
        for col in cat_features:
            le = encoders.get(col)
            if le:
                valor_str = str(df_input[col].iloc[0])
                if valor_str in le.classes_:
                    df_input[f'{col}_encoded'] = le.transform([valor_str])
                else:
                    logger.warning(f"Valor desconhecido: {valor_str}. Usando default.")
                    df_input[f'{col}_encoded'] = 0
            else:
                raise HTTPException(status_code=500, detail=f"Encoder ausente para {col}")

        features_ordenadas = [
            'Reportado por_encoded', 'Tipo_encoded', 'Área_encoded', 'Edifício_encoded',
            'Prioridade.1', 'hora_abertura', 'dia_semana', 'eh_fim_de_semana'
        ]
        X = df_input[features_ordenadas]

        # 3. Inferência DUPLA
        # Motor 1: Classificador (SLA)
        prob_estourar = float(modelo_classificador.predict_proba(X)[0][1])

        # Motor 2: Regressor (Tempo estimado)
        pred_log = modelo_regressor.predict(X)
        minutos_estimados = int(np.expm1(pred_log[0]))  # Inverte o Log

        # 4. Regras de Negócio
        estoura = prob_estourar > 0.50
        nivel = "ALTO" if prob_estourar > 0.70 else ("MÉDIO" if prob_estourar > 0.40 else "BAIXO")

        texto_tempo = formatar_tempo(minutos_estimados)
        msg = f"Alto risco de atraso! Estimativa: {texto_tempo}." if estoura else f"Dentro do prazo. Estimativa: {texto_tempo}."

        # 5. Montar a resposta estruturada
        return PredictionResponse(
            risco_sla=RiscoSLA(
                estoura_sla=estoura,
                probabilidade_atraso=prob_estourar,
                nivel_risco=nivel
            ),
            estimativa_tempo=EstimativaTempo(
                minutos_estimados=minutos_estimados,
                texto_formatado=texto_tempo
            ),
            mensagem=msg
        )

    except Exception as e:
        logger.error(f"Erro interno na inferência: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro interno ao processar a IA: {str(e)}")