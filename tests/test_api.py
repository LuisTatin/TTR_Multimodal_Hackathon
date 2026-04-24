from fastapi.testclient import TestClient
from api.api import app

def test_health_check():
    """Testa se a API está no ar e se os modelos foram carregados."""
    # Usar o 'with' liga a API de verdade, ativando o evento de startup
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "online"
        assert data["motor_classificador"] == "Carregado"
        assert data["motor_regressor"] == "Carregado"

def test_predict_endpoint_validation():
    """Testa se a API bloqueia dados inválidos (ex: prioridade 5 que não existe)."""
    with TestClient(app) as client:
        payload_invalido = {
            "reportado_por": "User_123",
            "tipo": "Vazamento",
            "area": "Águas/Esgotos",
            "edificio": "Sede Principal",
            "prioridade": 5, # INVÁLIDO (O máximo é 4)
            "hora_abertura": 14,
            "dia_semana": 2
        }
        response = client.post("/predict", json=payload_invalido)
        # Deve retornar Erro 422 (Unprocessable Entity) do Pydantic
        assert response.status_code == 422