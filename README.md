
# ⚡ Infraspeak Dual-AI Predictor

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-009688?style=for-the-badge&logo=fastapi)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-FF9900?style=for-the-badge)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker)

> **Motor de Inteligência Artificial para prevenção de quebras de SLA e estimativa de tempo de resolução em operações de facilities e manutenção.**

---

## 🎯 O Desafio & A Solução
Na gestão de facilities, um chamado estourar o SLA significa perda de dinheiro (multas) e quebra de confiança do cliente. Desenvolvemos uma **Arquitetura de Dois Motores (Dual-Model)** que não apenas estima o tempo, mas entende o risco real da operação:

1. **Motor 1 (Classificador XGBoost):** Focado no risco de negócio. Alerta o gestor se o ticket tem alto risco de estourar o SLA limite de 4 horas. *Na dúvida, foi desenhado para ser cauteloso e evitar Falsos Negativos (Multas).*
2. **Motor 2 (Regressor XGBoost):** Focado na UX e planejamento. Estima de forma robusta o tempo esperado de resolução (em minutos) para ajudar na priorização do backlog de manutenções preventivas e corretivas.

---

## 🧠 Arquitetura do Sistema

Nosso sistema adota um padrão de microsserviço desacoplado:

* **ML Pipeline:** Esteira automatizada de extração, tratamento de outliers, encodings e treinamento via sub-processos.
* **XAI (Explainable AI):** Gráficos de *Feature Importance* garantem que a decisão da IA seja auditável (ex: provando que o 'Edifício' e a 'Hora de Abertura' ditam os atrasos).
* **API Gateway:** FastAPI com validação rigorosa via Pydantic, devolvendo o nível de risco e a formatação de tempo já calculados para o Front-end.

### 📂 Estrutura de Diretórios
```text
📦 TTR_Multimodal
 ┣ 📂 api/          # Microsserviço FastAPI e endpoints
 ┣ 📂 data/         # Data Lake simulado (Arquivos ignorados no Git)
 ┣ 📂 frontend/     # Interface de consumo (Streamlit/React)
 ┣ 📂 models/       # Modelos salvos (.json) e Encoders (.joblib)
 ┣ 📂 src/          # Scripts do Pipeline de ML (Ingestão, Treino, XAI)
 ┣ 📂 tests/        # Testes unitários de CI/CD (Pytest)
 ┣ 📜 pipeline.py   # Orquestrador da esteira de dados
 ┗ 📜 requirements.txt
```

---

## 🚀 Como Executar o Projeto Localmente

### 1. Preparação do Ambiente
Crie e ative um ambiente virtual para isolar as dependências:

```bash
python -m venv .venv
source .venv/bin/activate
```
*(No Windows, use: `.venv\Scripts\activate`)*

### 2. Instalação de Dependências

```bash
pip install -r requirements.txt
```

### 3. Subir o Servidor (API)
Inicie o motor da Inteligência Artificial. A API carregará os modelos em memória instantaneamente.

```bash
python -m uvicorn api.api:app --reload --host 0.0.0.0 --port 8000
```
Acesse a documentação interativa em: `http://localhost:8000/docs`

---

## 🛠️ Comandos de Manutenção (MLOps)

### Retreinar toda a Inteligência Artificial
Executa o pipeline de ponta a ponta (Extração > Processamento > Treino).

```bash
python src/pipeline.py
```

### Rodar a Suíte de Testes (CI/CD)
Valida os endpoints e regras do Pydantic garantindo a estabilidade da API.

```bash
python -m pytest tests/test_api.py -v
```

### Gerar Gráficos para o Board (XAI)
Atualiza as imagens de Matriz de Confusão e Importância de Variáveis.

```bash
python src/plot_metrics.py
```

---

## 📡 Documentação de Integração (Frontend)

O endpoint principal de consumo é o `POST /predict`.

**Exemplo de Payload (Request):**

```json
{
  "reportado_por": "User_123",
  "tipo": "Vazamento",
  "area": "Águas/Esgotos",
  "edificio": "Sede Principal",
  "prioridade": 2,
  "hora_abertura": 14,
  "dia_semana": 2
}
```

**Exemplo de Resposta (Response):**

```json
{
  "risco_sla": {
    "estoura_sla": true,
    "probabilidade_atraso": 0.85,
    "nivel_risco": "ALTO"
  },
  "estimativa_tempo": {
    "minutos_estimados": 315,
    "texto_formatado": "5h 15m"
  },
  "mensagem": "Alto risco de atraso! Estimativa: 5h 15m."
}
