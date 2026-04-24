import streamlit as st
import pandas as pd
import time
from pathlib import Path
import requests

# ==========================================
# CONFIGURAÇÕES GERAIS
# ==========================================
st.set_page_config(page_title="Infraspeak AI - Operações", layout="wide", initial_sidebar_state="expanded")

BASE_DIR = Path(__file__).resolve().parent.parent
# Ajuste este caminho se o seu parquet estiver noutra pasta
DATA_PATH = BASE_DIR / "data" / "raw" / "tickets_data.parquet"


# ==========================================
# 1. CARREGAMENTO DE DADOS REAIS
# ==========================================
@st.cache_data
def get_real_options():
    """Carrega os valores únicos do parquet para os seletores do formulário"""
    if not DATA_PATH.exists():
        st.error(f"Erro: Arquivo Parquet não encontrado em {DATA_PATH}")
        return pd.DataFrame(), ["Sem dados"], ["Sem dados"]

    df = pd.read_parquet(DATA_PATH)
    areas = sorted(df['Área'].dropna().unique().tolist())
    edificios = sorted(df['Edifício'].dropna().unique().tolist())
    return df, areas, edificios


df_raw, lista_areas, lista_edificios = get_real_options()

# ==========================================
# 2. MEMÓRIA DA APLICAÇÃO (Session State)
# ==========================================
if 'fila_pedidos' not in st.session_state:
    if not df_raw.empty:
        # Pega os últimos 10 tickets reais para começar a tabela
        df_inicial = df_raw[['Data de reporte', 'Área', 'Edifício', 'Prioridade.1']].dropna().tail(10).copy()
        # Mock do resultado da IA para os dados históricos
        df_inicial['Predição IA'] = df_inicial['Prioridade.1'].apply(lambda x: "⚠️ RISCO" if x <= 2 else "✅ OK")
        # Inverte a ordem para os mais recentes ficarem no topo
        st.session_state.fila_pedidos = df_inicial.iloc[::-1].reset_index(drop=True)
    else:
        st.session_state.fila_pedidos = pd.DataFrame(
            columns=['Data de reporte', 'Área', 'Edifício', 'Prioridade.1', 'Predição IA'])

# ==========================================
# 3. MENU LATERAL (SIDEBAR)
# ==========================================
with st.sidebar:
    st.title("INFRASPEAK")
    st.write("---")
    st.button("Pedidos", type="primary", use_container_width=True)
    st.write("---")
    st.caption("AI SLA Predictor v2.0")

# ==========================================
# 4. INTERFACE PRINCIPAL E ABAS
# ==========================================
st.header("Pedidos")
abas = st.tabs(["Aguardando aprovação", "Não iniciado", "Em curso", "Pausados", "Fechados"])

with abas[1]:  # Focando na aba "Não Iniciado"
    col_tabela, col_form = st.columns([2, 1])

    # ------------------------------------------
    # PAINEL DIREITO: FORMULÁRIO DE CRIAÇÃO
    # ------------------------------------------
    with col_form:
        st.subheader("➕ Criar novo pedido")
        st.write("Preencha os dados para análise de risco da IA.")

        with st.container(border=True):
            # Seletores com dados reais
            area_selecionada = st.selectbox("* Área", lista_areas)

            # Lógica Dinâmica: O Tipo muda dependendo da Área
            if not df_raw.empty:
                tipos_disponiveis = sorted(
                    df_raw[df_raw['Área'] == area_selecionada]['Tipo'].dropna().unique().tolist())
            else:
                tipos_disponiveis = ["---"]

            tipo_selecionado = st.selectbox("* Tipo de Falha", tipos_disponiveis)
            edificio_selecionado = st.selectbox("* Edifício", lista_edificios)

            st.write("* Prioridade")
            # Ajuste os valores da prioridade conforme o seu dataset real (ex: 1, 2, 3, 4 ou Texto)
            prioridade = st.radio("Selecione:", [1, 2, 3, 4], horizontal=True, label_visibility="collapsed")

            reportado_por = st.text_input("ID do Solicitante", "User_123")

            # BOTÃO DE AÇÃO
            submit_button = st.button("Analisar Risco & Criar", type="primary", use_container_width=True)

            if submit_button:
                with st.spinner("Inteligência Artificial analisando histórico..."):
                    # Aqui ficará a chamada requests.post() para a sua API!

                    payload = {
                        "reportado_por": str(reportado_por),
                        "tipo": str(tipo_selecionado),
                        "area": str(area_selecionada),
                        "edificio": str(edificio_selecionado),
                        "prioridade": int(prioridade),
                        "hora_abertura": int(pd.Timestamp.now().hour),
                        "dia_semana": int(pd.Timestamp.now().dayofweek)
                    }

                    # Chamada Real para a API
                    resposta = requests.post("http://localhost:8000/predict", json=payload)

                    if resposta.status_code == 200:
                        dados_ia = resposta.json()
                        risco_alto = dados_ia["estoura_sla"]
                        probabilidade = dados_ia["probabilidade_atraso"]
                        status_ia = "⚠️ RISCO" if risco_alto else "✅ OK"
                    else:
                        st.error("Erro na API de IA!")
                        st.write(resposta.text)
                        st.stop()

                    # INSERIR NA MEMÓRIA DA TABELA EM TEMPO REAL
                    novo_registro = pd.DataFrame([{
                        'Data de reporte': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
                        'Área': area_selecionada,
                        'Edifício': edificio_selecionado,
                        'Prioridade.1': prioridade,
                        'Predição IA': status_ia
                    }])

                    # Atualiza a fila colocando o novo no topo e mantendo apenas os 10 últimos
                    st.session_state.fila_pedidos = pd.concat([novo_registro, st.session_state.fila_pedidos],
                                                              ignore_index=True).head(10)

    # ------------------------------------------
    # PAINEL ESQUERDO: TABELA AO VIVO
    # ------------------------------------------
    with col_tabela:
        st.subheader("Fila de Atendimento")


        # Função para pintar os resultados da IA na tabela
        def color_status(val):
            if val == "⚠️ RISCO":
                return 'background-color: #ffcccc; color: #cc0000; font-weight: bold'
            elif val == "✅ OK":
                return 'background-color: #e6f9ec; color: #009933; font-weight: bold'
            return ''


        # Exibe a tabela lendo do session_state (memória viva)
        st.dataframe(
            st.session_state.fila_pedidos.style.applymap(color_status, subset=['Predição IA']),
            use_container_width=True,
            hide_index=True,
            height=600
        )