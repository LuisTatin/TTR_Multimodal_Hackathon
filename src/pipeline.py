import subprocess
import time
import sys
from pathlib import Path

# Garante que os caminhos funcionem independentemente de onde você rodar o script
BASE_DIR = Path(__file__).resolve().parent.parent


def run_step(script_name, step_title):
    """Executa um script Python como um subprocesso e monitora erros e tempo."""
    print(f"\n{'=' * 60}")
    print(f"INICIANDO ETAPA: {step_title}")
    print(f"{'=' * 60}")

    start_time = time.time()
    script_path = BASE_DIR / "src" / script_name

    try:
        # sys.executable garante que usaremos o Python do ambiente virtual (.venv) atual
        subprocess.run([sys.executable, str(script_path)], check=True)

        elapsed_time = time.time() - start_time
        print(f"\n✅ SUCESSO: {step_title} concluída em {elapsed_time:.2f} segundos.")

    except subprocess.CalledProcessError as e:
        print(f"\nERRO FATAL: Falha ao executar {script_name}.")
        print("O Pipeline foi interrompido para evitar a propagação de dados corrompidos.")
        sys.exit(1)  # Para a execução de todo o pipeline
    except FileNotFoundError:
        print(f"\nERRO: Arquivo {script_path} não encontrado!")
        sys.exit(1)


def run_pipeline():
    print("INICIANDO PIPELINE DE MACHINE LEARNING - INFRASPEAK DUAL-AI 🌟\n")
    total_start_time = time.time()

    # 1. Extração (Conectar ao banco/arquivo e salvar em data/raw)
    run_step("ingestion.py", "1. Ingestão de Dados Brutos (Data Lake)")

    # 2. Transformação (Limpeza, Encoders e Split em data/processed)
    run_step("preprocessing.py", "2. Feature Engineering & Processamento")

    # 3. Treinamento (Classificador SLA + Regressor de Tempo -> Salvar em models/)
    run_step("train.py", "3. Treinamento da Arquitetura Dual-Model")

    # Finalização
    total_elapsed = time.time() - total_start_time
    print(f"\n{'=' * 60}")
    print(f"PIPELINE 100% FINALIZADO COM SUCESSO!")
    print(f"Tempo total da esteira: {total_elapsed:.2f} segundos.")
    print("Modelos atualizados e disponíveis para consumo da FastAPI.")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    run_pipeline()