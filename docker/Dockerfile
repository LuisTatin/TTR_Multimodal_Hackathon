# 1. Usa uma versão oficial e super leve do Python
FROM python:3.11-slim

# 2. Informações do autor
LABEL authors="luistatin"

# 3. Define a pasta de trabalho lá dentro da "máquina virtual"
WORKDIR /app

# 4. Copia o nosso arquivo de dependências e instala tudo
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copia todo o nosso código para dentro do container
COPY . .

# 6. Avisa que o container vai se comunicar pela porta 8000
EXPOSE 8000

# 7. O comando oficial que liga o Motor da API!
CMD ["python", "-m", "uvicorn", "api.api:app", "--host", "0.0.0.0", "--port", "8000"]