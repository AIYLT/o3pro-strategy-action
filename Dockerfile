# 使用官方 Python Slim 映像作為基底
FROM python:3.11-slim

# 設定工作目錄
WORKDIR /app

# 安裝必要的系統依賴（支援 pip 套件編譯）
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    libffi-dev \
    gcc \
    libssl-dev \
    curl && \
    rm -rf /var/lib/apt/lists/*

# 建立虛擬環境
RUN python3 -m venv /opt/venv

# 將虛擬環境加入 PATH（取代 . activate）
ENV PATH="/opt/venv/bin:$PATH"

# 複製 requirements.txt
COPY requirements.txt /app/requirements.txt

# 升級 pip 並安裝套件
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# 複製其餘程式碼（確保 Docker cache 生效）
COPY . /app

# Railway动态端口启动命令
CMD uvicorn main:app --host 0.0.0.0 --port $PORT 