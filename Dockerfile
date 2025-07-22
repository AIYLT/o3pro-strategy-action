# 使用Python 3.11官方镜像
FROM python:3.11-slim

# 安装必要的系统依赖（解决编译问题）
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    python3-pip \
    python3-venv \
    libffi-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 复制requirements.txt
COPY requirements.txt /app/

# 创建虚拟环境并安装依赖
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# 复制应用代码
COPY . /app/

# 设置环境变量
ENV PYTHONPATH=/app
ENV PATH="/opt/venv/bin:$PATH"

# 暴露端口
EXPOSE $PORT

# 启动命令
CMD uvicorn main:app --host 0.0.0.0 --port $PORT 