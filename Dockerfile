# Dockerfile

# --- 基础镜像 ---
# 使用一个官方的、包含 Python 3.9 的轻量级 Debian 镜像作为基础
FROM python:3.9-slim

# --- 设置工作目录 ---
# 在容器内部创建一个 /app 目录，并将其设置为后续命令的执行目录
WORKDIR /app

# --- 安装系统依赖 ---
# (如果未来需要安装如 TA-Lib 等需要C编译环境的库，可以在这里添加)
# RUN apt-get update && apt-get install -y build-essential

# --- 复制并安装 Python 依赖 ---
# 首先只复制 requirements.txt 文件，利用Docker的层缓存机制，
# 只要这个文件不变，就不需要重新安装所有依赖，加快后续构建速度。
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# --- 复制项目代码 ---
# 将项目根目录下的所有文件复制到容器的 /app 目录中
COPY . .

# --- 暴露端口 ---
# 声明容器将监听 8501 端口，这是 Streamlit 的默认端口
EXPOSE 8501

# --- 容器启动命令 ---
# 当容器启动时，执行此命令来运行 Streamlit 应用
# --server.port 8501: 明确指定端口
# --server.address 0.0.0.0: 允许从容器外部访问
CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
