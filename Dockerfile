# 使用官方 Python 镜像作为基础镜像
FROM python:3.9-slim-buster

# 设置工作目录
WORKDIR /app

# 复制 requirements.txt 并安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制所有应用代码到容器中
COPY . .

# 暴露 Streamlit 应用的端口
EXPOSE 8501

# 定义容器启动时运行的命令
# 这里假设您的主应用入口是 app.py，并且您会通过 streamlit run 来启动它
# 如果您的启动命令不同，请根据实际情况修改
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]