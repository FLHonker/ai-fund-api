# 使用官方Python镜像作为基础镜像
FROM python:3.10-slim as builder

# 设置工作目录
WORKDIR /app

# 复制依赖文件
COPY requirements.txt .

# 安装依赖
RUN pip install --user -r requirements.txt

# 第二阶段：创建运行时镜像
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 从builder阶段复制已安装的Python包
COPY --from=builder /root/.local /root/.local
COPY . .

# 确保脚本可执行
RUN chmod +x src/api.py

# 设置环境变量
ENV PATH=/root/.local/bin:$PATH

# 暴露端口
EXPOSE 8000

# 启动应用
CMD ["python", "src/api.py"]
