FROM continuumio/anaconda3:latest

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 复制项目文件到容器
COPY . /app/

# 安装PyTorch与CUDA支持(取消注释下一行可以指定CUDA 12.1版本)
# RUN pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
RUN pip install torch==2.5.1

# 安装项目依赖
RUN pip install transformers rouge_score numpy datasets tqdm modelscope accelerate

# 设置环境变量
ENV PYTHONPATH=/app:$PYTHONPATH

# 容器启动时默认执行的命令
CMD ["/bin/bash"]

# 使用说明:
# 1. 构建镜像: docker build -t bert-summarization .
# 2. 运行容器: 
#    docker run --name bert --gpus all --network=host --ipc=host --privileged \
#    --restart=always -v /data:/data -v $(pwd):/app -it bert-summarization
#
# 注意:
# - 包含了GPU支持，需要宿主机安装NVIDIA Container Toolkit
# - 挂载了/data目录和当前目录到容器中
# - 使用了host网络模式
# - 使用--privileged模式赋予容器扩展权限