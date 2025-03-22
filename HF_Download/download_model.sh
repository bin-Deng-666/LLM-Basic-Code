#!/bin/bash

# 设置镜像端点
export HF_ENDPOINT="https://hf-mirror.com"

# 获取参数
model_name=${1:-"gpt2"}  # 第一个参数，默认为gpt2
base_dir=${2:-"./"}  # 第二个参数，默认为当前目录下的models文件夹

# 创建下载目录（使用相对路径）
download_dir="$base_dir/$model_name"
mkdir -p "$download_dir"

# 下载模型
echo "开始下载模型: $model_name 到目录: $download_dir"
huggingface-cli download --resume-download "$model_name" --local-dir "$download_dir"

# 检查是否下载成功
if [ $? -eq 0 ]; then
    echo -e "\n模型下载成功！"
    echo "模型保存位置: $download_dir"
else
    echo -e "\n模型下载失败，请检查错误信息" >&2
    exit 1
fi