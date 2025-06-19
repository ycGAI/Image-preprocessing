#!/bin/bash

# 图像清晰度分类项目 Conda 环境安装脚本

# 设置环境名称
ENV_NAME="image_classifier"

# 推荐的Python版本
PYTHON_VERSION="3.9"

echo "================================="
echo "图像清晰度分类项目环境安装"
echo "================================="

# 检查conda是否已安装
if ! command -v conda &> /dev/null; then
    echo "❌ 错误: conda 未安装或未在PATH中"
    echo "请先安装 Anaconda 或 Miniconda"
    exit 1
fi

echo "✅ 检测到 conda 已安装"

# 检查环境是否已存在
if conda env list | grep -q "^${ENV_NAME}"; then
    echo "⚠️  环境 '${ENV_NAME}' 已存在"
    read -p "是否删除现有环境并重新创建? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🗑️  删除现有环境..."
        conda env remove -n ${ENV_NAME} -y
    else
        echo "❌ 取消安装"
        exit 1
    fi
fi

# 创建新环境
echo "🚀 创建新的conda环境: ${ENV_NAME} (Python ${PYTHON_VERSION})"
conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y

if [ $? -ne 0 ]; then
    echo "❌ 创建环境失败"
    exit 1
fi

echo "✅ 环境创建成功"

# 激活环境
echo "🔄 激活环境..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${ENV_NAME}

if [ $? -ne 0 ]; then
    echo "❌ 激活环境失败"
    exit 1
fi

echo "✅ 环境激活成功"

# 安装核心依赖（通过conda，不包括opencv）
echo "📦 安装核心依赖包（通过conda）..."

conda install -c conda-forge -y \
    numpy \
    scipy \
    tqdm \
    matplotlib \
    seaborn \
    pandas \
    jupyter \
    notebook \
    ipykernel

if [ $? -ne 0 ]; then
    echo "❌ conda依赖包安装失败"
    exit 1
fi

echo "✅ conda依赖安装完成"

# 通过pip安装OpenCV（避免conda/pip冲突）
echo "📦 安装OpenCV（通过pip）..."
pip install opencv-python

if [ $? -ne 0 ]; then
    echo "❌ OpenCV安装失败"
    exit 1
fi

echo "✅ OpenCV安装完成"

# 安装开发依赖（可选）
read -p "是否安装开发依赖 (pytest, black, flake8)? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "📦 安装开发依赖..."
    conda install -c conda-forge -y pytest pytest-cov black flake8
    echo "✅ 开发依赖安装完成"
fi

# 验证安装
echo "🔍 验证安装..."
python -c "
import cv2
import numpy as np
import scipy
import tqdm
import matplotlib
import seaborn
import pandas
print('✅ 所有依赖包导入成功')
print(f'Python版本: {cv2.__version__}')
print(f'OpenCV版本: {cv2.__version__}')
print(f'NumPy版本: {np.__version__}')
print(f'SciPy版本: {scipy.__version__}')
print(f'Matplotlib版本: {matplotlib.__version__}')
"

if [ $? -eq 0 ]; then
    echo "🎉 环境安装完成！"
    echo ""
    echo "使用方法："
    echo "  conda activate ${ENV_NAME}"
    echo "  python main.py --help"
    echo ""
    echo "环境信息："
    echo "  环境名称: ${ENV_NAME}"
    echo "  Python版本: ${PYTHON_VERSION}"
    echo "  安装路径: $(conda info --envs | grep ${ENV_NAME} | awk '{print $2}')"
    echo ""
    echo "⚠️  注意: OpenCV通过pip安装，避免conda/pip包管理冲突"
else
    echo "❌ 环境验证失败"
    exit 1
fi