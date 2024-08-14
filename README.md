# bce-embedding-TPU
bce-embedding Sophgo platform adaption

## 介绍
此项目是对Huggingface项目[maidalun1020/bce-embedding-base_v1](https://hf-mirror.com/maidalun1020/bce-embedding-base_v1)的移植，在 Sophgo TPU 环境下使用 Sophon-SAIL 接口进行推理加速

## 安装
按照以下步骤，可以将这个项目部署到Sophgo设备上

### 安装第三方库
```bash
cd bce-embedding-TPU
# 考虑到 sail 版本依赖，推荐在 python>=3.8 环境运行
pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 安装sail
此例程`依赖新版本sail`，旧版本需要更新，安装方法请参考[SOPHON-SAIL用户手册](https://doc.sophgo.com/sdk-docs/v24.04.01/docs_latest_release/docs/sophon-sail/docs/zh/html/index.html)

### 模型准备
您可以使用已经下载好的模型（推荐），或自行编译模型。
#### 模型下载
```bash
# 在项目主目录下
./scripts/download.sh
```
模型下载好后，将会在主目录下生成models与token_config
#### 模型编译
模型编译需要使用tpu-mlir,请参考[TPU-MLIR官方文档：开发环境配置](https://doc.sophgo.com/sdk-docs/v24.04.01/docs_latest_release/docs/tpu-mlir/quick_start/html/02_env.html)
```bash
# 在主目录下下载模型
# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install
git clone https://hf-mirror.com/maidalun1020/bce-embedding-base_v1
cd compile

# 导出onnx
python3 export_onnx.py

# 编译模型（此步骤需要在已经部署好的tpu-mlir docker环境中进行）
./compile.sh
```
完成此步骤后，将会在bce-embedding-TPU/compile/bmodel下生成bce-embedding-base_v1.bmodel，默认生成F16 4batch模型，可通过compile脚本中的参数进行修改。

### 例程运行
在准备好模型与token配置后，运行下列命令进行推理
```bash
# 在项目主目录下
python3 sail_inference.py
```
推理结果将打印于命令行。
注：如果需要修改模型的输入，token config的位置，模型的路径，请修改sail_inference.py中的对应部分。