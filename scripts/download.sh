#!/bin/bash
set -ex

res=$(which unzip)

if [ $? != 0 ];
then
    echo "Please install unzip on your system!"
    exit
fi

pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade

# download models
if [ ! -d "./models" ]; then
    echo "./models does not exist, download..."
    python3 -m dfss --url=open@sophgo.com:ext_model_information/RAG/models.zip
    unzip models.zip
    rm models.zip
    echo "models download!"
else
    echo "models already exist..."
fi

# download token_config
if [ ! -d "./token_config" ]; then
    echo "./token_config does not exist, download..."
    python3 -m dfss --url=open@sophgo.com:ext_model_information/RAG/token_config.zip
    unzip token_config.zip
    rm token_config.zip
    echo "token_config download!"
else
    echo "token_config already exist..."
fi