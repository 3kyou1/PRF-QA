#!/bin/bash

# 设置要使用的CUDA设备
export CUDA_VISIBLE_DEVICES=1  # 使用第一个GPU，可以根据需要修改

# 设置Ollama的监听地址和端口
export OLLAMA_HOST=127.0.0.1:6666  # 可以根据需要修改IP和端口

# 启动Ollama服务
ollama serve