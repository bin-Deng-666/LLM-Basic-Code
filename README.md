# LLM-Code-Learning

一个用于学习大语言模型（LLM）核心算法和组件的代码库，包含从基础到进阶的各种实现。

## 📚 项目概述

本项目旨在通过代码实现来深入理解大语言模型的核心组件和算法。每个模块都包含独立的实现，方便学习和调试。

## 🏗️ 项目结构

```
LLM-Basic-Code/
├── HF_Download/         # HuggingFace模型下载
│   └── download_model.sh
├── LLM_Fundamentals/    # LLM基础组件
│   ├── Activation/      # 激活函数
│   │   ├── Softmax.py   # Softmax激活函数
│   │   └── SwiGLU.py    # SwiGLU激活函数
│   ├── Attention/       # 注意力机制
│   │   ├── GQA.py       # Grouped Query Attention
│   │   ├── MHA.py       # Multi-Head Attention
│   │   ├── MLA.py       # Multi-Head Latent Attention
│   │   ├── MQA.py       # Multi-Query Attention
│   │   └── SA.py        # Self-Attention
│   ├── LoRA/            # LoRA微调
│   │   └── LoRA.py
│   ├── Loss/            # 损失函数
│   │   ├── CrossEntropy.py # 交叉熵损失
│   │   ├── DPO.py       # DPO
│   │   └── KL.py        # KL散度
│   ├── Normalization/   # 归一化层
│   │   ├── LayerNorm.py # 层归一化
│   │   └── RMSNorm.py   # RMS归一化
│   ├── RNN/             # 循环神经网络
│   │   └── rnn.py
│   ├── RoPE/            # 旋转位置编码
│   │   └── RoPE.py
│   └── Transformer/     # Transformer模型
│       └── Transformer.py
├── RAG/                 # 检索增强生成
│   ├── chat_with_db.py
│   ├── create_db.py
│   └── data/
│       └── alice_in_wonderland.md
├── README.md
├── ReAct-Agent/         # ReAct智能体
│   ├── agent.py
│   ├── llm.py
│   └── tool.py
└── test.py             # 测试文件
```