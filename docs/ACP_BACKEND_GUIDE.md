# ACP Backend Integration Guide for QuantaAlpha

## 概述

本指南说明如何使用 ACP (Agent Client Protocol) 后端替代 QuantaAlpha 中的传统 LLM API 调用。

## 架构

```
QuantaAlpha (Python)
    ↓ JSON-RPC over stdio
OpenCode (ACP Agent)
    ↓ (内部使用)
各种 LLM 模型 (Claude, GPT, DeepSeek, etc.)

Embedding:
    ↓ HTTP API
硅基流动 / 智谱 GLM / 其他兼容 API
```

## 环境配置

### 1. 安装 OpenCode

```bash
# 使用 npm 安装
npm install -g @opencode-ai/opencode

# 或使用 cargo 安装
cargo install opencode
```

### 2. 配置环境变量

创建 `.env` 文件或设置环境变量：

```bash
# 启用 ACP 后端
USE_ACP_BACKEND=true

# OpenCode 配置 (可选，默认使用 "opencode acp")
ACP_AGENT_COMMAND=opencode
ACP_AGENT_ARGS=acp

# 外部 Embedding API 配置
# 硅基流动示例
EXTERNAL_EMBEDDING_API=https://api.siliconflow.cn/v1/embeddings
EXTERNAL_EMBEDDING_API_KEY=your_siliconflow_api_key
EXTERNAL_EMBEDDING_MODEL=BAAI/bge-large-zh-v1.5

# 智谱 GLM 示例
# EXTERNAL_EMBEDDING_API=https://open.bigmodel.cn/api/paas/v4/embeddings
# EXTERNAL_EMBEDDING_API_KEY=your_glm_api_key
# EXTERNAL_EMBEDDING_MODEL=embedding-v2

# Embedding 批处理大小
EMBEDDING_BATCH_SIZE=10
```

### 3. 在 QuantaAlpha 中启用 ACP

有两种方式：

#### 方式 A: 通过环境变量（推荐）

```python
import os
os.environ["USE_ACP_BACKEND"] = "true"

from quantaalpha.llm.acp_patch import patch_apibackend
patch_apibackend()

# 正常使用 APIBackend
from quantaalpha.llm.client import APIBackend

backend = APIBackend()
response = backend.build_messages_and_create_chat_completion(
    user_prompt="生成一个量化因子...",
    system_prompt="你是一个量化研究员..."
)
```

#### 方式 B: 通过配置文件

在 `experiment.yaml` 或配置中设置：

```yaml
llm:
  use_acp_backend: true
  acp_agent_command: "opencode"
  acp_agent_args: "acp"
  external_embedding_api: "https://api.siliconflow.cn/v1/embeddings"
  external_embedding_api_key: "${EXTERNAL_EMBEDDING_API_KEY}"
  external_embedding_model: "BAAI/bge-large-zh-v1.5"
```

## 支持的 Embedding API

### 硅基流动 (SiliconFlow)

```bash
EXTERNAL_EMBEDDING_API=https://api.siliconflow.cn/v1/embeddings
EXTERNAL_EMBEDDING_API_KEY=sk-xxx
EXTERNAL_EMBEDDING_MODEL=BAAI/bge-large-zh-v1.5
```

### 智谱 GLM

```bash
EXTERNAL_EMBEDDING_API=https://open.bigmodel.cn/api/paas/v4/embeddings
EXTERNAL_EMBEDDING_API_KEY=your_glm_api_key
EXTERNAL_EMBEDDING_MODEL=embedding-v2
```

### Ollama (本地)

```bash
EXTERNAL_EMBEDDING_API=http://localhost:11434/api/embeddings
EXTERNAL_EMBEDDING_MODEL=nomic-embed-text
```

## 工作原理

1. **Chat Completion**: QuantaAlpha 通过 JSON-RPC 发送消息给 OpenCode
2. **OpenCode 处理**: OpenCode 使用其配置的 LLM 模型生成响应
3. **Embedding**: 通过外部 API (硅基流动/GLM) 获取向量表示
4. **缓存**: SQLite 缓存仍然有效，减少重复调用

## 优势

| 特性 | 传统 API | ACP 后端 |
|------|----------|----------|
| 模型切换 | 修改代码配置 | OpenCode 内部切换 |
| 多模型支持 | 需要多个 API key | OpenCode 统一管理 |
| 成本 | 按 API 计费 | 可用本地模型 |
| 隐私 | 数据发送到 API | 可用本地模型 |
| Embedding | 需要单独服务 | 外部 API 灵活配置 |

## 故障排查

### OpenCode 未找到

```
Error: Failed to initialize ACP connection
```

**解决方案**:
```bash
# 检查 OpenCode 是否安装
which opencode

# 或指定完整路径
ACP_AGENT_COMMAND=/usr/local/bin/opencode
```

### Embedding API 超时

```
Error: EXTERNAL_EMBEDDING_API request timeout
```

**解决方案**:
- 减小 `EMBEDDING_BATCH_SIZE`
- 检查网络连接
- 验证 API key 是否正确

### JSON 格式错误

```
Error: Failed to parse ACP message
```

**解决方案**:
- 确保 OpenCode 版本 >= 1.0.0
- 检查 OpenCode 日志: `opencode acp --debug`

## 示例代码

### 完整示例

```python
import os
from quantaalpha.llm.acp_patch import patch_apibackend
from quantaalpha.llm.client import APIBackend, ChatSession

# 启用 ACP
os.environ["USE_ACP_BACKEND"] = "true"
patch_apibackend()

# 创建后端
backend = APIBackend()

# Chat Completion
messages = [
    {"role": "system", "content": "你是量化因子专家"},
    {"role": "user", "content": "基于动量效应设计一个因子"}
]
response = backend._try_create_chat_completion_or_embedding(
    messages=messages,
    chat_completion=True,
    temperature=0.7,
    max_tokens=2000
)
print(response)

# 使用 ChatSession
session = backend.build_chat_session(
    system_prompt="你是量化因子专家"
)
response = session.build_chat_completion(
    "基于动量效应设计一个因子"
)
print(response)

# Embedding
embeddings = backend.create_embedding([
    "动量因子",
    "波动率因子"
])
print(f"Embedding shape: {len(embeddings[0])}")
```

## 性能优化

1. **启用缓存**: 在 `experiment.yaml` 中设置 `use_chat_cache: true`
2. **批处理 Embedding**: 调整 `EMBEDDING_BATCH_SIZE`
3. **本地模型**: OpenCode 可配置使用本地 Ollama 模型

## 安全建议

1. **不要提交 API key**: 使用 `.env` 文件并添加到 `.gitignore`
2. **使用环境变量**: 敏感信息通过环境变量传递
3. **限制权限**: API key 只授予必要权限
