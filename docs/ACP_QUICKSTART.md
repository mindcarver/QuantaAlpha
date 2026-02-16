# ACP 后端集成 - 快速开始指南

## ✅ 测试状态

所有代码逻辑测试已通过（6/6）！

```
✅ ACP Client Module
✅ External Embedding Code
✅ SiliconFlow API Format
✅ OpenCode ACP
✅ File Structure
✅ Config Settings
```

## 📁 已创建的文件

| 文件 | 描述 |
|------|------|
| `quantaalpha/llm/acp_client.py` | ACP 客户端，通过 stdio 与 OpenCode 通信 |
| `quantaalpha/llm/acp_patch.py` | 运行时补丁，拦截 APIBackend 调用 |
| `quantaalpha/llm/config.py` | 已添加 ACP 相关配置项 |
| `tests/test_acp_standalone.py` | 独立测试脚本（无需全部依赖） |
| `docs/ACP_BACKEND_GUIDE.md` | 详细使用指南 |
| `docs/ACP_ARCHITECTURE.md` | 架构设计文档 |
| `run_with_acp.sh` | 快速启动脚本 |

## 🚀 使用步骤

### 1. 设置环境变量

```bash
# 使用 SiliconFlow Embedding API
export EXTERNAL_EMBEDDING_API=https://api.siliconflow.cn/v1/embeddings
export EXTERNAL_EMBEDDING_API_KEY=你的有效API_KEY
export EXTERNAL_EMBEDDING_MODEL=BAAI/bge-large-zh-v1.5
export EMBEDDING_BATCH_SIZE=10
```

### 2. 启动 QuantaAlpha

```bash
cd /Users/mac08/workspace/ai-tools/QuantaAlpha

# 方式 A: 使用启动脚本
./run_with_acp.sh

# 方式 B: 手动启用
python -c "
import os
os.environ['EXTERNAL_EMBEDDING_API'] = 'https://api.siliconflow.cn/v1/embeddings'
os.environ['EXTERNAL_EMBEDDING_API_KEY'] = '你的API_KEY'
os.environ['EXTERNAL_EMBEDDING_MODEL'] = 'BAAI/bge-large-zh-v1.5'

from quantaalpha.llm.acp_patch import patch_apibackend
patch_apibackend()

# 现在可以正常使用 QuantaAlpha
from quantaalpha.llm.client import APIBackend
backend = APIBackend()
"
```

### 3. Chat Completion 通过 OpenCode (可选)

```bash
# 启用 ACP 后端用于 Chat Completion
export USE_ACP_BACKEND=true
export ACP_AGENT_COMMAND=opencode
export ACP_AGENT_ARGS=acp

# OpenCode 会自动启动作为子进程
```

## 🔧 可用的 Embedding 模型

### SiliconFlow (推荐)

| 模型 | 描述 | 最大 Token |
|------|------|-----------|
| `BAAI/bge-large-zh-v1.5` | 中文嵌入 | 512 |
| `BAAI/bge-large-en-v1.5` | 英文嵌入 | 512 |
| `Qwen/Qwen3-Embedding-8B` | Qwen3嵌入 | 32768 |
| `Qwen/Qwen3-Embedding-4B` | Qwen3嵌入 | 32768 |

### 其他选择

```bash
# 智谱 GLM
export EXTERNAL_EMBEDDING_API=https://open.bigmodel.cn/api/paas/v4/embeddings
export EXTERNAL_EMBEDDING_MODEL=embedding-v2

# Ollama (本地)
export EXTERNAL_EMBEDDING_API=http://localhost:11434/api/embeddings
export EXTERNAL_EMBEDDING_MODEL=nomic-embed-text
```

## 📊 代码示例

```python
from quantaalpha.llm.acp_patch import patch_apibackend
from quantaalpha.llm.client import APIBackend

# 设置环境变量
import os
os.environ["EXTERNAL_EMBEDDING_API"] = "https://api.siliconflow.cn/v1/embeddings"
os.environ["EXTERNAL_EMBEDDING_API_KEY"] = "你的API_KEY"
os.environ["EXTERNAL_EMBEDDING_MODEL"] = "BAAI/bge-large-zh-v1.5"

# 应用补丁
patch_apibackend()

# 创建后端实例
backend = APIBackend()

# 使用 Embedding（会自动使用 SiliconFlow）
embeddings = backend.create_embedding(["动量因子", "波动率因子"])
print(f"Got {len(embeddings)} embeddings, each with {len(embeddings[0])} dimensions")

# 使用 Chat Completion（如果有 OpenAI API key 配置）
response = backend.build_messages_and_create_chat_completion(
    user_prompt="生成一个基于动量的量化因子",
    system_prompt="你是一个量化研究员"
)
print(response)
```

## ⚠️ 关于 API Key

测试时使用的 API key 返回 401 错误。你需要：

1. 访问 https://siliconflow.cn/ 注册账号
2. 在控制台创建 API Key（格式：sk-xxxxx）
3. 将有效的 key 设置到环境变量

## 🎯 架构流程

```
用户代码
    ↓
APIBackend.build_messages_and_create_chat_completion()
    ↓
[acp_patch 拦截]
    ↓
ACPBackend.chat_completion()
    ↓
OpenCode (via ACP/stdio)
    ↓
LLM 模型响应

Embedding 流程:
APIBackend.create_embedding()
    ↓
[acp_patch 拦截]
    ↓
ACPBackend._external_embedding()
    ↓
SiliconFlow API (HTTP)
    ↓
返回向量
```

## ✨ 下一步

1. 获取有效的 SiliconFlow API key
2. 安装全部依赖：`pip install -r requirements.txt`
3. 运行完整测试：`python tests/test_acp_integration.py`
