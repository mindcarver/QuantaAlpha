# ACP Backend as a Reusable Library

## 🎯 概述

将 ACP (Agent Client Protocol) 后端集成抽象成一个可复用的 Python 库，使任何使用 LLM 作为后端的 agent 项目都能通过 ACP 调用 OpenCode，实现相同的功能。

## 🏗️ 架构设计

### 目录结构

```
acp-backend/
├── acp_client/              # 核心 ACP 客户端
│   ├── __init__.py
│   ├── client.py            # ACPClient - JSON-RPC 通信
│   ├── backend.py           # ACPBackend - 单例后端
│   └── config.py            # 配置管理
├── acp_patch/               # 运行时补丁
│   ├── __init__.py
│   └── patch.py             # 自动补丁机制
├── embedding/               # Embedding 提供商
│   ├── __init__.py
│   ├── siliconflow.py       # SiliconFlow API
│   ├── jina.py              # Jina AI
│   └── base.py              # 基类接口
├── tests/                   # 测试
├── examples/                # 使用示例
├── pyproject.toml          # 项目配置
└── README.md
```

### 核心组件

#### 1. ACPClient (acp_client/client.py)

```python
class ACPClient:
    """通用的 ACP 客户端，可与任何 ACP 兼容的 agent 通信"""

    def __init__(
        self,
        agent_command: str = "opencode",
        agent_args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ):
        ...

    def start(self) -> None:
        """启动 ACP agent 子进程"""

    def stop(self) -> None:
        """停止 ACP agent"""

    def chat_completion(self, messages, **kwargs) -> str:
        """通过 ACP 请求对话完成"""

    def embedding(self, inputs, **kwargs) -> list[list[float]]:
        """通过 ACP 请求 embedding"""
```

#### 2. ACPBackend (acp_client/backend.py)

```python
class ACPBackend:
    """单例后端，管理 ACP 连接和外部 embedding"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def client(self) -> ACPClient:
        """获取 ACP 客户端实例"""

    def create_embedding(self, texts, **kwargs) -> list[list[float]]:
        """统一的 embedding 接口"""
        # 支持多种 embedding 提供商
        # - ACP agent 的 embedding
        # - 外部 API (SiliconFlow, Jina, etc.)
```

#### 3. 自动补丁 (acp_patch/patch.py)

```python
def patch_llm_backend(
    target_module: str,
    backend_class: str,
    embedding_fn: str = "create_embedding",
    chat_fn: str = "chat",
) -> None:
    """
    自动补丁任何 LLM 后端类

    Args:
        target_module: 目标模块路径 (如 "myapp.llm.client")
        backend_class: 后端类名 (如 "APIBackend")
        embedding_fn: embedding 方法名
        chat_fn: chat 方法名
    """
```

## 📦 使用方式

### 方式 1: 作为库安装

```bash
pip install acp-backend
```

```python
from acp_client import ACPBackend
from acp_patch import patch_llm_backend

# 补丁你的 LLM 后端
patch_llm_backend(
    target_module="myapp.llm.client",
    backend_class="APIBackend"
)

# 配置环境变量
import os
os.environ["ACP_AGENT_COMMAND"] = "opencode"
os.environ["EXTERNAL_EMBEDDING_API"] = "https://api.siliconflow.cn/v1/embeddings"
os.environ["EXTERNAL_EMBEDDING_API_KEY"] = "your-key"

# 使用
from myapp.llm.client import APIBackend
backend = APIBackend()
# 现在会自动使用 ACP + SiliconFlow
```

### 方式 2: 直接复制代码

```bash
cp -r acp_client/ your_project/llm/
cp -r acp_patch/ your_project/llm/
```

## 🔌 支持的 Embedding 提供商

| 提供商 | 环境变量 | 模型示例 |
|--------|----------|----------|
| SiliconFlow | `EXTERNAL_EMBEDDING_API` | Pro/BAAI/bge-m3 |
| Jina AI | `EXTERNAL_EMBEDDING_API` | jina-embeddings-v2 |
| OpenAI | `EXTERNAL_EMBEDDING_API` | text-embedding-3-small |
| Cohere | `EXTERNAL_EMBEDDING_API` | embed-english-v3.0 |

## 🚀 快速开始示例

### 示例 1: 替换 OpenAI 后端

```python
# 原代码
from openai import OpenAI
client = OpenAI(api_key="...")
response = client.chat.completions.create(...)

# 使用 ACP 后端
from acp_patch import patch_llm_backend
import os

os.environ["USE_ACP_BACKEND"] = "true"
os.environ["ACP_AGENT_COMMAND"] = "opencode"

patch_llm_backend("openai", "OpenAI")

# 代码不变，但底层使用 OpenCode
from openai import OpenAI
client = OpenAI()  # 现在通过 ACP 运行
response = client.chat.completions.create(...)
```

### 示例 2: 替换 LangChain 后端

```python
from acp_patch import patch_llm_backend
import os

os.environ["USE_ACP_BACKEND"] = "true"

patch_llm_backend(
    target_module="langchain.llms",
    backend_class="OpenAI"
)

# LangChain 代码不变
from langchain.llms import OpenAI
llm = OpenAI()  # 通过 ACP 运行
```

## 🧩 配置选项

```bash
# ACP Agent 配置
ACP_AGENT_COMMAND=opencode          # ACP agent 命令
ACP_AGENT_ARGS=acp                  # ACP agent 参数

# 外部 Embedding 配置
EXTERNAL_EMBEDDING_API=https://...    # Embedding API 地址
EXTERNAL_EMBEDDING_API_KEY=sk-...     # API Key
EXTERNAL_EMBEDDING_MODEL=model-name   # 模型名称

# 批处理配置
EMBEDDING_BATCH_SIZE=10               # 批处理大小
EMBEDDING_TIMEOUT=30                  # 请求超时(秒)
EMBEDDING_MAX_RETRIES=3               # 最大重试次数
```

## 📊 优势对比

| 特性 | 直接 LLM API | ACP Backend |
|------|-------------|-------------|
| 成本 | 按 token 计费 | OpenCode 本地运行 |
| 隐私 | 数据上传到云端 | 本地处理 |
| 灵活性 | 绑定特定提供商 | 可切换任何 ACP agent |
| Embedding | 需要同一提供商 | 支持任意 embedding API |
| 离线 | 需要网络 | 支持完全离线 |

## 🛠️ 开发计划

- [ ] 独立的 Python 包发布
- [ ] 支持更多 ACP agent (不只是 OpenCode)
- [ ] 支持 Async/Await
- [ ] 内置缓存机制
- [ ] 更完善的错误处理
- [ ] 性能监控和日志
