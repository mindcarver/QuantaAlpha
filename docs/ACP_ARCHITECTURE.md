# ACP Backend Architecture for QuantaAlpha

## 系统架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              QuantaAlpha                                     │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                         LLM Integration Layer                           ││
│  │  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ││
│  │  │  HypothesisGen  │    │  FactorMining   │    │  EvolutionOps   │    ││
│  │  └────────┬────────┘    └────────┬────────┘    └────────┬────────┘    ││
│  │           │                      │                      │              ││
│  │           └──────────────────────┼──────────────────────┘              ││
│  │                                  │                                     ││
│  │  ┌───────────────────────────────▼─────────────────────────────────┐  ││
│  │  │                    APIBackend (Modified)                         │  ││
│  │  │  ┌─────────────────────────────────────────────────────────┐    │  ││
│  │  │  │  ACP Patch Layer                                         │    │  ││
│  │  │  │  - Intercepts chat completion calls                     │    │  ││
│  │  │  │  - Intercepts embedding calls                           │    │  ││
│  │  │  │  - Routes to ACP backend when enabled                   │    │  ││
│  │  │  └────────────────────┬────────────────────────────────────┘    │  ││
│  │  └───────────────────────┼──────────────────────────────────────┘  ││
│  └──────────────────────────┼──────────────────────────────────────────┘│
└─────────────────────────────┼─────────────────────────────────────────────┘
                              │
                    ┌─────────▼──────────┐
                    │  ACP Protocol Layer│
                    │  (JSON-RPC stdio)  │
                    └─────────┬──────────┘
                              │
        ┌─────────────────────┴─────────────────────┐
        │                                           │
┌───────▼────────┐                          ┌──────▼─────────┐
│   OpenCode     │                          │ External APIs  │
│  (ACP Agent)   │                          │                │
│  ┌───────────┐ │                          │ ┌─────────────┐│
│  │ LLM Models│ │                          │ │硅基流动      ││
│  │ - Claude  │ │                          │ │智谱 GLM     ││
│  │ - GPT     │ │                          │ │Ollama       ││
│  │ - DeepSeek│ │                          │ └─────────────┘│
│  │ - Local   │ │                          │                │
│  └───────────┘ │                          │                │
└────────────────┘                          └────────────────┘
```

## 数据流

### Chat Completion 流程

```
1. QuantaAlpha: FactorHypothesisGen.generate()
   ↓
2. APIBackend.build_messages_and_create_chat_completion()
   ↓
3. ACP Patch: Intercept if USE_ACP_BACKEND=true
   ↓
4. ACPClient.chat_completion()
   │
   ├─→ Start OpenCode subprocess (if not running)
   ├─→ Send JSON-RPC request via stdin
   │   {
   │     "jsonrpc": "2.0",
   │     "id": 1,
   │     "method": "chat/completions",
   │     "params": {
   │       "messages": [...],
   │       "temperature": 0.7,
   │       "max_tokens": 2000
   │     }
   │   }
   ↓
5. OpenCode: Process with LLM
   ↓
6. OpenCode: Send JSON-RPC response via stdout
   │   {
   │     "jsonrpc": "2.0",
   │     "id": 1,
   │     "result": {
   │       "content": "生成的因子表达式..."
   │     }
   │   }
   ↓
7. ACPClient: Parse and return response
   ↓
8. QuantaAlpha: Use generated hypothesis
```

### Embedding 流程

```
1. QuantaAlpha: FactorLoader.calculate_embedding_distance()
   ↓
2. APIBackend.create_embedding()
   ↓
3. ACP Patch: Intercept if USE_ACP_BACKEND=true
   ↓
4. ACPBackend.create_embedding()
   │
   ├─→ Check EXTERNAL_EMBEDDING_API env var
   │
   ├─→ If set: Use external API (硅基流动/GLM)
   │   │
   │   ├─→ HTTP POST to EXTERNAL_EMBEDDING_API
   │   │   {
   │   │     "input": ["动量因子", "波动率因子"],
   │   │     "model": "BAAI/bge-large-zh-v1.5"
   │   │   }
   │   ↓
   │   └─→ Return embedding vectors
   │
   └─→ If not set: Use OpenCode (if supported)
       ↓
       Return embedding vectors
   ↓
5. QuantaAlpha: Calculate similarity matrix
```

## 核心组件

### 1. ACP Client (`acp_client.py`)

```python
class ACPClient:
    """Communicates with OpenCode via JSON-RPC over stdio"""

    - start(): Launch OpenCode subprocess
    - stop(): Terminate subprocess
    - chat_completion(): Send chat request
    - embedding(): Send embedding request
    - _send_request(): Core JSON-RPC method
```

### 2. ACP Patch (`acp_patch.py`)

```python
class ACPChatCompletionMixin:
    """Mixin to add ACP capability to APIBackend"""

def patch_apibackend():
    """Monkey-patch APIBackend at runtime"""
```

### 3. Configuration (`config.py`)

```python
class LLMSettings:
    use_acp_backend: bool = False
    acp_agent_command: str = "opencode"
    external_embedding_api: str = ""
    external_embedding_model: str = "BAAI/bge-large-zh-v1.5"
```

## 环境变量

| 变量 | 描述 | 默认值 |
|------|------|--------|
| `USE_ACP_BACKEND` | 启用 ACP 后端 | `false` |
| `ACP_AGENT_COMMAND` | OpenCode 命令 | `opencode` |
| `ACP_AGENT_ARGS` | OpenCode 参数 | `acp` |
| `EXTERNAL_EMBEDDING_API` | Embedding API URL | - |
| `EXTERNAL_EMBEDDING_API_KEY` | Embedding API Key | - |
| `EXTERNAL_EMBEDDING_MODEL` | Embedding 模型 | `BAAI/bge-large-zh-v1.5` |
| `EMBEDDING_BATCH_SIZE` | Embedding 批大小 | `10` |

## 部署模式

### 模式 A: 完全本地

```
QuantaAlpha → OpenCode → Ollama (本地模型)
                 ↓
              本地 Embedding
```

### 模式 B: 混合云 (推荐)

```
QuantaAlpha → OpenCode → 云端 LLM (Claude/DeepSeek)
                 ↓
              硅基流动 Embedding API
```

### 模式 C: 传统模式 (回退)

```
QuantaAlpha → OpenAI API 直接调用
                 ↓
              OpenAI Embedding
```

## 兼容性

| QuantaAlpha 组件 | ACP 后端支持 | 备注 |
|------------------|--------------|------|
| LLMHypothesisGen | ✅ | 完全支持 |
| LLMHypothesis2Experiment | ✅ | JSON 模式支持 |
| FactorHypothesisGen | ✅ | 完全支持 |
| MutationOperator | ✅ | 完全支持 |
| CrossoverOperator | ✅ | 完全支持 |
| FeedbackGenerator | ✅ | 完全支持 |
| Vector Similarity | ✅ | 外部 API |
| ChatSession | ✅ | 会话管理 |

## 性能考虑

1. **启动开销**: OpenCode 首次启动 ~1-2 秒
2. **通信延迟**: JSON-RPC stdio ~10-50ms
3. **缓存兼容**: SQLite 缓存仍然有效
4. **并发处理**: 单实例 OpenCode，顺序处理请求

## 故障转移

```python
try:
    # Try ACP backend
    response = acp_backend.chat_completion(...)
except Exception as e:
    logger.warning(f"ACP failed: {e}, falling back to OpenAI")
    # Fallback to original OpenAI backend
    response = openai_backend.chat_completion(...)
```
