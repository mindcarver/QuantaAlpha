# ACP 集成 - SiliconFlow Embedding 配置说明

## 🎯 推荐的 Embedding 模型

根据硅基流动文档，以下是推荐的 embedding 模型：

### 中文嵌入模型（推荐）

| 模型 | 维度 | 最大 Token | 说明 |
|------|------|-----------|------|
| `Pro/BAAI/bge-m3` | 1024 | 8192 | **推荐**，支持长文本 |
| `BAAI/bge-large-zh-v1.5` | 1024 | 512 | 中文嵌入经典模型 |
| `Qwen/Qwen3-Embedding-8B` | 可变 | 32768 | Qwen3 系列，支持多维度 |

### 英文嵌入模型

| 模型 | 维度 | 最大 Token |
|------|------|-----------|
| `BAAI/bge-large-en-v1.5` | 1024 | 512 |
| `Pro/BAAI/bge-large-en-v1.5` | 1024 | 8192 |

## 🔧 环境变量配置

### 方式 1: 终端命令

```bash
export EXTERNAL_EMBEDDING_API=https://api.siliconflow.cn/v1/embeddings
export EXTERNAL_EMBEDDING_API_KEY=sk-你的API密钥
export EXTERNAL_EMBEDDING_MODEL=Pro/BAAI/bge-m3
export EMBEDDING_BATCH_SIZE=10
```

### 方式 2: .env 文件

```bash
EXTERNAL_EMBEDDING_API=https://api.siliconflow.cn/v1/embeddings
EXTERNAL_EMBEDDING_API_KEY=sk-你的API密钥
EXTERNAL_EMBEDDING_MODEL=Pro/BAAI/bge-m3
EMBEDDING_BATCH_SIZE=10
```

## 🧪 测试 API Key

使用以下命令测试你的 API key 是否有效：

```bash
curl -X POST https://api.siliconflow.cn/v1/embeddings \
  -H "Authorization: Bearer sk-你的API密钥" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Pro/BAAI/bge-m3",
    "input": "测试文本"
  }'
```

预期响应：
```json
{
  "object": "list",
  "model": "Pro/BAAI/bge-m3",
  "data": [
    {
      "object": "embedding",
      "embedding": [0.1, 0.2, ...],
      "index": 0
    }
  ],
  "usage": {...}
}
```

## ❌ 故障排查

### 401 Unauthorized

**原因**: API key 无效或过期

**解决**:
1. 访问 https://siliconflow.cn/ 控制台
2. 检查 API key 状态
3. 重新生成 API key

### 模型不存在

**原因**: 模型名称错误

**解决**:
1. 使用正确的模型名称（注意大小写）
2. 查看官方文档确认可用模型
3. 推荐使用 `Pro/BAAI/bge-m3`（稳定性好）

### 400 Bad Request

**原因**: 请求格式错误或超出限制

**解决**:
1. 检查输入文本长度（不同模型有不同限制）
2. 确保 encoding_format 设置正确
3. 减小 batch_size

## 📝 代码示例

```python
from quantaalpha.llm.acp_patch import patch_apibackend
from quantaalpha.llm.client import APIBackend
import os

# 配置
os.environ["EXTERNAL_EMBEDDING_API"] = "https://api.siliconflow.cn/v1/embeddings"
os.environ["EXTERNAL_EMBEDDING_API_KEY"] = "sk-你的密钥"
os.environ["EXTERNAL_EMBEDDING_MODEL"] = "Pro/BAAI/bge-m3"

# 应用补丁
patch_apibackend()

# 使用
backend = APIBackend()
embeddings = backend.create_embedding(["动量因子", "波动率因子"])
print(f"Got {len(embeddings)} embeddings, dim={len(embeddings[0])}")
```

## 💡 提示

1. **API Key 保护**: 不要将 API key 提交到代码仓库
2. **环境变量**: 始终使用环境变量存储敏感信息
3. **限额管理**: 注意 API 调用频率限制
4. **模型选择**: 根据需求选择合适的模型（中文用 bge-m3，英文用 bge-en）
