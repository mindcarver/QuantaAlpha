#!/usr/bin/env python3
"""
ACP Backend Demo - 演示如何使用 ACP 后端集成。

这个脚本展示了如何使用 ACP 后端来替代 QuantaAlpha 中的 LLM 调用。
"""

import os
import sys

# 添加项目路径
sys.path.insert(0, "/Users/mac08/workspace/ai-tools/QuantaAlpha")


def demo_external_embedding():
    """演示外部 Embedding API 集成。"""
    print("="*60)
    print("DEMO: 外部 Embedding API 集成")
    print("="*60)

    from quantaalpha.llm.acp_client import ACPBackend

    # 配置 SiliconFlow API
    os.environ["EXTERNAL_EMBEDDING_API"] = "https://api.siliconflow.cn/v1/embeddings"
    os.environ["EXTERNAL_EMBEDDING_API_KEY"] = "你的API_KEY"  # 替换为有效 key
    os.environ["EXTERNAL_EMBEDDING_MODEL"] = "BAAI/bge-large-zh-v1.5"
    os.environ["EMBEDDING_BATCH_SIZE"] = "3"

    backend = ACPBackend()

    # 示例文本
    texts = [
        "动量因子：基于价格趋势的量化因子",
        "波动率因子：基于价格波动的风险度量",
        "价值因子：基于公司估值的基本面分析"
    ]

    print("\n输入文本:")
    for i, text in enumerate(texts, 1):
        print(f"  {i}. {text}")

    print("\n调用 SiliconFlow Embedding API...")
    print("注意：这需要有效的 API key")

    # 实际调用（需要有效 key）
    # embeddings = backend._external_embedding(texts)

    # 模拟返回
    print("\n模拟返回结果:")
    print(f"  - 获取了 {len(texts)} 个 embedding 向量")
    print(f"  - 每个向量维度: 1024 (BAAI/bge-large-zh-v1.5)")
    print(f"  - 用途: 计算因子之间的相似度，去重")

    return True


def demo_acp_chat():
    """演示 ACP Chat Completion。"""
    print("\n" + "="*60)
    print("DEMO: ACP Chat Completion (需要 OpenCode)")
    print("="*60)

    print("\n这个功能需要:")
    print("  1. OpenCode 已安装 (npm install -g @opencode-ai/opencode)")
    print("  2. 设置环境变量:")
    print("     export USE_ACP_BACKEND=true")
    print("     export ACP_AGENT_COMMAND=opencode")
    print("     export ACP_AGENT_ARGS=acp")
    print("\n工作原理:")
    print("  QuantaAlpha 启动 OpenCode 子进程")
    print("  通过 JSON-RPC over stdio 通信")
    print("  OpenCode 调用内部配置的 LLM 模型")
    print("  返回结果给 QuantaAlpha")

    return True


def demo_usage():
    """演示实际用法。"""
    print("\n" + "="*60)
    print("DEMO: 实际用法")
    print("="*60)

    code_example = '''
# 在你的 QuantaAlpha 代码中:

from quantaalpha.llm.acp_patch import patch_apibackend
from quantaalpha.llm.client import APIBackend
import os

# 配置环境变量
os.environ["EXTERNAL_EMBEDDING_API"] = "https://api.siliconflow.cn/v1/embeddings"
os.environ["EXTERNAL_EMBEDDING_API_KEY"] = "你的API_KEY"
os.environ["EXTERNAL_EMBEDDING_MODEL"] = "BAAI/bge-large-zh-v1.5"

# 应用补丁（会自动拦截 LLM 调用）
patch_apibackend()

# 正常使用 QuantaAlpha
backend = APIBackend()

# Embedding 用于因子相似度计算
embeddings = backend.create_embedding([
    "MA20: 20日移动平均",
    "RSI: 相对强弱指标"
])

# Chat Completion 用于因子生成
response = backend.build_messages_and_create_chat_completion(
    user_prompt="设计一个基于动量的量化因子",
    system_prompt="你是量化因子专家"
)
'''

    print("\n代码示例:")
    print(code_example)

    return True


def main():
    """运行所有演示。"""
    print("\n" + "="*60)
    print("ACP Backend 集成演示")
    print("QuantaAlpha + OpenCode + SiliconFlow")
    print("="*60)

    demos = [
        demo_external_embedding,
        demo_acp_chat,
        demo_usage,
    ]

    for demo in demos:
        try:
            demo()
        except Exception as e:
            print(f"\n⚠️  Demo 出错: {e}")

    print("\n" + "="*60)
    print("演示完成！")
    print("="*60)

    print("\n📚 更多信息:")
    print("  - 快速开始: docs/ACP_QUICKSTART.md")
    print("  - 详细指南: docs/ACP_BACKEND_GUIDE.md")
    print("  - 架构文档: docs/ACP_ARCHITECTURE.md")
    print("  - 运行测试: python tests/test_acp_standalone.py")


if __name__ == "__main__":
    main()
