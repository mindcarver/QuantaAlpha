"""
Test script for ACP backend integration.

Usage:
    python tests/test_acp_backend.py
"""

import os
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pathlib import Path


def test_acp_client():
    """Test basic ACP client functionality."""
    from quantaalpha.llm.acp_client import ACPClient

    print("Testing ACP Client...")

    # Check if OpenCode is available
    import shutil
    if not shutil.which("opencode"):
        print("⚠️  OpenCode not found in PATH")
        print("   Install with: npm install -g @opencode-ai/opencode")
        return False

    print("✓ OpenCode found")

    # Test client creation
    client = ACPClient()
    print("✓ ACP Client created")

    # Note: Don't start the agent in test unless explicitly requested
    # as it requires user interaction
    print("✓ ACP Client test passed (agent not started)")
    return True


def test_external_embedding():
    """Test external embedding API."""
    api_url = os.environ.get("EXTERNAL_EMBEDDING_API")
    api_key = os.environ.get("EXTERNAL_EMBEDDING_API_KEY")

    if not api_url:
        print("⚠️  EXTERNAL_EMBEDDING_API not set")
        return False

    print(f"Testing external embedding API: {api_url}")

    try:
        import requests

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        response = requests.post(
            api_url,
            headers=headers,
            json={"input": ["测试文本"], "model": os.environ.get("EXTERNAL_EMBEDDING_MODEL", "embedding-model")},
            timeout=10,
        )

        if response.status_code == 200:
            data = response.json()
            if "data" in data or "embeddings" in data:
                print("✓ External embedding API test passed")
                return True

        print(f"✗ Embedding API error: {response.status_code}")
        return False

    except Exception as e:
        print(f"✗ Embedding API test failed: {e}")
        return False


def test_acp_patch():
    """Test ACP patch functionality."""
    from quantaalpha.llm.acp_patch import is_acp_enabled, patch_apibackend

    print("Testing ACP patch...")

    if not is_acp_enabled():
        print("⚠️  ACP backend not enabled")
        print("   Enable with: export USE_ACP_BACKEND=true")
        return False

    print("✓ ACP backend is enabled")

    # Test patching
    patch_apibackend()
    print("✓ ACP patch applied")

    return True


def test_full_integration():
    """Test full integration with ACP backend."""
    import os

    if not os.environ.get("USE_ACP_BACKEND"):
        print("⚠️  Skipping full integration test (ACP not enabled)")
        print("   Enable with: export USE_ACP_BACKEND=true")
        return None

    from quantaalpha.llm.acp_patch import patch_apibackend
    from quantaalpha.llm.client import APIBackend

    print("Testing full integration...")

    # Apply patch
    patch_apibackend()

    # Create backend
    backend = APIBackend()
    print("✓ APIBackend created with ACP support")

    # Test chat completion (this will start OpenCode)
    try:
        response = backend.build_messages_and_create_chat_completion(
            user_prompt="Hello, say 'OK' if you receive this.",
            system_prompt="You are a helpful assistant.",
            temperature=0.1,
            max_tokens=10,
        )
        print(f"✓ Chat completion response: {response[:100]}...")
        return True
    except Exception as e:
        print(f"✗ Chat completion failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("ACP Backend Integration Tests")
    print("=" * 60)
    print()

    results = {
        "ACP Client": test_acp_client(),
        "External Embedding": test_external_embedding(),
        "ACP Patch": test_acp_patch(),
    }

    # Only run full integration if explicitly requested
    if os.environ.get("TEST_FULL_INTEGRATION"):
        results["Full Integration"] = test_full_integration()

    print()
    print("=" * 60)
    print("Test Results:")
    print("=" * 60)
    for name, result in results.items():
        if result is True:
            print(f"✓ {name}: PASSED")
        elif result is False:
            print(f"✗ {name}: FAILED")
        else:
            print(f"○ {name}: SKIPPED")

    # Return exit code
    failed = sum(1 for r in results.values() if r is False)
    sys.exit(failed)


if __name__ == "__main__":
    main()
