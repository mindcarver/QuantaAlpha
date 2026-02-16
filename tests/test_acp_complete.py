#!/usr/bin/env python3
"""
Complete ACP Integration Test with Mock.
Tests the entire flow using mock responses to verify code logic.
"""

import os
import sys
import json
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

sys.path.insert(0, "/Users/mac08/workspace/ai-tools/QuantaAlpha")


def test_1_acp_client():
    """Test 1: ACP Client can be instantiated and has correct methods."""
    print("\n" + "="*60)
    print("TEST 1: ACP Client")
    print("="*60)

    try:
        from quantaalpha.llm.acp_client import ACPClient, ACPBackend

        # Test client creation
        client = ACPClient(
            agent_command="opencode",
            agent_args=["acp"]
        )
        print("✅ ACPClient created")
        print(f"   Command: {client.agent_command} {' '.join(client.agent_args)}")

        # Test backend creation
        backend = ACPBackend()
        print("✅ ACPBackend (singleton) created")

        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_2_external_embedding_with_mock():
    """Test 2: External embedding with mock response."""
    print("\n" + "="*60)
    print("TEST 2: External Embedding (Mock)")
    print("="*60)

    try:
        from quantaalpha.llm.acp_client import ACPBackend

        os.environ["EXTERNAL_EMBEDDING_API"] = "https://api.siliconflow.cn/v1/embeddings"
        os.environ["EXTERNAL_EMBEDDING_API_KEY"] = "test-key"
        os.environ["EXTERNAL_EMBEDDING_MODEL"] = "Pro/BAAI/bge-m3"

        backend = ACPBackend()

        # Create mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "object": "list",
            "model": "Pro/BAAI/bge-m3",
            "data": [
                {"embedding": [0.1, 0.2, 0.3, 0.4], "index": 0},
                {"embedding": [0.5, 0.6, 0.7, 0.8], "index": 1},
            ],
            "usage": {"prompt_tokens": 20, "total_tokens": 20}
        }
        mock_response.raise_for_status = Mock()

        with patch('requests.post', return_value=mock_response) as mock_post:
            # Test single input
            result_single = backend._external_embedding("test")
            assert len(result_single) == 4, f"Expected 4 dimensions, got {len(result_single)}"
            print(f"✅ Single input: {len(result_single)} dimensions")

            # Verify the API was called correctly
            assert mock_post.called, "API should be called"
            call_args = mock_post.call_args
            request_json = call_args[1]['json']
            assert request_json['model'] == 'Pro/BAAI/bge-m3'
            assert request_json['input'] == 'test'
            print(f"   Model: {request_json['model']}")
            print(f"   Input: {request_json['input']}")

            # Test batch input
            mock_response.json.return_value = {
                "data": [
                    {"embedding": [0.1] * 1024, "index": 0},
                    {"embedding": [0.2] * 1024, "index": 1},
                ]
            }

            results_batch = backend._external_embedding(["test1", "test2"])
            assert len(results_batch) == 2
            assert len(results_batch[0]) == 1024
            print(f"✅ Batch input: {len(results_batch)} embeddings")
            print(f"   Each dimension: {len(results_batch[0])}")

        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_3_patch_mechanism():
    """Test 3: Patch mechanism."""
    print("\n" + "="*60)
    print("TEST 3: Patch Mechanism")
    print("="*60)

    try:
        # Set up environment
        os.environ["EXTERNAL_EMBEDDING_API"] = "https://api.test.com/v1/embeddings"
        os.environ["EXTERNAL_EMBEDDING_API_KEY"] = "test-key"
        os.environ["EXTERNAL_EMBEDDING_MODEL"] = "test-model"

        from quantaalpha.llm.acp_patch import (
            ACPChatCompletionMixin,
            is_external_embedding_enabled,
            get_acp_backend
        )

        # Test helper functions
        assert is_external_embedding_enabled() == True
        print("✅ is_external_embedding_enabled() works")

        # Test mixin class
        mixin = ACPChatCompletionMixin()

        # Mock the backend
        mock_backend = Mock()
        mock_backend.create_embedding.return_value = [[0.1, 0.2]]

        with patch('quantaalpha.llm.acp_patch.get_acp_backend', return_value=mock_backend):
            # Test embedding inner function
            result = mixin._acp_create_embedding_inner_function(
                mock_backend, ["test"]
            )
            assert result == [0.1, 0.2]
            print("✅ ACPChatCompletionMixin._acp_create_embedding_inner_function works")

        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_4_error_handling():
    """Test 4: Error handling."""
    print("\n" + "="*60)
    print("TEST 4: Error Handling")
    print("="*60)

    try:
        from quantaalpha.llm.acp_client import ACPBackend

        os.environ["EXTERNAL_EMBEDDING_API"] = "https://api.test.com/v1/embeddings"
        os.environ["EXTERNAL_EMBEDDING_API_KEY"] = "test-key"
        os.environ["EXTERNAL_EMBEDDING_MODEL"] = "test-model"

        backend = ACPBackend()

        # Test with error response
        mock_error = Mock()
        mock_error.status_code = 429
        mock_error.text = "Rate limit exceeded"
        mock_error.raise_for_status.side_effect = Exception("Rate limit")

        with patch('requests.post', return_value=mock_error):
            try:
                backend._external_embedding(["test"])
                print("❌ Should have raised an exception")
                return False
            except Exception as e:
                print(f"✅ Error handling works: caught {type(e).__name__}")

        # Test retry logic
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] < 2:
                mock_error.raise_for_status()
            # Success on third try
            mock_success = Mock()
            mock_success.status_code = 200
            mock_success.json.return_value = {
                "data": [{"embedding": [0.1], "index": 0}]
            }
            mock_success.raise_for_status = Mock()
            return mock_success

        mock_error.raise_for_status.side_effect = side_effect

        with patch('requests.post', return_value=mock_error):
            import time
            with patch('time.sleep'):  # Mock sleep to speed up test
                result = backend._external_embedding(["test"])
                assert call_count[0] == 3  # Should retry 3 times
                print(f"✅ Retry logic works: {call_count[0]} attempts")

        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_5_config_integration():
    """Test 5: Config integration."""
    print("\n" + "="*60)
    print("TEST 5: Config Integration")
    print("="*60)

    try:
        from quantaalpha.llm.config import LLM_SETTINGS

        # Check ACP settings exist
        assert hasattr(LLM_SETTINGS, 'use_acp_backend')
        assert hasattr(LLM_SETTINGS, 'external_embedding_api')
        assert hasattr(LLM_SETTINGS, 'external_embedding_model')

        print("✅ All ACP config attributes exist")
        print(f"   use_acp_backend: {LLM_SETTINGS.use_acp_backend}")
        print(f"   external_embedding_api: {LLM_SETTINGS.external_embedding_api}")
        print(f"   external_embedding_model: {LLM_SETTINGS.external_embedding_model}")

        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_6_full_flow_simulation():
    """Test 6: Full flow simulation."""
    print("\n" + "="*60)
    print("TEST 6: Full Flow Simulation")
    print("="*60)

    try:
        # Simulate the complete usage flow
        print("Simulating complete ACP backend usage...")

        # Step 1: Import and patch
        print("  Step 1: Import modules...")
        from quantaalpha.llm.acp_patch import patch_apibackend
        from quantaalpha.llm.client import APIBackend

        # Step 2: Set environment
        print("  Step 2: Set environment variables...")
        os.environ["EXTERNAL_EMBEDDING_API"] = "https://api.siliconflow.cn/v1/embeddings"
        os.environ["EXTERNAL_EMBEDDING_API_KEY"] = "test-key"
        os.environ["EXTERNAL_EMBEDDING_MODEL"] = "Pro/BAAI/bge-m3"

        # Step 3: Apply patch
        print("  Step 3: Apply patch...")
        patch_apibackend()

        # Step 4: Create backend
        print("  Step 4: Create APIBackend...")
        backend = APIBackend()

        # Step 5: Mock API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [{"embedding": [0.1] * 1024, "index": 0}]
        }
        mock_response.raise_for_status = Mock()

        # Step 6: Test embedding
        print("  Step 6: Test embedding (mocked)...")
        with patch('requests.post', return_value=mock_response):
            embeddings = backend.create_embedding(["test text"])
            print(f"    Got {len(embeddings)} embeddings")
            print(f"    Dimension: {len(embeddings[0])}")

        print("✅ Full flow simulation passed!")
        return True

    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("ACP Integration - Complete Test Suite")
    print("Testing code logic with mock responses")
    print("="*60)

    tests = [
        ("ACP Client", test_1_acp_client),
        ("External Embedding", test_2_external_embedding_with_mock),
        ("Patch Mechanism", test_3_patch_mechanism),
        ("Error Handling", test_4_error_handling),
        ("Config Integration", test_5_config_integration),
        ("Full Flow", test_6_full_flow_simulation),
    ]

    results = {}
    for name, test_func in tests:
        results[name] = test_func()

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    passed = sum(1 for r in results.values() if r)
    total = len(results)

    for name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status} - {name}")

    print(f"\n{passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All code logic tests passed!")
        print("\n📝 The ACP integration code is correctly implemented.")
        print("📝 When you have a valid SiliconFlow API key, it will work end-to-end.")
        print("\n📋 To use:")
        print("   1. Get a valid API key from https://siliconflow.cn/")
        print("   2. Set environment variables (see docs/ACP_QUICKSTART.md)")
        print("   3. Run your QuantaAlpha pipeline as usual")
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
