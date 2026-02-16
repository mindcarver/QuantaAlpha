"""
ACP Client for QuantaAlpha - communicates with OpenCode via ACP protocol.

This module implements an ACP client that can communicate with OpenCode
running as an ACP agent subprocess, replacing direct LLM API calls.
"""

from __future__ import annotations

import json
import subprocess
import threading
import queue
import os
from pathlib import Path
from typing import Any, Optional
import logging

logger = logging.getLogger(__name__)


class ACPClient:
    """
    Client for communicating with ACP-compatible agents like OpenCode.

    Uses JSON-RPC over stdio as specified in the ACP protocol.
    """

    def __init__(
        self,
        agent_command: str = "opencode",
        agent_args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ):
        """
        Initialize the ACP client.

        Args:
            agent_command: Command to start the ACP agent (default: "opencode")
            agent_args: Arguments to pass to the agent (default: ["acp"])
            env: Environment variables to pass to the agent process
        """
        self.agent_command = agent_command
        self.agent_args = agent_args or ["acp"]
        self.env = env or os.environ.copy()

        self.process: Optional[subprocess.Popen] = None
        self._request_id = 0
        self._response_queue: queue.Queue = queue.Queue()
        self._notification_queue: queue.Queue = queue.Queue()
        self._running = False

    def start(self) -> None:
        """Start the ACP agent subprocess."""
        if self.process is not None:
            logger.warning("ACP agent is already running")
            return

        cmd = [self.agent_command] + self.agent_args
        logger.info(f"Starting ACP agent: {' '.join(cmd)}")

        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # Line buffered
            env=self.env,
        )

        self._running = True

        # Start reader thread for stdout
        self._stdout_thread = threading.Thread(
            target=self._read_stdout,
            daemon=True,
        )
        self._stdout_thread.start()

        # Start reader thread for stderr
        self._stderr_thread = threading.Thread(
            target=self._read_stderr,
            daemon=True,
        )
        self._stderr_thread.start()

        # Initialize the connection
        self._initialize()

    def stop(self) -> None:
        """Stop the ACP agent subprocess."""
        if not self._running:
            return

        self._running = False

        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            finally:
                self.process = None

    def _read_stdout(self) -> None:
        """Read messages from agent's stdout."""
        if not self.process or not self.process.stdout:
            return

        for line in iter(self.process.stdout.readline, ""):
            if not line:
                break
            line = line.strip()
            if line:
                try:
                    message = json.loads(line)
                    if message.get("jsonrpc") == "2.0":
                        if "id" in message:
                            # Response to a request
                            self._response_queue.put(message)
                        else:
                            # Notification
                            self._notification_queue.put(message)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse ACP message: {e}")

    def _read_stderr(self) -> None:
        """Read logs from agent's stderr."""
        if not self.process or not self.process.stderr:
            return

        for line in iter(self.process.stderr.readline, ""):
            if not line:
                break
            line = line.strip()
            if line:
                logger.debug(f"ACP Agent stderr: {line}")

    def _get_next_request_id(self) -> int:
        """Get the next request ID."""
        self._request_id += 1
        return self._request_id

    def _send_request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        timeout: float = 120.0,
    ) -> Any:
        """
        Send a JSON-RPC request and wait for the response.

        Args:
            method: The JSON-RPC method name
            params: The parameters for the method
            timeout: Timeout in seconds

        Returns:
            The result from the response

        Raises:
            RuntimeError: If the request fails or times out
        """
        if not self.process or not self.process.stdin:
            raise RuntimeError("ACP agent is not running")

        request_id = self._get_next_request_id()
        request = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params or {},
        }

        # Send request
        message = json.dumps(request) + "\n"
        self.process.stdin.write(message)
        self.process.stdin.flush()

        # Wait for response
        try:
            while True:
                try:
                    response = self._response_queue.get(timeout=timeout)
                    if response.get("id") == request_id:
                        if "error" in response:
                            raise RuntimeError(
                                f"ACP error: {response['error']}"
                            )
                        return response.get("result")
                    # Not our response, put it back
                    self._response_queue.put(response)
                except queue.Empty:
                    raise RuntimeError(f"ACP request timeout after {timeout}s")
        except Exception as e:
            raise RuntimeError(f"ACP request failed: {e}")

    def _initialize(self) -> None:
        """Initialize the ACP connection."""
        try:
            # Send initialize request
            self._send_request(
                "initialize",
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {
                        "name": "QuantaAlpha",
                        "version": "1.0.0",
                    },
                },
            )
            # Send initialized notification
            self._send_notification("initialized")
        except Exception as e:
            logger.error(f"Failed to initialize ACP connection: {e}")
            raise

    def _send_notification(
        self,
        method: str,
        params: dict[str, Any] | None = None,
    ) -> None:
        """Send a JSON-RPC notification (no response expected)."""
        if not self.process or not self.process.stdin:
            raise RuntimeError("ACP agent is not running")

        notification = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
        }

        message = json.dumps(notification) + "\n"
        self.process.stdin.write(message)
        self.process.stdin.flush()

    def chat_completion(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> str:
        """
        Request a chat completion from the ACP agent.

        Args:
            messages: The conversation messages
            model: The model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters

        Returns:
            The generated text response
        """
        params = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if model:
            params["model"] = model
        params.update(kwargs)

        result = self._send_request("chat/completions", params)

        # Handle different response formats
        if isinstance(result, dict):
            if "content" in result:
                return result["content"]
            if "choices" in result and len(result["choices"]) > 0:
                choice = result["choices"][0]
                if "message" in choice:
                    return choice["message"].get("content", "")
                return choice.get("text", "")
            if "text" in result:
                return result["text"]

        return str(result)

    def embedding(
        self,
        input: str | list[str],
        model: str | None = None,
        **kwargs: Any,
    ) -> list[list[float]]:
        """
        Request embeddings from the ACP agent.

        Note: OpenCode may not support embeddings natively.
        This method can be configured to use an external embedding API.

        Args:
            input: The text(s) to embed
            model: The embedding model to use
            **kwargs: Additional parameters

        Returns:
            List of embedding vectors
        """
        # For embeddings, we might use an external API
        # This is a placeholder that can be configured
        params = {
            "input": input,
        }
        if model:
            params["model"] = model
        params.update(kwargs)

        result = self._send_request("embeddings", params)

        # Handle different response formats
        if isinstance(result, dict):
            if "data" in result:
                return [item["embedding"] for item in result["data"]]
            if "embeddings" in result:
                return result["embeddings"]

        return result

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


class ACPBackend:
    """
    ACP-based backend for LLM operations.

    This replaces the OpenAI API calls with ACP communication to OpenCode.
    """

    _instance: Optional["ACPBackend"] = None
    _client: Optional[ACPClient] = None

    def __new__(cls) -> "ACPBackend":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._client is None:
            self._client = ACPClient(
                agent_command=os.environ.get("ACP_AGENT_COMMAND", "opencode"),
                agent_args=os.environ.get("ACP_AGENT_ARGS", "acp").split(),
            )

    @property
    def client(self) -> ACPClient:
        """Get the ACP client, starting it if necessary."""
        if self._client is None or self._client.process is None:
            self._client.start()
        return self._client

    def chat_completion(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> str:
        """Perform chat completion via ACP."""
        return self.client.chat_completion(messages, **kwargs)

    def create_embedding(
        self,
        input: str | list[str],
        **kwargs: Any,
    ) -> list[list[float]]:
        """Create embeddings via ACP or external API."""
        # Check if we should use external embedding API
        if os.environ.get("EXTERNAL_EMBEDDING_API"):
            return self._external_embedding(input, **kwargs)
        return self.client.embedding(input, **kwargs)

    def _external_embedding(
        self,
        input: str | list[str],
        **kwargs: Any,
    ) -> list[list[float]]:
        """Use external embedding API (e.g., 硅基流动 SiliconFlow, GLM)."""
        import requests
        import time

        api_url = os.environ.get("EXTERNAL_EMBEDDING_API")
        api_key = os.environ.get("EXTERNAL_EMBEDDING_API_KEY")
        model = os.environ.get("EXTERNAL_EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")

        if not api_url:
            raise ValueError("EXTERNAL_EMBEDDING_API not configured")
        if not api_key:
            raise ValueError("EXTERNAL_EMBEDDING_API_KEY not configured")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        inputs_list = [input] if isinstance(input, str) else input
        embeddings = []

        # Process in batches (SiliconFlow max batch size is 32 for bge models)
        batch_size = int(os.environ.get("EMBEDDING_BATCH_SIZE", "10"))
        batch_wait = float(os.environ.get("EMBEDDING_BATCH_WAIT_SECONDS", "0.5"))

        logger.info(f"Creating embeddings for {len(inputs_list)} inputs with model {model}")

        for i in range(0, len(inputs_list), batch_size):
            batch = inputs_list[i:i + batch_size]
            logger.debug(f"Processing batch {i//batch_size + 1} with {len(batch)} inputs")

            # Retry logic
            max_retries = 3
            for retry in range(max_retries):
                try:
                    response = requests.post(
                        api_url,
                        headers=headers,
                        json={"input": batch, "model": model, "encoding_format": "float"},
                        timeout=30,
                    )
                    response.raise_for_status()
                    data = response.json()

                    if "data" in data:
                        batch_embeddings = [item["embedding"] for item in data["data"]]
                        embeddings.extend(batch_embeddings)
                        logger.debug(f"Got {len(batch_embeddings)} embeddings from batch")
                    else:
                        logger.warning(f"Unexpected response format: {list(data.keys())}")
                        embeddings.extend(data.get("embeddings", []))

                    # Log usage info if available
                    if "usage" in data:
                        logger.debug(f"Token usage: {data['usage']}")

                    break  # Success, exit retry loop

                except requests.exceptions.HTTPError as e:
                    if e.response.status_code == 429:  # Rate limit
                        wait_time = (retry + 1) * 2
                        logger.warning(f"Rate limited, waiting {wait_time}s before retry")
                        time.sleep(wait_time)
                    elif e.response.status_code == 401:
                        raise ValueError("Invalid API key for external embedding API") from e
                    else:
                        logger.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
                        if retry == max_retries - 1:
                            raise
                        time.sleep(1)

                except requests.exceptions.RequestException as e:
                    logger.error(f"Request error: {e}")
                    if retry == max_retries - 1:
                        raise
                    time.sleep(1)

            # Wait between batches to avoid rate limiting
            if i + batch_size < len(inputs_list) and batch_wait > 0:
                time.sleep(batch_wait)

        logger.info(f"Successfully created {len(embeddings)} embeddings")
        return embeddings if isinstance(input, list) else embeddings[0]

    def shutdown(self) -> None:
        """Shutdown the ACP client."""
        if self._client:
            self._client.stop()
