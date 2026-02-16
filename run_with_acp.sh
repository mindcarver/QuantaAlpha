#!/bin/bash
# QuantaAlpha with ACP Backend Launcher
#
# Usage:
#   ./run_with_acp.sh [quantalpha_args...]

set -e

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}QuantaAlpha with ACP Backend${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Check if OpenCode is installed
if ! command -v opencode &> /dev/null; then
    echo -e "${YELLOW}⚠️  OpenCode not found in PATH${NC}"
    echo "Installing OpenCode..."
    npm install -g @opencode-ai/opencode
fi

# Set default environment variables if not set
export USE_ACP_BACKEND="${USE_ACP_BACKEND:-true}"
export ACP_AGENT_COMMAND="${ACP_AGENT_COMMAND:-opencode}"
export ACP_AGENT_ARGS="${ACP_AGENT_ARGS:-acp}"

# Display configuration
echo "Configuration:"
echo "  USE_ACP_BACKEND=$USE_ACP_BACKEND"
echo "  ACP_AGENT_COMMAND=$ACP_AGENT_COMMAND"
echo "  ACP_AGENT_ARGS=$ACP_AGENT_ARGS"
if [ -n "$EXTERNAL_EMBEDDING_API" ]; then
    echo "  EXTERNAL_EMBEDDING_API=$EXTERNAL_EMBEDDING_API"
    echo "  EXTERNAL_EMBEDDING_MODEL=$EXTERNAL_EMBEDDING_MODEL"
fi
echo ""

# Check if external embedding is configured
if [ -z "$EXTERNAL_EMBEDDING_API" ] && [ -z "$EXTERNAL_EMBEDDING_API_KEY" ]; then
    echo -e "${YELLOW}⚠️  External embedding API not configured${NC}"
    echo "  Embedding will use OpenCode's internal method (if available)"
    echo ""
    echo "To configure external embedding (推荐):"
    echo "  export EXTERNAL_EMBEDDING_API=https://api.siliconflow.cn/v1/embeddings"
    echo "  export EXTERNAL_EMBEDDING_API_KEY=your_api_key"
    echo "  export EXTERNAL_EMBEDDING_MODEL=BAAI/bge-large-zh-v1.5"
    echo ""
fi

# Run QuantaAlpha with the remaining arguments
echo -e "${GREEN}Starting QuantaAlpha...${NC}"
echo ""

# Run the Python script with ACP patch
python -c "
import sys
sys.path.insert(0, '.')

from quantaalpha.llm.acp_patch import patch_apibackend
patch_apibackend()

# Import and run the main module
import quantaalpha
" "$@"

# Or for specific commands:
# python -m quantaalpha.pipeline "$@"
