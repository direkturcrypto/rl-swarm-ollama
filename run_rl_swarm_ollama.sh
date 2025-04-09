#!/bin/bash

#General args
ROOT=$PWD

export PUB_MULTI_ADDRS
export PEER_MULTI_ADDRS
export HOST_MULTI_ADDRS
export IDENTITY_PATH
export CONNECT_TO_TESTNET
export ORG_ID
export HF_HUB_DOWNLOAD_TIMEOUT=120  # 2 minutes

#Check if public multi-address is given else set to default
DEFAULT_PUB_MULTI_ADDRS=""
PUB_MULTI_ADDRS=${PUB_MULTI_ADDRS:-$DEFAULT_PUB_MULTI_ADDRS}

#Check if peer multi-address is given else set to default
DEFAULT_PEER_MULTI_ADDRS="/ip4/38.101.215.13/tcp/30002/p2p/QmQ2gEXoPJg6iMBSUFWGzAabS2VhnzuS782Y637hGjfsRJ" # gensyn coordinator node
PEER_MULTI_ADDRS=${PEER_MULTI_ADDRS:-$DEFAULT_PEER_MULTI_ADDRS}

#Check if host multi-address is given else set to default
DEFAULT_HOST_MULTI_ADDRS="/ip4/0.0.0.0/tcp/38331"
HOST_MULTI_ADDRS=${HOST_MULTI_ADDRS:-$DEFAULT_HOST_MULTI_ADDRS}

# Path to an RSA private key. If this path does not exist, a new key pair will be created.
# Remove this file if you want a new PeerID.
DEFAULT_IDENTITY_PATH="$ROOT"/swarm.pem
IDENTITY_PATH=${IDENTITY_PATH:-$DEFAULT_IDENTITY_PATH}

# Check if Ollama is installed and running
if ! command -v ollama &> /dev/null; then
    echo "Ollama is not installed. Please install Ollama first."
    echo "Visit https://ollama.ai/download for installation instructions."
    exit 1
fi

# Check if Ollama service is running
if ! curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "Ollama service is not running. Starting Ollama..."
    ollama serve &
    sleep 5
fi

# Pull the Qwen model if not already present
echo "Checking for Qwen model..."
if ! ollama list | grep -q "qwen2.5:0.5b"; then
    echo "Pulling Qwen 0.5B model..."
    ollama pull qwen2.5:0.5b
fi

# Install requirements
echo "Installing requirements..."
pip install -r "$ROOT"/requirements.txt

# Set config path based on Ollama
CONFIG_PATH="$ROOT/hivemind_exp/configs/ollama/grpo-qwen-2.5-0.5b-ollama.yaml"

echo "Starting training with Ollama..."
python -m hivemind_exp.gsm8k.train_ollama \
    --config "$CONFIG_PATH" \
    --identity_path "$IDENTITY_PATH" \
    --public_maddr "$PUB_MULTI_ADDRS" \
    --initial_peers "$PEER_MULTI_ADDRS" \
    --host_maddr "$HOST_MULTI_ADDRS"

wait  # Keep script running until Ctrl+C 