import logging
import requests
import colorlog
import yaml
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import argparse

from trl import GRPOConfig, ModelConfig, TrlParser

from hivemind_exp.chain_utils import (
    ModalSwarmCoordinator,
    WalletSwarmCoordinator,
    setup_web3,
)
from hivemind_exp.gsm8k.generate_prompts import get_stage1_samples
from hivemind_exp.runner.gensyn.testnet_grpo_runner import (
    TestnetGRPOArguments,
    TestnetGRPORunner,
)
from hivemind_exp.runner.grpo_runner import GRPOArguments, GRPORunner
from hivemind.utils.logging import get_logger

# Setup logging
logger = get_logger(__name__)

# Simple telemetry function
def log_telemetry(event_name, **kwargs):
    """Simple telemetry function to log events"""
    logger.info(f"TELEMETRY: {event_name} - {kwargs}")

@dataclass
class OllamaConfig:
    """Konfigurasi untuk Ollama API"""
    base_url: str
    model_name: str

class OllamaGRPORunner(GRPORunner):
    """GRPORunner yang menggunakan Ollama API alih-alih model lokal"""
    
    def __init__(self, ollama_config: OllamaConfig, coordinator=None):
        # Initialize parent class without coordinator first
        super().__init__()
        # Then set coordinator if provided
        if coordinator:
            self.coordinator = coordinator
        self.ollama_config = ollama_config
        
    def generate_response(self, prompt: str, max_tokens: int = 1024) -> str:
        """Menghasilkan respons dari Ollama API"""
        try:
            response = requests.post(
                f"{self.ollama_config.base_url}/api/generate",
                json={
                    "model": self.ollama_config.model_name,
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": 0.7
                }
            )
            response.raise_for_status()
            return response.json()['response']
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return ""
    
    def run(self, model_args, grpo_args, training_args, get_samples_fn):
        """Menjalankan training dengan Ollama"""
        # Override metode run dari parent class
        logger.info("Starting training with Ollama...")
        log_telemetry("training_started", model=self.ollama_config.model_name)
        
        # Dapatkan dataset
        samples = get_samples_fn()
        
        # Loop training
        for step in range(training_args.max_steps):
            # Ambil batch
            batch = next(iter(samples))
            
            # Hasilkan respons dengan Ollama
            prompts = batch['question']
            responses = [self.generate_response(prompt, training_args.max_new_tokens) for prompt in prompts]
            
            # Log hasil
            if step % training_args.logging_steps == 0:
                logger.info(f"Step {step}")
                log_telemetry("training_step", step=step)
            
            # Simpan checkpoint
            if step % training_args.save_steps == 0:
                logger.info(f"Saving checkpoint at step {step}")
                log_telemetry("checkpoint_saved", step=step)
        
        log_telemetry("training_completed", steps=training_args.max_steps)

def load_config(config_path: str) -> dict:
    """Muat konfigurasi dari file YAML"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    # Setup logging.
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    handler = colorlog.StreamHandler()
    handler.setFormatter(
        colorlog.ColoredFormatter("%(light_red)s%(levelname)s:%(name)s:%(message)s")
    )
    root_logger.addHandler(handler)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--hf_token", type=str, default="None", help="Hugging Face token")
    parser.add_argument("--identity_path", type=str, default="", help="Path to identity file")
    parser.add_argument("--modal_org_id", type=str, default="", help="Modal organization ID")
    parser.add_argument("--public_maddr", type=str, default="", help="Public multi-address")
    parser.add_argument("--initial_peers", type=str, default="", help="Initial peers")
    parser.add_argument("--host_maddr", type=str, default="", help="Host multi-address")
    cli_args = parser.parse_args()
    
    # Load config
    config_dict = load_config(cli_args.config)
    
    # Setup TRL parser untuk kompatibilitas dengan runner
    parser = TrlParser((ModelConfig, GRPOArguments, TestnetGRPOArguments, GRPOConfig))
    model_args, grpo_args, testnet_args, training_args = parser.parse_args_and_config()
    
    # Setup Ollama config
    ollama_config = OllamaConfig(
        base_url=config_dict.get("ollama_base_url", "http://107.222.215.224:36001"),
        model_name=config_dict.get("model_name", "qwen2.5:0.5b")
    )

    # Run main training loop with appropriate coordinator
    if org_id := cli_args.modal_org_id or testnet_args.modal_org_id:
        runner = OllamaGRPORunner(
            ollama_config,
            coordinator=ModalSwarmCoordinator(org_id, web3=setup_web3())
        )
    elif priv_key := testnet_args.wallet_private_key:
        runner = OllamaGRPORunner(
            ollama_config,
            coordinator=WalletSwarmCoordinator(priv_key, web3=setup_web3())
        )
    else:
        runner = OllamaGRPORunner(ollama_config)

    runner.run(model_args, grpo_args, training_args, get_stage1_samples)

if __name__ == "__main__":
    main() 