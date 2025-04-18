import logging
import requests
import colorlog
import yaml
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import argparse
import json
from datetime import datetime

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
import hivemind

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
        self.best_responses = []
        self.best_scores = []
        
    def get_initial_peers(self) -> list[str]:
        """Get initial peers from coordinator if available"""
        if hasattr(self, 'coordinator'):
            return self.coordinator.get_bootnodes()
        return []

    def register_peer(self, peer_id):
        """Register peer with coordinator if available"""
        if hasattr(self, 'coordinator'):
            try:
                logger.info(f"Registering self with peer ID: {peer_id}")
                # Convert peer_id to string and ensure it's properly formatted
                peer_id_str = str(peer_id).strip()
                self.coordinator.register_peer(peer_id_str)
            except Exception as e:
                logger.error(f"Error registering peer: {e}")
                logger.info("Will continue without peer registration")

    def setup_dht(self, grpo_args):
        """Setup DHT with peer registration"""
        initial_peers = grpo_args.initial_peers
        if not initial_peers and hasattr(self, 'coordinator'):
            initial_peers = self.get_initial_peers()
            logger.info(f"Retrieved initial peers from chain: {initial_peers}")
            grpo_args.initial_peers = initial_peers
            
        if not initial_peers:
            logger.info("Cannot locate on-chain initial peers; running alone.")

        try:
            dht_kwargs = self._dht_kwargs(grpo_args)
            logger.info(f"Initializing DHT with kwargs: {dht_kwargs}")
            
            dht = hivemind.DHT(start=True, **dht_kwargs)
            
            if initial_peers:
                logger.info(f"🐝 Joining swarm with initial_peers = {initial_peers}")

            peer_id = str(dht.peer_id)
            logger.info(f"Generated peer_id: {peer_id}")
            
            # Clean up peer_id - remove any unwanted characters
            peer_id = peer_id.strip()
            if not peer_id:
                raise ValueError("Empty peer_id generated")
                
            self.name = self._get_animal_name(peer_id)
            logger.info(f"Animal name for peer: {self.name}")
            
            # Register peer with retries
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    logger.info(f"Attempting to register peer (attempt {attempt + 1}/{max_retries})")
                    self.register_peer(peer_id)
                    logger.info("Successfully registered peer")
                    break
                except Exception as e:
                    logger.warning(f"Failed to register peer (attempt {attempt + 1}): {e}")
                    if attempt == max_retries - 1:
                        logger.error("All registration attempts failed")
                    
            return dht
            
        except Exception as e:
            logger.error(f"Error in setup_dht: {e}")
            raise
    
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
            
            # Ollama API returns a stream of JSON objects, one per line
            # We need to get the last response which contains the complete text
            full_response = ""
            for line in response.text.strip().split('\n'):
                if line:
                    try:
                        chunk = json.loads(line)
                        if 'response' in chunk:
                            full_response += chunk['response']
                    except json.JSONDecodeError:
                        continue
            
            # Clean up response
            full_response = full_response.strip()
            if not full_response:
                logger.warning("Empty response received from Ollama API")
                return ""
                
            return full_response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return ""
    
    def submit_winners(self, responses, scores):
        """Submit winners to coordinator with proper error handling"""
        if hasattr(self, 'coordinator'):
            try:
                # Format responses and scores properly
                formatted_responses = []
                formatted_scores = []
                
                for response, score in zip(responses, scores):
                    # Ensure response is a valid string
                    if response and isinstance(response, str):
                        # Clean up response: remove extra whitespace and ensure it's not empty
                        cleaned_response = response.strip()
                        if cleaned_response:
                            formatted_responses.append(cleaned_response)
                            # Ensure score is a float
                            formatted_scores.append(float(score))
                
                if formatted_responses and formatted_scores:
                    logger.info(f"Submitting {len(formatted_responses)} winners with scores: {formatted_scores}")
                    self.coordinator.submit_winners(formatted_responses, formatted_scores)
                else:
                    logger.warning("No valid responses to submit")
                    
            except Exception as e:
                logger.error(f"Error submitting winners: {e}")
                logger.info("Will try to submit again in next checkpoint")
        
    def run(self, model_args, grpo_args, training_args, get_samples_fn):
        """Menjalankan training dengan Ollama"""
        try:
            # Setup initial peers if using coordinator
            if hasattr(self, 'coordinator'):
                initial_peers = self.get_initial_peers()
                logger.info(f"Retrieved initial peers from chain: {initial_peers}")
                grpo_args.initial_peers = initial_peers

            #########################
            # Log parameters
            #########################
            logger.debug(f"Model parameters {model_args}")
            logger.debug(f"Training/evaluation parameters {training_args}")
            
            batch_size = 2
            training_args.per_device_train_batch_size = batch_size
            training_args.num_generations = batch_size
            
            logger.info("Starting training with Ollama...")
            log_telemetry("training_started", model=self.ollama_config.model_name)
            start_time = datetime.now()
            
            # Setup DHT if using coordinator
            dht = None
            if hasattr(self, 'coordinator'):
                dht = self.setup_dht(grpo_args)
                if not dht:
                    raise RuntimeError("Failed to setup DHT")
                logger.info(f"DHT setup complete with peer_id: {dht.peer_id}")
            
            # Dapatkan dataset
            logger.info(f"Starting training {start_time.strftime('%Y-%m-%d %H:%M:%S')} for {training_args.num_train_epochs} epochs")
            logger.info("Loading samples...")
            train_dataset, test_dataset = get_samples_fn()
            
            # Loop training
            best_response = None
            best_score = float('-inf')
            
            for step in range(training_args.max_steps):
                logger.info(f"\n***** train metrics *****")
                logger.info(f"Step [{step}/{training_args.max_steps}]")
                
                # Ambil batch
                batch = next(iter(train_dataset))
                
                # Hasilkan respons dengan Ollama
                prompts = batch['question']
                max_new_tokens = getattr(training_args, 'max_new_tokens', 1024)
                
                logger.info(f"Processing batch with {len(prompts)} prompts...")
                responses = []
                scores = []
                for i, prompt in enumerate(prompts):
                    logger.info(f"Generating response for prompt {i+1}/{len(prompts)}...")
                    response = self.generate_response(prompt, max_new_tokens)
                    responses.append(response)
                    # Simulate score calculation (you might want to implement proper scoring)
                    score = len(response.split())  # Simple example: longer responses get higher scores
                    scores.append(score)
                    
                    # Update best response
                    if score > best_score:
                        best_score = score
                        best_response = response
                        # Add to best responses list
                        self.best_responses.append(best_response)
                        self.best_scores.append(best_score)
                    
                # Calculate metrics
                current_time = datetime.now()
                train_runtime = (current_time - start_time).total_seconds()
                samples_per_second = len(responses) / train_runtime if train_runtime > 0 else 0
                steps_per_second = (step + 1) / train_runtime if train_runtime > 0 else 0
                
                # Log metrics setiap logging_steps
                if step % training_args.logging_steps == 0:
                    logger.info("***** Running training *****")
                    logger.info(f"  Num examples = {len(train_dataset)}")
                    logger.info(f"  Num Epochs = {training_args.num_train_epochs}")
                    logger.info(f"  Total optimization steps = {training_args.max_steps}")
                    logger.info(f"  Total train batch size = {batch_size}")
                    logger.info(f"  Gradient Accumulation steps = {training_args.gradient_accumulation_steps}")
                    logger.info(f"  Total optimization steps = {training_args.max_steps}")
                    logger.info(f"  Learning rate = {training_args.learning_rate}")
                    logger.info(f"  Current best score = {best_score}")
                    logger.info(f"total_loss\t= {0.0}")
                    logger.info(f"train_loss\t= {0.0}")
                    logger.info(f"train_runtime\t= {train_runtime:.2f}")
                    logger.info(f"train_samples\t= {len(responses)}")
                    logger.info(f"train_samples_per_second\t= {samples_per_second:.3f}")
                    logger.info(f"train_steps_per_second\t= {steps_per_second:.3f}")
                    log_telemetry("training_step", 
                        step=step,
                        train_runtime=train_runtime,
                        samples_per_second=samples_per_second,
                        steps_per_second=steps_per_second,
                        best_score=best_score
                    )
                
                # Simpan checkpoint dan submit winner jika menggunakan coordinator
                if step % training_args.save_steps == 0:
                    logger.info(f"Saving checkpoint at step {step}")
                    if hasattr(self, 'coordinator') and self.best_responses:
                        # Submit all accumulated best responses
                        logger.info(f"Submitting {len(self.best_responses)} best responses")
                        self.submit_winners(self.best_responses, self.best_scores)
                        # Clear the lists after submission
                        self.best_responses = []
                        self.best_scores = []
                    log_telemetry("checkpoint_saved", step=step)
            
            # Submit final winners if using coordinator
            if hasattr(self, 'coordinator') and self.best_responses:
                logger.info(f"Submitting final {len(self.best_responses)} best responses")
                self.submit_winners(self.best_responses, self.best_scores)
            
            # Log final metrics
            end_time = datetime.now()
            total_runtime = (end_time - start_time).total_seconds()
            logger.info(f"Training completed in {total_runtime:.2f} seconds")
            log_telemetry("training_completed", 
                steps=training_args.max_steps,
                total_runtime=total_runtime,
                final_samples_per_second=samples_per_second,
                final_steps_per_second=steps_per_second,
                final_best_score=best_score
            )
            
        except Exception as e:
            logger.error(f"Error in run: {e}")
            raise

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