"""MLX fine-tuning implementation optimized for Apple Silicon."""

import os
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Optional, List, Any

from ..utils.logging import get_logger
from ..utils.config import MLXConfig

logger = get_logger(__name__)

# Check for MLX availability
try:
    import mlx.core as mx
    import mlx.nn as nn
    HAS_MLX = True
    logger.info("MLX available for Apple Silicon optimization")
except ImportError:
    HAS_MLX = False
    mx = None
    nn = None
    logger.warning("MLX not available - install with: pip install mlx>=0.12.0")


class MLXFineTuner:
    """MLX fine-tuning implementation for Apple Silicon."""
    
    def __init__(self, config: Optional[MLXConfig] = None):
        """Initialize MLX fine-tuner.
        
        Args:
            config: MLX configuration
        """
        self.config = config or MLXConfig()
        
        if not HAS_MLX:
            logger.error("MLX not available. Install with: pip install mlx>=0.12.0 mlx-lm>=0.8.0")
            return
        
        # Auto-optimize for M3 Max if enabled
        if self.config.auto_optimize_m3:
            self._optimize_for_m3_max()
    
    def _optimize_for_m3_max(self):
        """Optimize settings for M3 Max hardware."""
        if self.config.model_size in self.config.m3_max_settings:
            settings = self.config.m3_max_settings[self.config.model_size]
            self.config.batch_size = settings["batch_size"]
            self.config.learning_rate = settings["learning_rate"]
            self.config.num_iters = settings["num_iters"]
            
            logger.info(f"Optimized for M3 Max with {self.config.model_size} settings:")
            logger.info(f"  Batch size: {self.config.batch_size}")
            logger.info(f"  Learning rate: {self.config.learning_rate}")
            logger.info(f"  Iterations: {self.config.num_iters}")
    
    def finetune(self, dataset_path: str, output_dir: str) -> Optional[str]:
        """Run MLX fine-tuning on the dataset.
        
        Args:
            dataset_path: Path to training dataset
            output_dir: Output directory for fine-tuned model
            
        Returns:
            Path to fine-tuned model or None if failed
        """
        if not HAS_MLX:
            logger.error("MLX not available")
            return None
        
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Prepare dataset for MLX
            formatted_dataset = self._format_dataset_for_mlx(dataset_path)
            if not formatted_dataset:
                logger.error("Failed to format dataset for MLX")
                return None
            
            # Run LoRA fine-tuning
            adapter_path = self._run_lora_training(formatted_dataset, output_path)
            if not adapter_path:
                logger.error("LoRA training failed")
                return None
            
            # Fuse model if requested
            if self.config.fuse_model:
                fused_path = self._fuse_model(adapter_path, output_path)
                if fused_path:
                    logger.info(f"✅ Fine-tuned model ready: {fused_path}")
                    return fused_path
                else:
                    logger.warning("Model fusing failed, returning adapter path")
                    return adapter_path
            
            logger.info(f"✅ LoRA adapter ready: {adapter_path}")
            return adapter_path
            
        except Exception as e:
            logger.error(f"MLX fine-tuning failed: {e}")
            return None
    
    def _format_dataset_for_mlx(self, dataset_path: str) -> Optional[str]:
        """Format dataset for MLX training.
        
        Args:
            dataset_path: Path to input dataset
            
        Returns:
            Path to formatted dataset or None if failed
        """
        try:
            # Load dataset
            with open(dataset_path, 'r', encoding='utf-8') as f:
                if dataset_path.endswith('.jsonl'):
                    data = [json.loads(line) for line in f]
                else:
                    data = json.load(f)
            
            # Format for MLX training
            formatted_data = []
            
            if isinstance(data, dict) and 'training_data' in data:
                # Instruction-response format
                for item in data['training_data']:
                    formatted_item = {
                        "text": f"<|user|>\n{item['instruction']}\n{item.get('input', '')}\n<|assistant|>\n{item['output']}"
                    }
                    formatted_data.append(formatted_item)
            
            elif isinstance(data, dict) and 'conversations' in data:
                # Conversation format
                for conv in data['conversations']:
                    if 'messages' in conv:
                        text_parts = []
                        for msg in conv['messages']:
                            role = msg['role']
                            content = msg['content']
                            text_parts.append(f"<|{role}|>\n{content}")
                        
                        formatted_item = {"text": "\n".join(text_parts)}
                        formatted_data.append(formatted_item)
            
            elif isinstance(data, list):
                # JSONL format
                for item in data:
                    if 'prompt' in item and 'completion' in item:
                        formatted_item = {
                            "text": f"<|user|>\n{item['prompt']}\n<|assistant|>\n{item['completion']}"
                        }
                        formatted_data.append(formatted_item)
                    elif 'text' in item:
                        formatted_data.append({"text": item['text']})
            
            # Save formatted dataset
            formatted_path = Path(dataset_path).parent / "mlx_formatted_dataset.jsonl"
            with open(formatted_path, 'w', encoding='utf-8') as f:
                for item in formatted_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
            logger.info(f"Formatted {len(formatted_data)} examples for MLX training")
            return str(formatted_path)
            
        except Exception as e:
            logger.error(f"Error formatting dataset: {e}")
            return None
    
    def _run_lora_training(self, dataset_path: str, output_dir: Path) -> Optional[str]:
        """Run LoRA training using MLX.
        
        Args:
            dataset_path: Path to formatted dataset
            output_dir: Output directory
            
        Returns:
            Path to trained adapter or None if failed
        """
        try:
            adapter_dir = output_dir / self.config.adapter_dir
            adapter_dir.mkdir(parents=True, exist_ok=True)
            
            # Build MLX training command
            cmd = [
                "python", "-m", "mlx_lm.lora",
                "--model", self.config.model_name,
                "--train",
                "--data", dataset_path,
                "--batch-size", str(self.config.batch_size),
                "--lora-layers", str(self.config.num_layers),
                "--lora-rank", str(self.config.lora_rank),
                "--lora-alpha", str(self.config.lora_alpha),
                "--lora-dropout", str(self.config.lora_dropout),
                "--learning-rate", str(self.config.learning_rate),
                "--iters", str(self.config.num_iters),
                "--steps-per-eval", str(self.config.steps_per_eval),
                "--save-every", str(self.config.steps_per_eval),
                "--adapter-path", str(adapter_dir),
                "--val-batches", "10"
            ]
            
            logger.info("Starting LoRA training with MLX...")
            logger.info(f"Command: {' '.join(cmd)}")
            
            # Run training
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.timeouts.mlx_training_timeout
            )
            
            if result.returncode == 0:
                logger.info("✅ LoRA training completed successfully")
                return str(adapter_dir)
            else:
                logger.error(f"LoRA training failed:")
                logger.error(f"stdout: {result.stdout}")
                logger.error(f"stderr: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            logger.error("LoRA training timed out")
            return None
        except Exception as e:
            logger.error(f"Error during LoRA training: {e}")
            return None
    
    def _fuse_model(self, adapter_path: str, output_dir: Path) -> Optional[str]:
        """Fuse LoRA adapter with base model.
        
        Args:
            adapter_path: Path to LoRA adapter
            output_dir: Output directory
            
        Returns:
            Path to fused model or None if failed
        """
        try:
            fused_dir = output_dir / self.config.fused_model_dir
            fused_dir.mkdir(parents=True, exist_ok=True)
            
            # Build fuse command
            cmd = [
                "python", "-m", "mlx_lm.fuse",
                "--model", self.config.model_name,
                "--adapter-path", adapter_path,
                "--save-path", str(fused_dir)
            ]
            
            logger.info("Fusing LoRA adapter with base model...")
            logger.info(f"Command: {' '.join(cmd)}")
            
            # Run fusing
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.timeouts.mlx_model_fusing_timeout
            )
            
            if result.returncode == 0:
                logger.info("✅ Model fusing completed successfully")
                return str(fused_dir)
            else:
                logger.error(f"Model fusing failed:")
                logger.error(f"stdout: {result.stdout}")
                logger.error(f"stderr: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            logger.error("Model fusing timed out")
            return None
        except Exception as e:
            logger.error(f"Error during model fusing: {e}")
            return None
    
    def generate_text(
        self, 
        model_path: str, 
        prompt: str, 
        max_tokens: int = 100,
        temperature: float = 0.7
    ) -> Optional[str]:
        """Generate text using the fine-tuned model.
        
        Args:
            model_path: Path to fine-tuned model
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated text or None if failed
        """
        if not HAS_MLX:
            logger.error("MLX not available")
            return None
        
        try:
            # Build generation command
            cmd = [
                "python", "-m", "mlx_lm.generate",
                "--model", model_path,
                "--prompt", prompt,
                "--max-tokens", str(max_tokens),
                "--temp", str(temperature)
            ]
            
            # Run generation
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.timeouts.mlx_generation_timeout
            )
            
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                logger.error(f"Text generation failed: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            logger.error("Text generation timed out")
            return None
        except Exception as e:
            logger.error(f"Error during text generation: {e}")
            return None
    
    def interactive_chat(self, model_path: str) -> None:
        """Start interactive chat with the fine-tuned model.
        
        Args:
            model_path: Path to fine-tuned model
        """
        if not HAS_MLX:
            logger.error("MLX not available")
            return
        
        try:
            print("\n🤖 Starting interactive chat with fine-tuned model...")
            print("Type 'quit' or 'exit' to end the session.\n")
            
            while True:
                user_input = input("\n👤 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit']:
                    print("👋 Goodbye!")
                    break
                
                if not user_input:
                    continue
                
                print("🤖 Assistant: ", end="", flush=True)
                
                # Generate response
                response = self.generate_text(
                    model_path=model_path,
                    prompt=f"<|user|>\n{user_input}\n<|assistant|>\n",
                    max_tokens=200,
                    temperature=0.7
                )
                
                if response:
                    # Extract assistant response (remove prompt echo)
                    assistant_part = response.split("<|assistant|>")[-1].strip()
                    print(assistant_part)
                else:
                    print("Sorry, I couldn't generate a response.")
                    
        except KeyboardInterrupt:
            print("\n👋 Chat session ended.")
        except Exception as e:
            logger.error(f"Error during interactive chat: {e}")
    
    def validate_model(self, model_path: str, test_prompts: List[str]) -> Dict[str, Any]:
        """Validate the fine-tuned model with test prompts.
        
        Args:
            model_path: Path to fine-tuned model
            test_prompts: List of test prompts
            
        Returns:
            Validation results
        """
        if not HAS_MLX:
            logger.error("MLX not available")
            return {"error": "MLX not available"}
        
        results = {
            "model_path": model_path,
            "test_results": [],
            "avg_response_length": 0,
            "successful_generations": 0,
            "total_prompts": len(test_prompts)
        }
        
        response_lengths = []
        
        for i, prompt in enumerate(test_prompts):
            logger.info(f"Testing prompt {i+1}/{len(test_prompts)}")
            
            response = self.generate_text(
                model_path=model_path,
                prompt=prompt,
                max_tokens=150
            )
            
            test_result = {
                "prompt": prompt,
                "response": response,
                "success": response is not None,
                "response_length": len(response.split()) if response else 0
            }
            
            results["test_results"].append(test_result)
            
            if response:
                results["successful_generations"] += 1
                response_lengths.append(len(response.split()))
        
        # Calculate statistics
        if response_lengths:
            results["avg_response_length"] = sum(response_lengths) / len(response_lengths)
        
        results["success_rate"] = results["successful_generations"] / results["total_prompts"]
        
        logger.info(f"Validation complete: {results['success_rate']:.2%} success rate")
        return results
    
    def is_available(self) -> bool:
        """Check if MLX is available and functional.
        
        Returns:
            True if MLX is available, False otherwise
        """
        return HAS_MLX