"""
SFT Training Pipeline Demo - End-to-End

Bu demo, tüm post-training bileşenlerini entegre eder:
1. Model initialization (small GPT)
2. LoRA adapter injection
3. Turkish instruction dataset loading
4. SFT training with instruction masking
5. Checkpoint management
6. Validation
7. Inference testing

Bu, base model'den instruction-following model'e tam dönüşüm örneğidir.

Version: 1.0.0
"""

import torch
import torch.nn as nn
from pathlib import Path
import json
import logging
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.gpt import GPTModel, GPTConfig
from dataclasses import dataclass
from src.tokenizer.sentencepiece_tokenizer import SentencePieceTokenizer
from src.training.lora import (
    LoRAConfig,
    add_lora_to_model,
    print_trainable_parameters,
    save_lora_weights
)
from src.training.sft_trainer import (
    InstructionExample,
    InstructionTemplate,
    InstructionDataset,
    SFTTrainer,
    TEMPLATES
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_sample_corpus(output_path: str):
    """
    Create a small Turkish corpus for tokenizer training.
    
    Args:
        output_path: Path to save corpus
    """
    logger.info("Creating sample Turkish corpus...")
    
    corpus = """
    Merhaba dünya! Bu bir test metnidir.
    Yapay zeka ve makine öğrenmesi çok ilginç konulardır.
    Python programlama dili veri bilimi için popülerdir.
    Türkiye'nin başkenti Ankara'dır.
    İstanbul Boğazı iki kıtayı birleştirir.
    Matematik ve fizik temel bilimlerdir.
    Bilgisayar mühendisliği yazılım geliştirme odaklıdır.
    Doğal dil işleme metinleri analiz eder.
    Derin öğrenme sinir ağları kullanır.
    Veri setleri model eğitimi için gereklidir.
    """ * 20  # Repeat for more data
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(corpus)
    
    logger.info(f"Corpus saved to {output_path}")


def train_or_load_tokenizer(
    corpus_path: str,
    model_path: str,
    vocab_size: int = 500
) -> SentencePieceTokenizer:
    """
    Train or load SentencePiece tokenizer.
    
    Args:
        corpus_path: Path to training corpus
        model_path: Path to tokenizer model
        vocab_size: Vocabulary size
        
    Returns:
        Trained tokenizer
    """
    model_path_obj = Path(model_path)
    
    if model_path_obj.exists():
        logger.info(f"Loading existing tokenizer from {model_path}")
        tokenizer = SentencePieceTokenizer()
        # Load expects model file path, not directory
        tokenizer.load(str(model_path_obj))
        return tokenizer
    
    logger.info("Training new tokenizer...")
    
    # Create output directory
    output_dir = model_path_obj.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Train tokenizer
    tokenizer = SentencePieceTokenizer()
    model_prefix = str(output_dir / "tokenizer")
    
    model_path, stats = tokenizer.train(
        corpus_path=corpus_path,
        vocab_size=vocab_size,
        model_prefix=model_prefix,
        model_type='bpe'
    )
    
    logger.info(f"Tokenizer trained: vocab_size={tokenizer.vocab_size}")
    
    return tokenizer


def load_instruction_dataset(
    train_path: str,
    val_path: str,
    tokenizer: SentencePieceTokenizer,
    template: InstructionTemplate,
    max_length: int = 256
) -> tuple:
    """
    Load instruction datasets.
    
    Args:
        train_path: Path to training data JSON
        val_path: Path to validation data JSON
        tokenizer: Tokenizer instance
        template: Instruction template
        max_length: Maximum sequence length
        
    Returns:
        (train_dataset, val_dataset)
    """
    logger.info("Loading instruction datasets...")
    
    # Load train data
    with open(train_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    train_examples = []
    for ex in train_data['examples']:
        train_examples.append(
            InstructionExample(
                instruction=ex['instruction'],
                response=ex['response']
            )
        )
    
    # Load val data
    with open(val_path, 'r', encoding='utf-8') as f:
        val_data = json.load(f)
    
    val_examples = []
    for ex in val_data['examples']:
        val_examples.append(
            InstructionExample(
                instruction=ex['instruction'],
                response=ex['response']
            )
        )
    
    # Create datasets
    train_dataset = InstructionDataset(
        examples=train_examples,
        tokenizer=tokenizer,
        template=template,
        max_length=max_length,
        mask_instruction=True
    )
    
    val_dataset = InstructionDataset(
        examples=val_examples,
        tokenizer=tokenizer,
        template=template,
        max_length=max_length,
        mask_instruction=True
    )
    
    logger.info(f"Train dataset: {len(train_dataset)} examples")
    logger.info(f"Val dataset: {len(val_dataset)} examples")
    
    return train_dataset, val_dataset


def initialize_model_with_lora(
    vocab_size: int,
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
    device: str = 'cpu'
) -> GPTModel:
    """
    Initialize GPT model with LoRA adapters.
    
    Args:
        vocab_size: Vocabulary size
        lora_rank: LoRA rank
        lora_alpha: LoRA alpha
        device: Device to use
        
    Returns:
        Model with LoRA
    """
    logger.info("Initializing model...")
    
    # Small GPT config for demo
    config = GPTConfig(
        vocab_size=vocab_size,
        max_seq_len=256,
        d_model=128,
        n_layers=4,
        n_heads=4,
        d_ff=512,
        dropout=0.1
    )
    
    # Create model
    model = GPTModel(config)
    model = model.to(device)
    
    # Count original parameters
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Base model parameters: {total_params:,}")
    
    # Add LoRA
    lora_config = LoRAConfig(
        rank=lora_rank,
        alpha=lora_alpha,
        dropout=0.1,
        target_modules=['w_q', 'w_v'],  # Apply to attention Q and V
        merge_weights=False
    )
    
    model = add_lora_to_model(model, lora_config, verbose=True)
    
    # Print parameter statistics
    print_trainable_parameters(model)
    
    return model


def run_training(
    model: GPTModel,
    train_dataset: InstructionDataset,
    val_dataset: InstructionDataset,
    tokenizer: SentencePieceTokenizer,
    output_dir: str,
    max_steps: int = 100,
    batch_size: int = 2,
    learning_rate: float = 5e-4,
    device: str = 'cpu'
):
    """
    Run SFT training.
    
    Args:
        model: Model with LoRA
        train_dataset: Training dataset
        val_dataset: Validation dataset
        tokenizer: Tokenizer
        output_dir: Output directory for checkpoints
        max_steps: Maximum training steps
        batch_size: Batch size
        learning_rate: Learning rate
        device: Device to use
    """
    logger.info("Starting SFT training...")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_steps=max_steps,
        eval_steps=20,
        save_steps=50,
        output_dir=output_dir,
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
        warmup_steps=10
    )
    
    # Train
    trainer.train()
    
    logger.info("Training complete!")
    
    # Save final LoRA weights
    lora_path = output_path / "lora_weights_final.pt"
    save_lora_weights(model, str(lora_path))
    logger.info(f"LoRA weights saved to {lora_path}")


def test_inference(
    model: GPTModel,
    tokenizer: SentencePieceTokenizer,
    template: InstructionTemplate,
    test_instructions: list,
    max_length: int = 100,
    device: str = 'cpu'
):
    """
    Test model inference with sample instructions.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        template: Instruction template
        test_instructions: List of test instructions
        max_length: Maximum generation length
        device: Device
    """
    logger.info("\n" + "="*80)
    logger.info("Testing Inference")
    logger.info("="*80)
    
    model.eval()
    
    for i, instruction in enumerate(test_instructions, 1):
        logger.info(f"\nTest {i}:")
        logger.info(f"Instruction: {instruction}")
        
        # Format with template (only instruction part)
        # Create example with empty response for formatting
        example = InstructionExample(instruction=instruction, response="")
        # formatted_text: str - template-formatted instruction
        formatted_text, _ = template.format(example.instruction, "")
        
        # Tokenize
        # input_ids: List[int] - tokenized instruction
        input_ids = tokenizer.encode(formatted_text)
        # input_tensor: torch.Tensor shape [1, seq_len] - batched input
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
        
        # Generate (simple greedy decoding for demo)
        with torch.no_grad():
            generated_ids = input_ids.copy()
            
            for _ in range(max_length):
                # Forward pass
                # curr_input: torch.Tensor shape [1, curr_len]
                curr_input = torch.tensor([generated_ids], dtype=torch.long).to(device)
                
                # logits: torch.Tensor shape [1, curr_len, vocab_size]
                # attention_weights: list of attention tensors (ignored)
                logits, _ = model(curr_input)
                
                # Get next token
                # next_token_logits: torch.Tensor shape [vocab_size]
                next_token_logits = logits[0, -1, :]
                # next_token: int - predicted token ID
                next_token = torch.argmax(next_token_logits).item()
                
                # Check for EOS
                if next_token == tokenizer.eos_id:
                    break
                
                generated_ids.append(next_token)
                
                # Stop if too long
                if len(generated_ids) > len(input_ids) + max_length:
                    break
            
            # Decode
            # response: str - generated text
            response = tokenizer.decode(generated_ids)
            
            # Extract only the response part (after instruction)
            if "Assistant:" in response:
                response = response.split("Assistant:")[-1].strip()
            
            logger.info(f"Response: {response[:200]}...")
    
    logger.info("\n" + "="*80)


def main():
    """Main execution."""
    print("\n" + "="*80)
    print("SFT Training Pipeline Demo - End-to-End")
    print("="*80 + "\n")
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Paths
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data"
    output_dir = base_dir / "sft_demo_output"
    
    corpus_path = data_dir / "sample_turkish_corpus.txt"
    tokenizer_dir = output_dir / "tokenizer"
    train_data_path = data_dir / "instruction_datasets" / "turkish_train.json"
    val_data_path = data_dir / "instruction_datasets" / "turkish_val.json"
    checkpoint_dir = output_dir / "checkpoints"
    
    # Step 1: Create corpus
    if not corpus_path.exists():
        create_sample_corpus(str(corpus_path))
    
    # Step 2: Train/load tokenizer
    tokenizer = train_or_load_tokenizer(
        corpus_path=str(corpus_path),
        model_path=str(tokenizer_dir / "tokenizer.model"),
        vocab_size=500
    )
    
    # Step 3: Load instruction datasets
    template = TEMPLATES['simple']  # Use simple template
    
    train_dataset, val_dataset = load_instruction_dataset(
        train_path=str(train_data_path),
        val_path=str(val_data_path),
        tokenizer=tokenizer,
        template=template,
        max_length=256
    )
    
    # Step 4: Initialize model with LoRA
    model = initialize_model_with_lora(
        vocab_size=tokenizer.vocab_size,
        lora_rank=8,
        lora_alpha=16.0,
        device=device
    )
    
    # Step 5: Run training
    run_training(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        tokenizer=tokenizer,
        output_dir=str(checkpoint_dir),
        max_steps=100,
        batch_size=4,
        learning_rate=5e-4,
        device=device
    )
    
    # Step 6: Test inference
    test_instructions = [
        "Türkiye'nin başkenti neresidir?",
        "Python'da bir liste nasıl oluşturulur?",
        "Fotosentez nedir?"
    ]
    
    test_inference(
        model=model,
        tokenizer=tokenizer,
        template=template,
        test_instructions=test_instructions,
        max_length=50,
        device=device
    )
    
    print("\n" + "="*80)
    print("Demo Complete!")
    print(f"Outputs saved to: {output_dir}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
