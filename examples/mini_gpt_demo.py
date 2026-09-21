"""
examples/mini_gpt_demo.py

Mini GPT End-to-End Demo

Bu demo tüm pipeline'ı gösterir:
1. Veri ingestion (text dosyalarından)
2. Dataset oluşturma
3. Tokenizer training
4. Model initialization
5. Training
6. Inference (text generation)

Demo küçük bir model ile hızlı bir eğitim yapar ve
sonuçları gösterir. Production-ready değil, eğitim amaçlıdır.

Kullanım:
    python examples/mini_gpt_demo.py
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import tempfile
import shutil
import logging
from datetime import datetime

# Import all components
from src.ingestion.text import TextParser
from src.tokenizer.bpe import BPETokenizer
from src.model.gpt import GPTModel, GPTConfig
from src.training.trainer import Trainer, TrainingConfig
from src.inference.pipeline import InferencePipeline
from src.inference.config import GenerationConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_sample_data(output_dir: Path, num_samples: int = 100) -> Path:
    """
    Create sample text data for training.
    
    Args:
        output_dir: Directory to save data
        num_samples: Number of text samples to generate
    
    Returns:
        Path to data file
    """
    logger.info(f"Creating {num_samples} sample texts...")
    
    # Sample Turkish texts (could be any text corpus)
    sample_texts = [
        "Bir varmış bir yokmuş, evvel zaman içinde.",
        "Kalbûr saman içinde, deve tellâl iken, ben babamın beşiğini tıngır mıngır sallar iken.",
        "Bilgisayar bilimi ve yapay zeka alanında çalışmalar yapılıyor.",
        "Makine öğrenmesi modelleri veri üzerinde eğitilir.",
        "Derin öğrenme neural network yapılarını kullanır.",
        "Transformer mimarisi dikkat mekanizması içerir.",
        "Tokenizer metni token'lara ayırır.",
        "Model eğitimi gradient descent ile yapılır.",
        "Loss function modelin hatasını ölçer.",
        "Optimization süreci parametreleri günceller.",
    ]
    
    data_file = output_dir / "sample_data.txt"
    
    with open(data_file, 'w', encoding='utf-8') as f:
        for _ in range(num_samples):
            for text in sample_texts:
                f.write(text + "\n")
    
    logger.info(f"Sample data created: {data_file}")
    return data_file


def main():
    """Run complete Mini GPT demo."""
    print("=" * 80)
    print("Mini GPT End-to-End Demo")
    print("=" * 80)
    print()
    
    # Create temporary workspace
    workspace = Path(tempfile.mkdtemp(prefix="minigpt_demo_"))
    logger.info(f"Workspace: {workspace}")
    
    try:
        # =================================================================
        # Step 1: Data Preparation
        # =================================================================
        print("\n" + "=" * 80)
        print("Step 1: Data Preparation")
        print("=" * 80)
        
        data_file = create_sample_data(workspace, num_samples=50)
        
        # Read data
        with open(data_file, 'r', encoding='utf-8') as f:
            corpus_text = f.read()
        
        print(f"✓ Created sample data: {len(corpus_text)} characters")
        
        # =================================================================
        # Step 2: Tokenizer Training
        # =================================================================
        print("\n" + "=" * 80)
        print("Step 2: Tokenizer Training")
        print("=" * 80)
        
        # Train BPE tokenizer
        tokenizer = BPETokenizer(vocab_size=500)
        tokenizer.train([corpus_text])
        
        tokenizer_file = workspace / "tokenizer.json"
        tokenizer.save_vocab(tokenizer_file)
        
        print(f"✓ Tokenizer trained: vocab_size={len(tokenizer.vocab)}")
        print(f"✓ Saved to: {tokenizer_file}")
        
        # Test tokenizer
        test_text = "Yapay zeka modeli eğitimi"
        tokens = tokenizer.encode(test_text)
        decoded = tokenizer.decode(tokens)
        print(f"\nTokenizer test:")
        print(f"  Original: '{test_text}'")
        print(f"  Tokens: {tokens[:10]}... ({len(tokens)} tokens)")
        print(f"  Decoded: '{decoded}'")
        
        # =================================================================
        # Step 3: Dataset Preparation
        # =================================================================
        print("\n" + "=" * 80)
        print("Step 3: Dataset Preparation")
        print("=" * 80)
        
        # Tokenize all data
        all_tokens = tokenizer.encode(corpus_text)
        print(f"✓ Tokenized corpus: {len(all_tokens)} tokens")
        
        # Create simple dataset (no complex dataset compiler for demo)
        from torch.utils.data import Dataset, DataLoader
        
        class SimpleTextDataset(Dataset):
            def __init__(self, tokens, seq_len=64):
                self.tokens = tokens
                self.seq_len = seq_len
            
            def __len__(self):
                return max(1, len(self.tokens) // self.seq_len)
            
            def __getitem__(self, idx):
                start = idx * self.seq_len
                end = min(start + self.seq_len, len(self.tokens))
                
                # Pad if necessary
                chunk = self.tokens[start:end]
                if len(chunk) < self.seq_len:
                    chunk = chunk + [0] * (self.seq_len - len(chunk))
                
                return torch.tensor(chunk, dtype=torch.long)
        
        dataset = SimpleTextDataset(all_tokens, seq_len=32)
        train_loader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            collate_fn=lambda x: {'input_ids': torch.stack(x)}
        )
        
        print(f"✓ Dataset created: {len(dataset)} samples, seq_len=32")
        
        # =================================================================
        # Step 4: Model Initialization
        # =================================================================
        print("\n" + "=" * 80)
        print("Step 4: Model Initialization")
        print("=" * 80)
        
        model_config = GPTConfig(
            vocab_size=len(tokenizer.vocab),
            max_seq_len=32,
            d_model=128,
            n_layers=4,
            n_heads=4,
            d_ff=512,
            dropout=0.1
        )
        
        model = GPTModel(model_config)
        
        print(f"✓ Model created:")
        print(f"  Parameters: {model.get_num_params():,}")
        print(f"  Layers: {model_config.n_layers}")
        print(f"  d_model: {model_config.d_model}")
        print(f"  Heads: {model_config.n_heads}")
        
        # =================================================================
        # Step 5: Training
        # =================================================================
        print("\n" + "=" * 80)
        print("Step 5: Model Training")
        print("=" * 80)
        
        train_config = TrainingConfig(
            learning_rate=3e-4,
            max_steps=100,  # Short training for demo
            warmup_steps=10,
            batch_size=4,
            gradient_accumulation_steps=2,
            log_interval=20,
            eval_interval=50,
            checkpoint_interval=50
        )
        
        checkpoint_dir = workspace / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        
        trainer = Trainer(
            model=model,
            train_dataloader=train_loader,
            val_dataloader=None,  # No validation for demo
            config=train_config,
            checkpoint_dir=str(checkpoint_dir),
            device='cpu'
        )
        
        print("Starting training...")
        print(f"  Max steps: {train_config.max_steps}")
        print(f"  Batch size: {train_config.batch_size}")
        print(f"  Gradient accumulation: {train_config.gradient_accumulation_steps}")
        print()
        
        trainer.train()
        
        # Get final metrics
        if trainer.train_metrics:
            final_metrics = trainer.train_metrics[-1]
            print(f"\n✓ Training completed!")
            print(f"  Final loss: {final_metrics.get('loss', 'N/A'):.4f}")
            print(f"  Total steps: {trainer.global_step}")
        
        # =================================================================
        # Step 6: Save Model
        # =================================================================
        print("\n" + "=" * 80)
        print("Step 6: Model Saving")
        print("=" * 80)
        
        model_path = workspace / "model.pt"
        
        # Save with config
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': {
                'vocab_size': model_config.vocab_size,
                'max_seq_len': model_config.max_seq_len,
                'd_model': model_config.d_model,
                'n_layers': model_config.n_layers,
                'n_heads': model_config.n_heads,
                'd_ff': model_config.d_ff,
                'dropout': model_config.dropout
            }
        }, model_path)
        
        print(f"✓ Model saved to: {model_path}")
        
        # =================================================================
        # Step 7: Inference Pipeline
        # =================================================================
        print("\n" + "=" * 80)
        print("Step 7: Inference & Text Generation")
        print("=" * 80)
        
        # Load pipeline
        # Note: Need to create tokenizer first since from_pretrained expects specific format
        loaded_tokenizer = BPETokenizer(vocab_size=len(tokenizer.vocab))
        loaded_tokenizer.load_vocab(tokenizer_file)
        
        # Create pipeline with loaded components
        loaded_model = GPTModel(model_config)
        checkpoint = torch.load(model_path, map_location='cpu')
        loaded_model.load_state_dict(checkpoint['model_state_dict'])
        
        pipeline = InferencePipeline(
            model=loaded_model,
            tokenizer=loaded_tokenizer,
            device='cpu'
        )
        
        print(f"✓ Pipeline loaded")
        
        # Test prompts
        test_prompts = [
            "Yapay zeka",
            "Model eğitimi",
            "Bir varmış bir"
        ]
        
        print("\nGenerating text samples...\n")
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"{i}. Prompt: '{prompt}'")
            
            # Generate with different strategies
            configs = [
                ("Greedy", GenerationConfig.greedy(max_new_tokens=15)),
                ("Sampling", GenerationConfig.sampling(temperature=0.8, max_new_tokens=15)),
                ("Creative", GenerationConfig.creative(temperature=1.0, max_new_tokens=15))
            ]
            
            for strategy_name, config in configs:
                output = pipeline(prompt, config=config)
                print(f"   [{strategy_name}]: {output}")
            
            print()
        
        # =================================================================
        # Step 8: Summary
        # =================================================================
        print("=" * 80)
        print("Demo Summary")
        print("=" * 80)
        print()
        print("✓ Data ingestion: Sample text created")
        print(f"✓ Tokenizer: {len(tokenizer.vocab)} vocab")
        print(f"✓ Dataset: {len(dataset)} samples")
        print(f"✓ Model: {model.get_num_params():,} parameters")
        print(f"✓ Training: {trainer.global_step} steps completed")
        print(f"✓ Inference: {len(test_prompts)} prompts tested")
        print()
        print(f"Workspace: {workspace}")
        print("Files:")
        for f in workspace.rglob("*"):
            if f.is_file():
                size_kb = f.stat().st_size / 1024
                print(f"  {f.relative_to(workspace)}: {size_kb:.1f} KB")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        raise
    
    finally:
        # Cleanup (optional)
        # Uncomment to auto-delete workspace
        # shutil.rmtree(workspace)
        pass
    
    print("\n" + "=" * 80)
    print("✅ Mini GPT Demo Completed Successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
