"""
examples/training_with_dashboard.py

Training with Complete Dashboard Demo

Bu demo complete training dashboard'unu gösterir:
- TrainingLogger: Structured metrics storage (JSON/CSV)
- TensorBoard: Real-time visualization
- SampleTracker: Generation quality tracking
- All integrated in Trainer

Kullanım:
    python examples/training_with_dashboard.py
    
    # TensorBoard'u başlatmak için:
    tensorboard --logdir=runs/dashboard_demo/tensorboard
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from torch.utils.data import Dataset, DataLoader
import tempfile
import logging

from src.model.gpt import GPTModel, GPTConfig
from src.tokenizer.bpe import BPETokenizer
from src.training.trainer import Trainer, TrainingConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_sample_data():
    """Create sample text data."""
    sample_texts = [
        "Yapay zeka ve makine öğrenmesi alanında çalışmalar yapılıyor.",
        "Derin öğrenme neural network yapılarını kullanır.",
        "Transformer mimarisi dikkat mekanizması içerir.",
        "Model eğitimi gradient descent ile yapılır.",
        "Loss function modelin hatasını ölçer.",
        "Tokenizer metni token'lara ayırır.",
        "Bir varmış bir yokmuş, evvel zaman içinde.",
        "Bilgisayar bilimi ve yapay zeka.",
        "Optimization süreci parametreleri günceller.",
        "Natural language processing çalışmaları devam ediyor.",
    ] * 20  # Repeat for more data
    
    return "\n".join(sample_texts)


class SimpleTextDataset(Dataset):
    """Simple dataset for demo."""
    
    def __init__(self, tokens, seq_len=64):
        self.tokens = tokens
        self.seq_len = seq_len
    
    def __len__(self):
        return max(1, len(self.tokens) // self.seq_len)
    
    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = min(start + self.seq_len, len(self.tokens))
        
        chunk = self.tokens[start:end]
        if len(chunk) < self.seq_len:
            chunk = chunk + [0] * (self.seq_len - len(chunk))
        
        return {
            'input_ids': torch.tensor(chunk, dtype=torch.long),
            'labels': torch.tensor(chunk, dtype=torch.long)
        }


def main():
    """Run training with dashboard demo."""
    
    print("=" * 80)
    print("Training with Complete Dashboard Demo")
    print("=" * 80)
    print()
    
    # Create workspace
    workspace = Path("runs/dashboard_demo")
    workspace.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Workspace: {workspace}")
    
    # =================================================================
    # Step 1: Data & Tokenizer
    # =================================================================
    print("\n" + "=" * 80)
    print("Step 1: Data & Tokenizer")
    print("=" * 80)
    
    corpus_text = create_sample_data()
    logger.info(f"Corpus: {len(corpus_text)} characters")
    
    # Train tokenizer
    tokenizer = BPETokenizer(vocab_size=500)
    tokenizer.train([corpus_text])
    logger.info(f"Tokenizer trained: {len(tokenizer.vocab)} vocab")
    
    # Tokenize corpus
    all_tokens = tokenizer.encode(corpus_text)
    logger.info(f"Tokenized: {len(all_tokens)} tokens")
    
    # Create dataset
    dataset = SimpleTextDataset(all_tokens, seq_len=64)
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
    logger.info(f"Dataset: {len(dataset)} samples")
    
    # =================================================================
    # Step 2: Model
    # =================================================================
    print("\n" + "=" * 80)
    print("Step 2: Model Initialization")
    print("=" * 80)
    
    model_config = GPTConfig(
        vocab_size=len(tokenizer.vocab),
        max_seq_len=64,
        d_model=128,
        n_layers=4,
        n_heads=4,
        d_ff=512,
        dropout=0.1
    )
    
    model = GPTModel(model_config)
    logger.info(f"Model: {model.get_num_params():,} parameters")
    
    # =================================================================
    # Step 3: Training Config
    # =================================================================
    print("\n" + "=" * 80)
    print("Step 3: Training Configuration")
    print("=" * 80)
    
    train_config = TrainingConfig(
        learning_rate=3e-4,
        max_steps=100,  # Short training for demo
        warmup_steps=10,
        batch_size=4,
        gradient_accumulation_steps=2,
        log_interval=10,
        eval_interval=50,
        checkpoint_interval=25,
        max_checkpoints=2
    )
    
    logger.info(f"Training config:")
    logger.info(f"  Max steps: {train_config.max_steps}")
    logger.info(f"  Learning rate: {train_config.learning_rate}")
    logger.info(f"  Batch size: {train_config.batch_size}")
    logger.info(f"  Gradient accumulation: {train_config.gradient_accumulation_steps}")
    
    # =================================================================
    # Step 4: Trainer with Dashboard
    # =================================================================
    print("\n" + "=" * 80)
    print("Step 4: Trainer with Complete Dashboard")
    print("=" * 80)
    
    # Test prompts for sample tracking
    test_prompts = [
        "Yapay zeka",
        "Model eğitimi",
        "Bir varmış"
    ]
    
    trainer = Trainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=None,
        config=train_config,
        checkpoint_dir=str(workspace / "checkpoints"),
        device='cpu',
        # Dashboard features
        enable_tensorboard=True,
        enable_sample_tracking=True,
        test_prompts=test_prompts,
        tokenizer=tokenizer
    )
    
    logger.info("✓ Trainer initialized with:")
    logger.info(f"  - TrainingLogger (JSON/CSV)")
    logger.info(f"  - TensorBoard (visualization)")
    logger.info(f"  - SampleTracker ({len(test_prompts)} prompts)")
    
    # =================================================================
    # Step 5: Training
    # =================================================================
    print("\n" + "=" * 80)
    print("Step 5: Training with Dashboard")
    print("=" * 80)
    
    logger.info("Starting training...")
    logger.info(f"  View TensorBoard: tensorboard --logdir={workspace}/checkpoints/tensorboard")
    print()
    
    trainer.train()
    
    # =================================================================
    # Step 6: Results
    # =================================================================
    print("\n" + "=" * 80)
    print("Step 6: Dashboard Outputs")
    print("=" * 80)
    
    # List generated files
    checkpoint_dir = workspace / "checkpoints"
    
    print("\n📁 Generated Files:")
    for file in sorted(checkpoint_dir.rglob("*")):
        if file.is_file():
            size_kb = file.stat().st_size / 1024
            rel_path = file.relative_to(workspace)
            print(f"  {rel_path}: {size_kb:.1f} KB")
    
    # Instructions
    print("\n📊 View Dashboard:")
    print(f"  1. TensorBoard: tensorboard --logdir={checkpoint_dir}/tensorboard")
    print(f"  2. Metrics JSON: {checkpoint_dir}/training_run_metrics.json")
    print(f"  3. Metrics CSV: {checkpoint_dir}/training_run_metrics.csv")
    print(f"  4. Samples: {checkpoint_dir}/samples/generation_samples.json")
    
    print("\n" + "=" * 80)
    print("✅ Training with Dashboard Demo Completed!")
    print("=" * 80)
    
    # Sample some generated text
    print("\n📝 Latest Generation Samples:")
    if trainer.sample_tracker and trainer.sample_tracker.history:
        latest_step = max(s.step for s in trainer.sample_tracker.history)
        latest_samples = [s for s in trainer.sample_tracker.history if s.step == latest_step]
        
        for sample in latest_samples[:3]:
            print(f"\n  Prompt: '{sample.prompt}'")
            print(f"  Generated: '{sample.generated[:100]}...'")
            print(f"  Metrics:")
            for name, value in sample.metrics.items():
                print(f"    {name}: {value:.3f}")


if __name__ == "__main__":
    main()
