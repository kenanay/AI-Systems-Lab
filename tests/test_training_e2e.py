"""
tests/test_training_e2e.py

End-to-End Training Integration Test

Bu test, complete training pipeline'ı gerçek data ile test eder:
- Dataset loading (compiled datasets)
- Tokenizer loading
- DataLoader setup
- Model initialization
- Training loop execution
- Checkpoint save/load
- Validation metrics

Test, tüm component'lerin birlikte çalıştığını verify eder.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import pytest
from pathlib import Path
import tempfile
import shutil

from src.model.gpt import GPTModel, GPTConfig
from src.training.trainer import Trainer, TrainingConfig
from src.training.checkpoint import CheckpointManager


class TestE2ETraining:
    """End-to-end training integration tests."""
    
    def test_training_with_dummy_data(self):
        """Test complete training pipeline with dummy data."""
        from torch.utils.data import TensorDataset, DataLoader
        
        # Create small model
        model_config = GPTConfig(
            vocab_size=100,
            max_seq_len=64,
            d_model=128,
            n_layers=2,
            n_heads=4,
            d_ff=512,
            dropout=0.1
        )
        
        model = GPTModel(model_config)
        
        # Create dummy dataset
        num_samples = 50
        seq_len = 32
        
        train_data = torch.randint(0, model_config.vocab_size, (num_samples, seq_len))
        val_data = torch.randint(0, model_config.vocab_size, (10, seq_len))
        
        train_dataset = TensorDataset(train_data)
        val_dataset = TensorDataset(val_data)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=4,
            shuffle=True,
            collate_fn=lambda x: {'input_ids': torch.stack([item[0] for item in x])}
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=4,
            shuffle=False,
            collate_fn=lambda x: {'input_ids': torch.stack([item[0] for item in x])}
        )
        
        # Create training config
        train_config = TrainingConfig(
            learning_rate=1e-3,
            max_steps=20,
            warmup_steps=5,
            batch_size=4,
            gradient_accumulation_steps=1,
            log_interval=5,
            eval_interval=10,
            checkpoint_interval=10,
            eval_steps=3
        )
        
        # Create temporary directory for checkpoints
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Create trainer
            trainer = Trainer(
                model=model,
                train_dataloader=train_loader,
                val_dataloader=val_loader,
                config=train_config,
                checkpoint_dir=temp_dir,
                device='cpu'
            )
            
            # Train
            trainer.train()
            
            # Verify training completed
            assert trainer.global_step == train_config.max_steps
            
            # Verify checkpoints were saved
            checkpoints = list(Path(temp_dir).glob('checkpoint_*.pt'))
            assert len(checkpoints) > 0
            
            # Verify best model was saved
            best_model_path = Path(temp_dir) / 'best_model.pt'
            assert best_model_path.exists()
            
            # Verify metrics were tracked
            assert len(trainer.train_metrics) > 0
            assert len(trainer.val_metrics) > 0
            
            # Verify loss decreased
            first_loss = trainer.train_metrics[0]['loss']
            last_loss = trainer.train_metrics[-1]['loss']
            assert last_loss < first_loss, "Loss should decrease during training"
            
        finally:
            shutil.rmtree(temp_dir)
    
    def test_checkpoint_resume(self):
        """Test training resume from checkpoint."""
        from torch.utils.data import TensorDataset, DataLoader
        
        model_config = GPTConfig(
            vocab_size=100,
            max_seq_len=64,
            d_model=128,
            n_layers=2,
            n_heads=4
        )
        
        # Dummy data
        train_data = torch.randint(0, 100, (50, 32))
        train_dataset = TensorDataset(train_data)
        train_loader = DataLoader(
            train_dataset,
            batch_size=4,
            collate_fn=lambda x: {'input_ids': torch.stack([item[0] for item in x])}
        )
        
        train_config = TrainingConfig(
            max_steps=20,
            checkpoint_interval=10
        )
        
        temp_dir = tempfile.mkdtemp()
        
        try:
            # First training session
            model1 = GPTModel(model_config)
            trainer1 = Trainer(
                model=model1,
                train_dataloader=train_loader,
                config=train_config,
                checkpoint_dir=temp_dir,
                device='cpu'
            )
            
            trainer1.train()
            
            # Get checkpoint path
            checkpoints = list(Path(temp_dir).glob('checkpoint_*.pt'))
            assert len(checkpoints) > 0
            
            checkpoint_path = str(checkpoints[0])
            
            # Second training session (resume)
            model2 = GPTModel(model_config)
            trainer2 = Trainer(
                model=model2,
                train_dataloader=train_loader,
                config=train_config,
                checkpoint_dir=temp_dir,
                device='cpu',
                resume_from=checkpoint_path
            )
            
            # Verify state was restored
            assert trainer2.global_step > 0
            
        finally:
            shutil.rmtree(temp_dir)
    
    def test_validation_metrics(self):
        """Test validation metrics computation."""
        from torch.utils.data import TensorDataset, DataLoader
        
        model_config = GPTConfig(
            vocab_size=100,
            max_seq_len=32,
            d_model=64,
            n_layers=2,
            n_heads=4
        )
        
        model = GPTModel(model_config)
        
        # Create validation data
        val_data = torch.randint(0, 100, (20, 16))
        val_dataset = TensorDataset(val_data)
        val_loader = DataLoader(
            val_dataset,
            batch_size=4,
            collate_fn=lambda x: {'input_ids': torch.stack([item[0] for item in x])}
        )
        
        train_config = TrainingConfig(eval_steps=5)
        
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Create trainer
            trainer = Trainer(
                model=model,
                train_dataloader=val_loader,  # Use same for train
                val_dataloader=val_loader,
                config=train_config,
                checkpoint_dir=temp_dir,
                device='cpu'
            )
            
            # Run evaluation
            metrics = trainer.evaluate()
            
            # Verify metrics exist
            assert 'loss' in metrics
            assert 'perplexity' in metrics
            assert 'accuracy' in metrics
            
            # Verify metrics are valid
            assert metrics['loss'] >= 0
            assert metrics['perplexity'] >= 1.0
            assert 0 <= metrics['accuracy'] <= 1.0
            
        finally:
            shutil.rmtree(temp_dir)
    
    def test_gradient_accumulation(self):
        """Test gradient accumulation works correctly."""
        from torch.utils.data import TensorDataset, DataLoader
        
        model_config = GPTConfig(
            vocab_size=50,
            max_seq_len=32,
            d_model=64,
            n_layers=2,
            n_heads=4
        )
        
        train_data = torch.randint(0, 50, (40, 16))
        train_dataset = TensorDataset(train_data)
        train_loader = DataLoader(
            train_dataset,
            batch_size=2,
            collate_fn=lambda x: {'input_ids': torch.stack([item[0] for item in x])}
        )
        
        # Test with gradient accumulation = 4
        train_config = TrainingConfig(
            max_steps=10,
            batch_size=2,
            gradient_accumulation_steps=4,
            log_interval=5
        )
        
        temp_dir = tempfile.mkdtemp()
        
        try:
            model = GPTModel(model_config)
            trainer = Trainer(
                model=model,
                train_dataloader=train_loader,
                config=train_config,
                checkpoint_dir=temp_dir,
                device='cpu'
            )
            
            trainer.train()
            
            # Verify training completed
            assert trainer.global_step == train_config.max_steps
            
            # Effective batch size = 2 * 4 = 8
            # This should work without errors
            
        finally:
            shutil.rmtree(temp_dir)
    
    def test_config_validation(self):
        """Test training config validation."""
        
        # Valid config
        config = TrainingConfig(
            learning_rate=1e-3,
            max_steps=100
        )
        assert config.learning_rate == 1e-3
        
        # Invalid: negative learning rate
        with pytest.raises(AssertionError):
            TrainingConfig(learning_rate=-1e-3)
        
        # Invalid: zero max_steps
        with pytest.raises(AssertionError):
            TrainingConfig(max_steps=0)
        
        # Invalid: label smoothing >= 1
        with pytest.raises(AssertionError):
            TrainingConfig(label_smoothing=1.0)
    
    @pytest.mark.skipif(
        not Path('datasets/compiled_datasets').exists(),
        reason="Real dataset not available"
    )
    def test_with_real_dataset(self):
        """
        Test training with real compiled dataset.
        
        This test only runs if compiled datasets are available.
        """
        import pyarrow.parquet as pq
        from torch.utils.data import Dataset, DataLoader
        
        # Find a compiled dataset
        dataset_dir = Path('datasets/compiled_datasets')
        dataset_versions = list(dataset_dir.glob('*/dataset.parquet'))
        
        if not dataset_versions:
            pytest.skip("No compiled datasets found")
        
        dataset_path = dataset_versions[0]
        
        # Load dataset
        table = pq.read_table(dataset_path)
        
        # Get token_ids column
        token_ids_column = table.column('token_ids').to_pylist()
        
        # Create simple dataset
        class TokenDataset(Dataset):
            def __init__(self, token_ids_list, max_len=128):
                self.token_ids_list = token_ids_list
                self.max_len = max_len
            
            def __len__(self):
                return len(self.token_ids_list)
            
            def __getitem__(self, idx):
                tokens = self.token_ids_list[idx][:self.max_len]
                # Pad if necessary
                if len(tokens) < self.max_len:
                    tokens = tokens + [0] * (self.max_len - len(tokens))
                return torch.tensor(tokens, dtype=torch.long)
        
        dataset = TokenDataset(token_ids_column)
        
        # Split train/val
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=2,
            shuffle=True,
            collate_fn=lambda x: {'input_ids': torch.stack(x)}
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=lambda x: {'input_ids': torch.stack(x)}
        )
        
        # Get vocab size from tokenizer metadata
        metadata_path = dataset_path.parent / 'metadata.json'
        import json
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        vocab_size = metadata.get('vocab_size', 1000)
        
        # Create model
        model_config = GPTConfig(
            vocab_size=vocab_size,
            max_seq_len=128,
            d_model=256,
            n_layers=4,
            n_heads=8
        )
        
        model = GPTModel(model_config)
        
        # Create training config
        train_config = TrainingConfig(
            learning_rate=3e-4,
            max_steps=20,
            warmup_steps=5,
            batch_size=2,
            eval_interval=10,
            checkpoint_interval=10,
            eval_steps=3
        )
        
        temp_dir = tempfile.mkdtemp()
        
        try:
            trainer = Trainer(
                model=model,
                train_dataloader=train_loader,
                val_dataloader=val_loader,
                config=train_config,
                checkpoint_dir=temp_dir,
                device='cpu'
            )
            
            trainer.train()
            
            # Verify training completed
            assert trainer.global_step == train_config.max_steps
            
            # Verify metrics
            assert len(trainer.train_metrics) > 0
            
            print(f"\n✅ Real dataset training test passed!")
            print(f"   Dataset: {dataset_path}")
            print(f"   Samples: {len(dataset)}")
            print(f"   Vocab size: {vocab_size}")
            print(f"   Final loss: {trainer.train_metrics[-1]['loss']:.4f}")
            
        finally:
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short", "-s"])
