"""
Tests for SFT Trainer

Bu modül SFT trainer'ın doğru çalıştığını test eder.
"""

import pytest
import torch
import tempfile
import json
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.sft_trainer import (
    InstructionExample,
    InstructionTemplate,
    InstructionDataset,
    SFTTrainer,
    TEMPLATES,
    create_instruction_dataset
)
from src.tokenizer.sentencepiece_tokenizer import SentencePieceTokenizer
from src.model.gpt import GPTModel, GPTConfig


class TestInstructionExample:
    """Test InstructionExample dataclass."""
    
    def test_create_example(self):
        """Test creating instruction example."""
        example = InstructionExample(
            instruction="What is AI?",
            response="AI is artificial intelligence.",
            system="You are a helpful assistant."
        )
        
        assert example.instruction == "What is AI?"
        assert example.response == "AI is artificial intelligence."
        assert example.system == "You are a helpful assistant."
    
    def test_to_dict(self):
        """Test converting to dict."""
        example = InstructionExample(
            instruction="Test",
            response="Answer"
        )
        
        data = example.to_dict()
        
        assert data['instruction'] == "Test"
        assert data['response'] == "Answer"
        assert 'metadata' in data
    
    def test_from_dict(self):
        """Test creating from dict."""
        data = {
            'instruction': "Question",
            'response': "Answer",
            'system': "System prompt"
        }
        
        example = InstructionExample.from_dict(data)
        
        assert example.instruction == "Question"
        assert example.response == "Answer"
        assert example.system == "System prompt"


class TestInstructionTemplate:
    """Test InstructionTemplate."""
    
    def test_simple_template(self):
        """Test simple template formatting."""
        template = TEMPLATES['simple']
        
        formatted, instr_len = template.format(
            instruction="Hello",
            response="Hi there!"
        )
        
        assert "Hello" in formatted
        assert "Hi there!" in formatted
        assert "User:" in formatted
        assert "Assistant:" in formatted
        assert instr_len > 0
    
    def test_alpaca_template(self):
        """Test Alpaca template."""
        template = TEMPLATES['alpaca']
        
        formatted, instr_len = template.format(
            instruction="Explain AI",
            response="AI is..."
        )
        
        assert "### Instruction:" in formatted
        assert "### Response:" in formatted
        assert "Explain AI" in formatted
    
    def test_with_system(self):
        """Test template with system prompt."""
        template = TEMPLATES['chatml']
        
        formatted, instr_len = template.format(
            instruction="Test",
            response="Answer",
            system="You are helpful"
        )
        
        assert "system" in formatted
        assert "You are helpful" in formatted
    
    def test_with_input(self):
        """Test template with additional input."""
        template = TEMPLATES['simple']
        
        formatted, instr_len = template.format(
            instruction="Summarize this",
            response="Summary",
            input_text="Long text here..."
        )
        
        assert "Long text here..." in formatted
        assert "Input:" in formatted


class TestInstructionDataset:
    """Test InstructionDataset."""
    
    @pytest.fixture
    def tokenizer(self):
        """Create test tokenizer."""
        # Create temp corpus - more diverse text for vocab
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            sentences = [
                "This is a test sentence for tokenizer training.",
                "Python is a programming language used for AI.",
                "Machine learning models require large datasets.",
                "Natural language processing is fascinating.",
                "Deep learning uses neural networks effectively."
            ]
            for sentence in sentences * 20:  # 100 sentences
                f.write(sentence + "\n")
            corpus_path = f.name
        
        # Train tokenizer
        tokenizer = SentencePieceTokenizer()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model_prefix = Path(tmpdir) / "test_tok"
            tokenizer.train(
                corpus_path=corpus_path,
                vocab_size=150,  # Increased for diverse text
                model_prefix=str(model_prefix)
            )
        
        Path(corpus_path).unlink()
        
        return tokenizer
    
    @pytest.fixture
    def examples(self):
        """Create test examples."""
        return [
            InstructionExample(
                instruction="What is 2+2?",
                response="2+2 equals 4."
            ),
            InstructionExample(
                instruction="Explain Python",
                response="Python is a programming language."
            ),
            InstructionExample(
                instruction="What is AI?",
                response="AI stands for Artificial Intelligence."
            )
        ]
    
    def test_dataset_creation(self, tokenizer, examples):
        """Test creating instruction dataset."""
        dataset = InstructionDataset(
            examples=examples,
            tokenizer=tokenizer,
            max_length=64,
            template=TEMPLATES['simple']
        )
        
        assert len(dataset) == 3
    
    def test_dataset_getitem(self, tokenizer, examples):
        """Test getting item from dataset."""
        dataset = InstructionDataset(
            examples=examples,
            tokenizer=tokenizer,
            max_length=64
        )
        
        item = dataset[0]
        
        assert 'input_ids' in item
        assert 'labels' in item
        assert 'attention_mask' in item
        
        assert item['input_ids'].dtype == torch.long
        assert item['labels'].dtype == torch.long
        assert item['attention_mask'].dtype == torch.long
    
    def test_instruction_masking(self, tokenizer, examples):
        """Test that instruction is masked in labels."""
        dataset = InstructionDataset(
            examples=examples,
            tokenizer=tokenizer,
            max_length=64,
            mask_instruction=True
        )
        
        item = dataset[0]
        labels = item['labels']
        
        # Should have some -100 (masked tokens)
        assert (labels == -100).any()
    
    def test_no_masking(self, tokenizer, examples):
        """Test without instruction masking."""
        dataset = InstructionDataset(
            examples=examples,
            tokenizer=tokenizer,
            max_length=64,
            mask_instruction=False
        )
        
        item = dataset[0]
        labels = item['labels']
        
        # Might have -100 from padding but not from masking
        # Check that most tokens are not masked
        non_masked = (labels != -100).sum()
        assert non_masked > len(labels) * 0.5
    
    def test_collate_fn(self, tokenizer, examples):
        """Test batch collation."""
        dataset = InstructionDataset(
            examples=examples,
            tokenizer=tokenizer,
            max_length=32
        )
        
        batch = [dataset[i] for i in range(2)]
        collated = InstructionDataset.collate_fn(batch)
        
        assert collated['input_ids'].shape[0] == 2  # batch size
        assert collated['labels'].shape[0] == 2
        assert collated['attention_mask'].shape[0] == 2
        
        # Check all have same length (padded)
        assert collated['input_ids'].shape[1] == collated['labels'].shape[1]
    
    def test_truncation(self, tokenizer, examples):
        """Test that long sequences are truncated."""
        dataset = InstructionDataset(
            examples=examples,
            tokenizer=tokenizer,
            max_length=10  # Very short
        )
        
        item = dataset[0]
        
        # Should be truncated to max_length
        assert item['input_ids'].size(0) <= 10


class TestSFTTrainer:
    """Test SFTTrainer."""
    
    @pytest.fixture
    def model(self):
        """Create small test model."""
        config = GPTConfig(
            vocab_size=150,  # Match tokenizer vocab size
            max_seq_len=32,
            d_model=64,
            n_layers=2,
            n_heads=2,
            d_ff=128
        )
        return GPTModel(config)
    
    @pytest.fixture
    def tokenizer(self):
        """Create test tokenizer."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            sentences = [
                "Test sentence for training tokenizer model.",
                "Another example with different words here.",
                "Machine learning and artificial intelligence.",
                "Python programming language for data science.",
                "Natural language processing techniques today."
            ]
            for sentence in sentences * 20:  # 100 sentences
                f.write(sentence + "\n")
            corpus_path = f.name
        
        tokenizer = SentencePieceTokenizer()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model_prefix = Path(tmpdir) / "test_tok"
            tokenizer.train(
                corpus_path=corpus_path,
                vocab_size=150,
                model_prefix=str(model_prefix)
            )
        
        Path(corpus_path).unlink()
        return tokenizer
    
    @pytest.fixture
    def train_dataset(self, tokenizer):
        """Create training dataset."""
        examples = [
            InstructionExample(
                instruction=f"Question {i}",
                response=f"Answer {i}"
            )
            for i in range(10)
        ]
        
        return InstructionDataset(
            examples=examples,
            tokenizer=tokenizer,
            max_length=32
        )
    
    def test_trainer_init(self, model, tokenizer, train_dataset):
        """Test trainer initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=train_dataset,
                output_dir=tmpdir,
                max_steps=5
            )
            
            assert trainer.model is model
            assert trainer.tokenizer is tokenizer
            assert trainer.max_steps == 5
            assert Path(tmpdir).exists()
    
    def test_loss_computation(self, model, tokenizer, train_dataset):
        """Test loss computation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=train_dataset,
                output_dir=tmpdir
            )
            
            # Create dummy batch
            logits = torch.randn(2, 10, 150)  # [batch, seq, vocab] - vocab=150
            labels = torch.randint(0, 150, (2, 10))  # [batch, seq]
            
            loss = trainer._compute_loss(logits, labels)
            
            assert isinstance(loss, torch.Tensor)
            assert loss.dim() == 0  # scalar
            assert loss.item() > 0
    
    def test_loss_with_masking(self, model, tokenizer, train_dataset):
        """Test that masked tokens are ignored in loss."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=train_dataset,
                output_dir=tmpdir
            )
            
            logits = torch.randn(2, 10, 150)  # vocab=150
            labels = torch.randint(0, 150, (2, 10))
            
            # Mask first half
            labels[:, :5] = -100
            
            loss = trainer._compute_loss(logits, labels)
            
            # Should still compute (on non-masked tokens)
            assert loss.item() > 0
    
    def test_training_short(self, model, tokenizer, train_dataset):
        """Test short training run."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=train_dataset,
                batch_size=2,
                max_steps=3,
                eval_steps=100,  # No eval
                save_steps=100,  # No save
                output_dir=tmpdir
            )
            
            history = trainer.train()
            
            assert 'loss' in history
            assert len(history['loss']) > 0
            assert trainer.global_step == 3


class TestCreateInstructionDataset:
    """Test create_instruction_dataset utility."""
    
    @pytest.fixture
    def data_file(self):
        """Create test data file."""
        data = [
            {
                "instruction": "What is Python?",
                "response": "Python is a programming language."
            },
            {
                "instruction": "Explain AI",
                "response": "AI is artificial intelligence.",
                "system": "Be concise"
            }
        ]
        
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.json',
            delete=False
        ) as f:
            json.dump(data, f)
            path = f.name
        
        yield path
        
        Path(path).unlink()
    
    @pytest.fixture
    def tokenizer(self):
        """Create tokenizer."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            sentences = [
                "Test data for tokenizer training process.",
                "Different sentences with varied vocabulary.",
                "Python language models and AI systems.",
                "Machine learning requires large datasets.",
                "Natural language understanding is important."
            ]
            for sentence in sentences * 20:
                f.write(sentence + "\n")
            corpus_path = f.name
        
        tokenizer = SentencePieceTokenizer()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model_prefix = Path(tmpdir) / "tok"
            tokenizer.train(
                corpus_path=corpus_path,
                vocab_size=150,
                model_prefix=str(model_prefix)
            )
        
        Path(corpus_path).unlink()
        return tokenizer
    
    def test_create_from_file(self, data_file, tokenizer):
        """Test creating dataset from JSON file."""
        dataset = create_instruction_dataset(
            data_path=data_file,
            tokenizer=tokenizer,
            max_length=64,
            template_name='simple'
        )
        
        assert len(dataset) == 2
        assert isinstance(dataset, InstructionDataset)
    
    def test_different_templates(self, data_file, tokenizer):
        """Test with different templates."""
        for template_name in ['simple', 'alpaca', 'chatml']:
            dataset = create_instruction_dataset(
                data_path=data_file,
                tokenizer=tokenizer,
                template_name=template_name
            )
            
            assert len(dataset) == 2


def test_templates_available():
    """Test that all templates are available."""
    assert 'simple' in TEMPLATES
    assert 'alpaca' in TEMPLATES
    assert 'chatml' in TEMPLATES


def test_import():
    """Test that module can be imported."""
    from src.training import sft_trainer
    assert hasattr(sft_trainer, 'SFTTrainer')
    assert hasattr(sft_trainer, 'InstructionDataset')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
