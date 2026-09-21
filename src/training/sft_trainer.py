"""
Supervised Fine-Tuning (SFT) Trainer

Bu modül base model'in instruction-following model'e dönüştürülmesi için
SFT training infrastrüktürünü sağlar.

SFT Süreci:
1. Base model: Next-token prediction ile eğitilmiş
2. Instruction data: (instruction, response) çiftleri
3. SFT training: Sadece response kısmında loss hesaplanır
4. Result: Instruction-following assistant model

Matematiksel Formül:
    Loss = -Σ log P(response_token | instruction, previous_response_tokens)
    
    Instruction kısmında loss = 0 (ignore_index=-100)
    Response kısmında loss hesaplanır

Örnek:
    Instruction: "Türkiye'nin başkenti neresidir?"
    Response: "Türkiye'nin başkenti Ankara'dır."
    
    Training sırasında:
    - "Türkiye'nin başkenti neresidir?" → loss hesaplanmaz
    - "Türkiye'nin başkenti Ankara'dır." → loss hesaplanır

Version: 1.0.0
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


@dataclass
class InstructionExample:
    """
    Single instruction-response example.
    
    Attributes:
        instruction: User instruction/prompt
        response: Model response (target)
        system: System prompt (optional)
        input: Additional input context (optional)
        metadata: Extra information (optional)
    """
    instruction: str
    response: str
    system: Optional[str] = None
    input: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Dict'e dönüştür."""
        return {
            'instruction': self.instruction,
            'response': self.response,
            'system': self.system,
            'input': self.input,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'InstructionExample':
        """Dict'ten oluştur."""
        return cls(
            instruction=data['instruction'],
            response=data['response'],
            system=data.get('system'),
            input=data.get('input'),
            metadata=data.get('metadata', {})
        )


@dataclass
class InstructionTemplate:
    """
    Instruction formatting template.
    
    Template kullanarak instruction-response çiftini
    model input formatına dönüştürür.
    
    Örnek Alpaca format:
        "Below is an instruction...\\n\\n### Instruction:\\n{instruction}\\n\\n### Response:\\n{response}"
    
    Örnek ChatML format:
        "<|im_start|>user\\n{instruction}<|im_end|>\\n<|im_start|>assistant\\n{response}<|im_end|>"
    """
    name: str
    system_template: str = ""
    user_template: str = "### Instruction:\n{instruction}\n\n"
    assistant_template: str = "### Response:\n{response}"
    separator: str = "\n\n"
    
    def format(
        self,
        instruction: str,
        response: str,
        system: Optional[str] = None,
        input_text: Optional[str] = None
    ) -> Tuple[str, int]:
        """
        Format instruction-response pair.
        
        Args:
            instruction: User instruction
            response: Model response
            system: System prompt (optional)
            input_text: Additional input (optional)
            
        Returns:
            Tuple[str, int]: (formatted_text, instruction_length)
            instruction_length: Instruction kısmının token sayısı (maskeleme için)
        """
        parts = []
        
        # System prompt
        if system and self.system_template:
            parts.append(self.system_template.format(system=system))
        
        # Instruction
        instruction_text = instruction
        if input_text:
            instruction_text = f"{instruction}\n\nInput: {input_text}"
        
        instruction_part = self.user_template.format(instruction=instruction_text)
        parts.append(instruction_part)
        
        # Calculate instruction length (for masking)
        instruction_full = self.separator.join(parts)
        
        # Response
        response_part = self.assistant_template.format(response=response)
        parts.append(response_part)
        
        # Full text
        full_text = self.separator.join(parts)
        
        return full_text, len(instruction_full)


# Predefined templates
TEMPLATES = {
    'alpaca': InstructionTemplate(
        name='alpaca',
        system_template="",
        user_template="### Instruction:\n{instruction}\n\n",
        assistant_template="### Response:\n{response}",
        separator=""
    ),
    'chatml': InstructionTemplate(
        name='chatml',
        system_template="<|im_start|>system\n{system}<|im_end|>\n",
        user_template="<|im_start|>user\n{instruction}<|im_end|>\n",
        assistant_template="<|im_start|>assistant\n{response}<|im_end|>",
        separator=""
    ),
    'simple': InstructionTemplate(
        name='simple',
        system_template="",
        user_template="User: {instruction}\n",
        assistant_template="Assistant: {response}",
        separator=""
    )
}


class InstructionDataset(Dataset):
    """
    Instruction dataset for SFT training.
    
    Bu dataset instruction-response çiftlerini model training formatına
    dönüştürür. Loss sadece response kısmında hesaplanır.
    
    Args:
        examples: List of InstructionExample
        tokenizer: Tokenizer instance
        max_length: Maximum sequence length
        template: Instruction template
        mask_instruction: Instruction kısmını maskele (loss=0)
    """
    
    def __init__(
        self,
        examples: List[InstructionExample],
        tokenizer: Any,
        max_length: int = 512,
        template: InstructionTemplate = TEMPLATES['simple'],
        mask_instruction: bool = True
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.template = template
        self.mask_instruction = mask_instruction
        
        logger.info(f"InstructionDataset initialized")
        logger.info(f"  Examples: {len(examples)}")
        logger.info(f"  Max length: {max_length}")
        logger.info(f"  Template: {template.name}")
        logger.info(f"  Mask instruction: {mask_instruction}")
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get single training example.
        
        Returns:
            Dict with:
                - input_ids: [seq_len]
                - labels: [seq_len] (instruction masked with -100)
                - attention_mask: [seq_len]
        """
        example = self.examples[idx]
        
        # Format with template
        full_text, instruction_length_chars = self.template.format(
            instruction=example.instruction,
            response=example.response,
            system=example.system,
            input_text=example.input
        )
        
        # Tokenize
        tokens = self.tokenizer.encode(full_text, add_bos=True, add_eos=True)
        
        # Truncate if needed
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        
        # Create labels
        labels = tokens.copy()
        
        if self.mask_instruction:
            # Tokenize instruction part to find where to mask
            instruction_text = full_text[:instruction_length_chars]
            instruction_tokens = self.tokenizer.encode(
                instruction_text, 
                add_bos=True, 
                add_eos=False
            )
            
            # Mask instruction part (set to -100)
            mask_length = min(len(instruction_tokens), len(labels))
            labels[:mask_length] = [-100] * mask_length
        
        # Convert to tensors
        input_ids = torch.tensor(tokens, dtype=torch.long)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        
        return {
            'input_ids': input_ids,
            'labels': labels_tensor,
            'attention_mask': attention_mask
        }
    
    @staticmethod
    def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate batch with padding.
        
        Args:
            batch: List of examples from __getitem__
            
        Returns:
            Batched tensors with padding
        """
        # Find max length in batch
        max_len = max(item['input_ids'].size(0) for item in batch)
        
        batch_input_ids = []
        batch_labels = []
        batch_attention_mask = []
        
        for item in batch:
            input_ids = item['input_ids']
            labels = item['labels']
            attention_mask = item['attention_mask']
            
            # Padding
            padding_length = max_len - input_ids.size(0)
            
            if padding_length > 0:
                # Pad with pad_token_id (assume 3)
                # input_ids shape: [seq_len] → [max_len]
                input_ids = torch.cat([
                    input_ids,
                    torch.full((padding_length,), 3, dtype=torch.long)
                ])
                # Pad labels with -100 (ignore)
                # labels shape: [seq_len] → [max_len]
                labels = torch.cat([
                    labels,
                    torch.full((padding_length,), -100, dtype=torch.long)
                ])
                # Pad attention mask with 0
                # attention_mask shape: [seq_len] → [max_len]
                attention_mask = torch.cat([
                    attention_mask,
                    torch.zeros(padding_length, dtype=torch.long)
                ])
            
            batch_input_ids.append(input_ids)
            batch_labels.append(labels)
            batch_attention_mask.append(attention_mask)
        
        # Stack to create batch
        # Each tensor: [max_len] → Stacked: [batch, max_len]
        return {
            'input_ids': torch.stack(batch_input_ids),        # [batch, seq_len]
            'labels': torch.stack(batch_labels),              # [batch, seq_len]
            'attention_mask': torch.stack(batch_attention_mask)  # [batch, seq_len]
        }


class SFTTrainer:
    """
    Supervised Fine-Tuning Trainer.
    
    Base model'i instruction-following model'e dönüştürür.
    
    Training Strategy:
    1. Instruction kısmında loss hesaplanmaz (labels=-100)
    2. Response kısmında cross-entropy loss
    3. Gradient accumulation support
    4. Checkpoint management
    
    Args:
        model: Base model to fine-tune
        tokenizer: Tokenizer
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset (optional)
        learning_rate: Learning rate
        batch_size: Batch size
        gradient_accumulation_steps: Accumulation steps
        max_steps: Maximum training steps
        eval_steps: Evaluation frequency
        save_steps: Checkpoint save frequency
        output_dir: Output directory
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        train_dataset: InstructionDataset,
        eval_dataset: Optional[InstructionDataset] = None,
        learning_rate: float = 2e-5,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 1,
        max_steps: int = 1000,
        eval_steps: int = 100,
        save_steps: int = 500,
        output_dir: str = "sft_output",
        warmup_steps: int = 0,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_steps = max_steps
        self.eval_steps = eval_steps
        self.save_steps = save_steps
        self.output_dir = Path(output_dir)
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Training state
        self.global_step = 0
        self.current_loss = 0.0
        
        logger.info("SFTTrainer initialized")
        logger.info(f"  Output dir: {self.output_dir}")
        logger.info(f"  Learning rate: {learning_rate}")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Gradient accumulation: {gradient_accumulation_steps}")
        logger.info(f"  Max steps: {max_steps}")
    
    def train(self) -> Dict[str, Any]:
        """
        Run SFT training.
        
        Returns:
            Training history
        """
        from torch.utils.data import DataLoader
        
        logger.info("Starting SFT training...")
        
        # DataLoader
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=InstructionDataset.collate_fn
        )
        
        self.model.train()
        device = next(self.model.parameters()).device
        
        history = {
            'loss': [],
            'eval_loss': [],
            'steps': []
        }
        
        steps_in_epoch = 0
        accumulated_loss = 0.0
        
        while self.global_step < self.max_steps:
            for batch in train_loader:
                # Move to device
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                # Forward pass
                outputs = self.model(input_ids)
                logits = outputs[0] if isinstance(outputs, tuple) else outputs
                
                # Compute loss (only on non-masked tokens)
                loss = self._compute_loss(logits, labels)
                
                # Scale loss for gradient accumulation
                loss = loss / self.gradient_accumulation_steps
                
                # Backward
                loss.backward()
                
                accumulated_loss += loss.item()
                steps_in_epoch += 1
                
                # Update weights
                if steps_in_epoch % self.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_grad_norm
                    )
                    
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    self.global_step += 1
                    self.current_loss = accumulated_loss
                    
                    # Log
                    if self.global_step % 10 == 0:
                        logger.info(
                            f"Step {self.global_step}/{self.max_steps} | "
                            f"Loss: {self.current_loss:.4f}"
                        )
                    
                    history['loss'].append(self.current_loss)
                    history['steps'].append(self.global_step)
                    
                    accumulated_loss = 0.0
                    
                    # Evaluation
                    if self.global_step % self.eval_steps == 0 and self.eval_dataset:
                        eval_loss = self._evaluate()
                        history['eval_loss'].append(eval_loss)
                        logger.info(f"  Eval loss: {eval_loss:.4f}")
                        self.model.train()
                    
                    # Save checkpoint
                    if self.global_step % self.save_steps == 0:
                        self._save_checkpoint()
                    
                    # Check if done
                    if self.global_step >= self.max_steps:
                        break
            
            if self.global_step >= self.max_steps:
                break
        
        # Final save
        self._save_checkpoint()
        
        logger.info("SFT training complete!")
        
        return history
    
    def _compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute SFT loss.
        
        Args:
            logits: Model predictions [batch, seq_len, vocab_size]
            labels: Target labels [batch, seq_len]
            
        Returns:
            Loss scalar
        """
        # Flatten for cross entropy
        # logits shape: [batch, seq_len, vocab_size] → [batch*seq_len, vocab_size]
        # labels shape: [batch, seq_len] → [batch*seq_len]
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),  # [batch*seq_len, vocab_size]
            labels.view(-1),                    # [batch*seq_len]
            ignore_index=-100  # Ignore masked instruction tokens
        )
        
        return loss
    
    def _evaluate(self) -> float:
        """
        Run evaluation.
        
        Returns:
            Average evaluation loss
        """
        if not self.eval_dataset:
            return 0.0
        
        from torch.utils.data import DataLoader
        
        self.model.eval()
        device = next(self.model.parameters()).device
        
        eval_loader = DataLoader(
            self.eval_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=InstructionDataset.collate_fn
        )
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in eval_loader:
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = self.model(input_ids)
                logits = outputs[0] if isinstance(outputs, tuple) else outputs
                
                loss = self._compute_loss(logits, labels)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        return avg_loss
    
    def _save_checkpoint(self):
        """Save training checkpoint."""
        checkpoint_path = self.output_dir / f"checkpoint-{self.global_step}.pt"
        
        torch.save({
            'step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.current_loss
        }, checkpoint_path)
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")


def create_instruction_dataset(
    data_path: str,
    tokenizer: Any,
    max_length: int = 512,
    template_name: str = 'simple',
    mask_instruction: bool = True
) -> InstructionDataset:
    """
    Create instruction dataset from JSON file.
    
    Args:
        data_path: Path to JSON file with instruction examples
        tokenizer: Tokenizer instance
        max_length: Maximum sequence length
        template_name: Template name ('simple', 'alpaca', 'chatml')
        mask_instruction: Mask instruction in labels
        
    Returns:
        InstructionDataset
        
    JSON Format:
        [
            {
                "instruction": "Question or task",
                "response": "Answer or completion",
                "system": "System prompt (optional)",
                "input": "Additional context (optional)"
            },
            ...
        ]
    """
    # Load data
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Create examples
    examples = [InstructionExample.from_dict(item) for item in data]
    
    # Get template
    template = TEMPLATES.get(template_name, TEMPLATES['simple'])
    
    # Create dataset
    dataset = InstructionDataset(
        examples=examples,
        tokenizer=tokenizer,
        max_length=max_length,
        template=template,
        mask_instruction=mask_instruction
    )
    
    return dataset


def main():
    """Demo SFT training."""
    # This would be used with actual model and tokenizer
    print("SFT Trainer module loaded successfully!")
    print(f"Available templates: {list(TEMPLATES.keys())}")


if __name__ == "__main__":
    main()
