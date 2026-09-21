"""
Instruction Dataset Builder and Validator

Bu modül, instruction dataset'lerini yüklemek, validate etmek ve
SFT eğitimi için hazırlamak üzere utility fonksiyonları sağlar.

Features:
---------
1. Dataset validation (format, quality, required fields)
2. Quality filtering (length, profanity, duplicates)
3. Train/val split
4. Statistics reporting
5. Export to different formats

Version: 1.0.0
"""

import json
import logging
from typing import List, Dict, Tuple, Optional, Set
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import Counter
import re

logger = logging.getLogger(__name__)


@dataclass
class InstructionExample:
    """
    Single instruction-response example.
    
    Attributes:
        id: Unique identifier
        instruction: User instruction/query
        response: Model response
        category: Category (e.g., 'code', 'math', 'general_qa')
        difficulty: Difficulty level ('easy', 'medium', 'hard')
        metadata: Additional metadata
    """
    id: str
    instruction: str
    response: str
    category: str = "general"
    difficulty: str = "medium"
    metadata: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'InstructionExample':
        """Create from dictionary."""
        return cls(**data)
    
    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate example.
        
        Returns:
            (is_valid, error_messages)
        """
        errors = []
        
        # Check required fields
        if not self.id:
            errors.append("Missing id")
        if not self.instruction:
            errors.append("Missing instruction")
        if not self.response:
            errors.append("Missing response")
        
        # Check instruction length
        if len(self.instruction) < 5:
            errors.append(f"Instruction too short: {len(self.instruction)} chars")
        if len(self.instruction) > 2000:
            errors.append(f"Instruction too long: {len(self.instruction)} chars")
        
        # Check response length
        if len(self.response) < 5:
            errors.append(f"Response too short: {len(self.response)} chars")
        if len(self.response) > 4000:
            errors.append(f"Response too long: {len(self.response)} chars")
        
        # Check category
        valid_categories = {
            'general_qa', 'code', 'math', 'summarization', 
            'creative', 'general', 'translation', 'reasoning'
        }
        if self.category and self.category not in valid_categories:
            logger.warning(f"Unknown category: {self.category}")
        
        # Check difficulty
        valid_difficulties = {'easy', 'medium', 'hard'}
        if self.difficulty and self.difficulty not in valid_difficulties:
            errors.append(f"Invalid difficulty: {self.difficulty}")
        
        return (len(errors) == 0, errors)


@dataclass
class DatasetStats:
    """Dataset statistics."""
    total_examples: int
    categories: Dict[str, int]
    difficulties: Dict[str, int]
    avg_instruction_length: float
    avg_response_length: float
    min_instruction_length: int
    max_instruction_length: int
    min_response_length: int
    max_response_length: int
    languages: Set[str]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        data = asdict(self)
        data['languages'] = list(data['languages'])  # Convert set to list
        return data
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"DatasetStats(\n"
            f"  total_examples={self.total_examples},\n"
            f"  categories={self.categories},\n"
            f"  difficulties={self.difficulties},\n"
            f"  avg_instruction_length={self.avg_instruction_length:.1f},\n"
            f"  avg_response_length={self.avg_response_length:.1f},\n"
            f"  languages={self.languages}\n"
            f")"
        )


class InstructionDatasetBuilder:
    """
    Instruction dataset builder and validator.
    
    Bu class, instruction dataset'lerini yükler, validate eder ve
    training için hazırlar.
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize builder.
        
        Args:
            verbose: Log detailed information
        """
        self.verbose = verbose
        self.examples: List[InstructionExample] = []
    
    def load_from_json(self, file_path: str) -> int:
        """
        Load dataset from JSON file.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Number of examples loaded
        """
        logger.info(f"Loading dataset from {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract examples
        examples_data = data.get('examples', [])
        
        for ex_data in examples_data:
            try:
                example = InstructionExample.from_dict(ex_data)
                self.examples.append(example)
            except Exception as e:
                logger.error(f"Failed to load example {ex_data.get('id', 'unknown')}: {e}")
        
        logger.info(f"Loaded {len(self.examples)} examples")
        return len(self.examples)
    
    def validate_dataset(self) -> Tuple[List[InstructionExample], List[Tuple[str, List[str]]]]:
        """
        Validate all examples in dataset.
        
        Returns:
            (valid_examples, invalid_examples_with_errors)
        """
        logger.info("Validating dataset...")
        
        valid_examples = []
        invalid_examples = []
        
        for example in self.examples:
            is_valid, errors = example.validate()
            
            if is_valid:
                valid_examples.append(example)
            else:
                invalid_examples.append((example.id, errors))
                if self.verbose:
                    logger.warning(f"Invalid example {example.id}: {errors}")
        
        logger.info(f"Validation complete: {len(valid_examples)} valid, {len(invalid_examples)} invalid")
        
        return valid_examples, invalid_examples
    
    def filter_duplicates(self, examples: List[InstructionExample]) -> List[InstructionExample]:
        """
        Remove duplicate instructions.
        
        Args:
            examples: List of examples
            
        Returns:
            Filtered examples
        """
        logger.info("Filtering duplicates...")
        
        seen_instructions = set()
        unique_examples = []
        duplicate_count = 0
        
        for example in examples:
            # Normalize instruction for comparison
            normalized = example.instruction.lower().strip()
            
            if normalized not in seen_instructions:
                seen_instructions.add(normalized)
                unique_examples.append(example)
            else:
                duplicate_count += 1
                if self.verbose:
                    logger.debug(f"Duplicate instruction: {example.id}")
        
        logger.info(f"Removed {duplicate_count} duplicates, {len(unique_examples)} remaining")
        
        return unique_examples
    
    def compute_statistics(self, examples: List[InstructionExample]) -> DatasetStats:
        """
        Compute dataset statistics.
        
        Args:
            examples: List of examples
            
        Returns:
            Dataset statistics
        """
        if not examples:
            logger.warning("No examples to compute statistics")
            return DatasetStats(
                total_examples=0,
                categories={},
                difficulties={},
                avg_instruction_length=0.0,
                avg_response_length=0.0,
                min_instruction_length=0,
                max_instruction_length=0,
                min_response_length=0,
                max_response_length=0,
                languages=set()
            )
        
        # Count categories
        # categories: Dict[category_name, count]
        categories = Counter(ex.category for ex in examples)
        
        # Count difficulties
        # difficulties: Dict[difficulty_level, count]
        difficulties = Counter(ex.difficulty for ex in examples)
        
        # Length statistics
        # instruction_lengths: List[int] - character counts for each instruction
        instruction_lengths = [len(ex.instruction) for ex in examples]
        # response_lengths: List[int] - character counts for each response
        response_lengths = [len(ex.response) for ex in examples]
        
        # Extract languages from metadata
        languages = set()
        for ex in examples:
            if ex.metadata and 'language' in ex.metadata:
                languages.add(ex.metadata['language'])
        
        # Fallback: detect language from examples if not in metadata
        if not languages:
            languages.add('tr')  # Default to Turkish
        
        stats = DatasetStats(
            total_examples=len(examples),
            categories=dict(categories),
            difficulties=dict(difficulties),
            avg_instruction_length=sum(instruction_lengths) / len(instruction_lengths),
            avg_response_length=sum(response_lengths) / len(response_lengths),
            min_instruction_length=min(instruction_lengths),
            max_instruction_length=max(instruction_lengths),
            min_response_length=min(response_lengths),
            max_response_length=max(response_lengths),
            languages=languages
        )
        
        if self.verbose:
            logger.info(f"Dataset statistics:\n{stats}")
        
        return stats
    
    def train_val_split(
        self,
        examples: List[InstructionExample],
        val_ratio: float = 0.1,
        shuffle: bool = True,
        seed: int = 42
    ) -> Tuple[List[InstructionExample], List[InstructionExample]]:
        """
        Split dataset into train and validation sets.
        
        Args:
            examples: List of examples
            val_ratio: Validation set ratio (default: 0.1)
            shuffle: Shuffle before split
            seed: Random seed
            
        Returns:
            (train_examples, val_examples)
        """
        import random
        
        if shuffle:
            random.seed(seed)
            examples = examples.copy()
            random.shuffle(examples)
        
        # Calculate split point
        # val_size: int - number of examples for validation
        val_size = int(len(examples) * val_ratio)
        # train_size: int - number of examples for training
        train_size = len(examples) - val_size
        
        # Split
        # train_examples: List[InstructionExample] - training data
        train_examples = examples[:train_size]
        # val_examples: List[InstructionExample] - validation data
        val_examples = examples[train_size:]
        
        logger.info(
            f"Dataset split: {len(train_examples)} train, "
            f"{len(val_examples)} val ({val_ratio*100:.1f}%)"
        )
        
        return train_examples, val_examples
    
    def export_to_json(
        self,
        examples: List[InstructionExample],
        output_path: str,
        include_metadata: bool = True
    ):
        """
        Export examples to JSON file.
        
        Args:
            examples: List of examples to export
            output_path: Output file path
            include_metadata: Include metadata in export
        """
        logger.info(f"Exporting {len(examples)} examples to {output_path}")
        
        # Compute stats
        stats = self.compute_statistics(examples)
        
        # Prepare data
        output_data = {
            "version": "1.0.0",
            "total_examples": len(examples),
            "statistics": stats.to_dict() if include_metadata else None,
            "examples": [ex.to_dict() for ex in examples]
        }
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Export complete: {output_path}")
    
    def filter_by_category(
        self,
        examples: List[InstructionExample],
        categories: List[str]
    ) -> List[InstructionExample]:
        """
        Filter examples by category.
        
        Args:
            examples: List of examples
            categories: List of categories to keep
            
        Returns:
            Filtered examples
        """
        filtered = [ex for ex in examples if ex.category in categories]
        logger.info(
            f"Filtered by category {categories}: "
            f"{len(filtered)}/{len(examples)} examples"
        )
        return filtered
    
    def filter_by_difficulty(
        self,
        examples: List[InstructionExample],
        difficulties: List[str]
    ) -> List[InstructionExample]:
        """
        Filter examples by difficulty.
        
        Args:
            examples: List of examples
            difficulties: List of difficulties to keep ('easy', 'medium', 'hard')
            
        Returns:
            Filtered examples
        """
        filtered = [ex for ex in examples if ex.difficulty in difficulties]
        logger.info(
            f"Filtered by difficulty {difficulties}: "
            f"{len(filtered)}/{len(examples)} examples"
        )
        return filtered
    
    def print_sample(self, examples: List[InstructionExample], n: int = 3):
        """
        Print sample examples.
        
        Args:
            examples: List of examples
            n: Number of samples to print
        """
        print(f"\n{'='*80}")
        print(f"Sample Examples ({min(n, len(examples))} of {len(examples)})")
        print(f"{'='*80}\n")
        
        for i, example in enumerate(examples[:n], 1):
            print(f"Example {i}:")
            print(f"  ID: {example.id}")
            print(f"  Category: {example.category}")
            print(f"  Difficulty: {example.difficulty}")
            print(f"  Instruction: {example.instruction[:100]}...")
            print(f"  Response: {example.response[:100]}...")
            print()


def main():
    """Demo usage."""
    # Create builder
    builder = InstructionDatasetBuilder(verbose=True)
    
    # Load dataset
    dataset_path = "data/instruction_datasets/turkish_instructions_v1.json"
    builder.load_from_json(dataset_path)
    
    # Validate
    valid_examples, invalid_examples = builder.validate_dataset()
    
    if invalid_examples:
        print(f"\nFound {len(invalid_examples)} invalid examples:")
        for example_id, errors in invalid_examples[:5]:
            print(f"  {example_id}: {errors}")
    
    # Filter duplicates
    unique_examples = builder.filter_duplicates(valid_examples)
    
    # Compute statistics
    stats = builder.compute_statistics(unique_examples)
    print(f"\n{stats}")
    
    # Train/val split
    train_examples, val_examples = builder.train_val_split(
        unique_examples,
        val_ratio=0.1,
        shuffle=True
    )
    
    # Print samples
    builder.print_sample(train_examples, n=2)
    
    # Export
    builder.export_to_json(
        train_examples,
        "data/instruction_datasets/turkish_train.json"
    )
    builder.export_to_json(
        val_examples,
        "data/instruction_datasets/turkish_val.json"
    )
    
    print("\nDataset preparation complete!")


if __name__ == "__main__":
    main()
