"""
examples/end_to_end_demo_simple.py

Simplified End-to-End Demo

Demonstrates the complete pipeline with all Sprint 9 & 10 components:
1. Data Preparation
2. Tokenizer Training
3. Model Training  
4. Evaluation & Benchmarks
5. Model Comparison
6. Quality Tracking
7. Model Registry
8. Inference Testing
9. Dashboard Generation

Usage:
    python examples/end_to_end_demo_simple.py
"""

import sys
from pathlib import Path
import torch
import time

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.tokenizer.sentencepiece_tokenizer import SentencePieceTokenizer
from src.data.dataset import TextDataset, DataLoader
from src.model.gpt import GPTModel, GPTConfig
from src.training.checkpoint_manager import CheckpointManager, CheckpointConfig
from src.evaluation.metrics import evaluate_batch, aggregate_results
from src.evaluation.benchmarks import create_turkish_generation_benchmark, run_generation_benchmark
from src.evaluation.comparison import ModelComparison
from src.evaluation.quality_tracker import QualityTracker
from src.registry.model_registry import ModelRegistry
from src.evaluation.dashboard import DashboardGenerator
from src.inference.generation import generate_text

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    print("\n" + "=" * 80)
    print("AI RESEARCH LAB - END-TO-END DEMO (SIMPLIFIED)")
    print("=" * 80)
    
    start_time = time.time()
    output_dir = project_root / "demo_output"
    output_dir.mkdir(exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Sample data
    sample_texts = [
        "Türkiye Cumhuriyeti Anadolu yarımadasında yer alır.",
        "İstanbul Boğazı Avrupa ve Asya'yı ayırır.",
        "Ankara Türkiye'nin başkentidir.",
        "Yapay zeka bilgisayar biliminin bir dalıdır.",
        "Derin öğrenme sinir ağları kullanır.",
    ] * 40  # 200 sentences
    
    # Step 1: Tokenizer
    print("\n[1/9] Tokenizer Training...")
    tokenizer_dir = output_dir / "tokenizer"
    tokenizer_dir.mkdir(exist_ok=True)
    
    train_file = tokenizer_dir / "data.txt"
    with open(train_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(sample_texts))
    
    tokenizer = SentencePieceTokenizer()
    tokenizer.train(str(train_file), vocab_size=300, model_prefix=str(tokenizer_dir / "tokenizer"))
    print(f"✓ Tokenizer: {tokenizer.vocab_size} vocab")
    
    # Step 2: Dataset
    print("\n[2/9] Dataset Preparation...")
    train_data = sample_texts[:160]
    val_data = sample_texts[160:]
    
    train_dataset = TextDataset(train_data, tokenizer, max_length=32)
    val_dataset = TextDataset(val_data, tokenizer, max_length=32)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    print(f"✓ Dataset: {len(train_dataset)} train, {len(val_dataset)} val")
    
    # Step 3: Model
    print("\n[3/9] Model Initialization...")
    model_config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=32,
        d_model=128,
        n_layers=4,
        n_heads=4,
        d_ff=512,
        dropout=0.1
    )
    
    model = GPTModel(model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    print(f"✓ Model: {sum(p.numel() for p in model.parameters()):,} params")
    
    # Step 4: Training with Quality Tracking
    print("\n[4/9] Training (3 epochs)...")
    
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=str(output_dir / "checkpoints"),
        model_name="gpt-demo",
        config=CheckpointConfig(keep_top_k=2, metric_for_best='loss', mode='min')
    )
    
    quality_tracker = QualityTracker(
        metrics_to_track=['loss', 'perplexity', 'accuracy'],
        regression_threshold=0.05
    )
    
    for epoch in range(1, 4):
        print(f"\nEpoch {epoch}/3")
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            output = model(input_ids)
            logits = output[0] if isinstance(output, tuple) else output
            
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_results_list = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                output = model(input_ids)
                logits = output[0] if isinstance(output, tuple) else output
                val_results_list.append(evaluate_batch(logits, labels))
        
        val_results = aggregate_results(val_results_list)
        
        print(f"  Train loss: {avg_loss:.4f}")
        print(f"  Val perplexity: {val_results.get_metric('perplexity').value:.4f}")
        
        # Quality tracking
        quality_tracker.add_evaluation(epoch, val_results)
        
        # Save checkpoint
        checkpoint_manager.save_checkpoint(
            model=model,
            epoch=epoch,
            step=epoch * len(train_loader),  # Total steps so far
            metrics={
                'loss': avg_loss,
                'perplexity': val_results.get_metric('perplexity').value,
                'accuracy': val_results.get_metric('accuracy').value
            },
            optimizer=optimizer,
            model_config=model_config.__dict__,
            version=f"0.1.{epoch}"
        )
    
    print("✓ Training complete")
    
    # Step 5: Evaluation
    print("\n[5/9] Evaluation & Benchmarks...")
    benchmark = create_turkish_generation_benchmark()
    benchmark_results = run_generation_benchmark(
        model, tokenizer, benchmark, max_length=15, device=str(device)
    )
    print(f"✓ Benchmark: BLEU-4={benchmark_results.get_metric('bleu-4').value:.2f}")
    
    # Step 6: Model Comparison
    print("\n[6/9] Model Comparison...")
    comparison = ModelComparison()
    comparison.add_model("Final Model", val_results)
    comparison.add_model("Benchmark", benchmark_results)
    report = comparison.compare()
    report.save(str(output_dir / "comparison.json"))
    print(f"✓ Winner: {report.overall_winner}")
    
    # Step 7: Model Registry
    print("\n[7/9] Model Registry...")
    registry = ModelRegistry(registry_dir=str(output_dir / "registry"))
    
    # Get latest checkpoint
    latest_checkpoint = str(output_dir / "checkpoints" / f"gpt-demo_v0.1.3_epoch3_step60.pt")
    
    metadata = registry.register_model(
        model_name="gpt-demo",
        version="1.0.0",
        checkpoint_path=latest_checkpoint,
        tokenizer_path=str(tokenizer_dir / "tokenizer.model"),
        description="Demo GPT model for E2E integration test",
        architecture="GPT",
        parameters=sum(p.numel() for p in model.parameters()),
        metrics={
            'perplexity': float(val_results.get_metric('perplexity').value),
            'accuracy': float(val_results.get_metric('accuracy').value)
        },
        training_config=model_config.__dict__,
        tags=['demo', 'e2e-test'],
        environment='development'
    )
    print(f"✓ Registered: {metadata.model_name} v{metadata.version}")
    
    # Step 8: Inference
    print("\n[8/9] Inference Testing...")
    model.eval()
    test_prompts = ["Türkiye", "Yapay zeka"]
    
    for prompt in test_prompts:
        input_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
        
        with torch.no_grad():
            output = generate_text(model, input_tensor, max_new_tokens=10, temperature=0.8)
        
        generated = tokenizer.decode(output[0].cpu().tolist(), skip_special_tokens=True)
        print(f"  '{prompt}' → '{generated}'")
    
    print("✓ Inference complete")
    
    # Step 9: Dashboard
    print("\n[9/9] Dashboard Generation...")
    dashboard = DashboardGenerator(
        title="AI Research Lab - Demo Dashboard",
        description="End-to-end pipeline results"
    )
    
    dashboard.add_evaluation_results("Final Model", val_results)
    dashboard.add_evaluation_results("Benchmark", benchmark_results)
    dashboard.add_comparison_report(report)
    dashboard.add_quality_tracking(quality_tracker)
    
    dashboard_path = output_dir / "dashboard.html"
    dashboard.generate(str(dashboard_path))
    print(f"✓ Dashboard: {dashboard_path}")
    
    # Summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"\nTime: {elapsed:.1f}s")
    print(f"Output: {output_dir}")
    print(f"Dashboard: {dashboard_path}")
    print("\nOpen dashboard.html in browser to view results!")
    print("=" * 80)


if __name__ == "__main__":
    main()
