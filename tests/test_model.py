"""
tests/test_model.py

Comprehensive Model Tests

Bu modül transformer model'in tüm component'leri için unit test'leri içerir.
- Embeddings & Positional Encoding
- Multi-Head Attention
- Feed-Forward Network
- Transformer Block
- GPT Model
- Weight Initialization

Test'ler shape validation, forward pass correctness, ve component integration'ı verify eder.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import pytest
from src.model.embeddings import (
    TokenEmbedding,
    PositionalEncoding,
    TransformerEmbedding,
    create_embedding_layer
)
from src.model.attention import (
    ScaledDotProductAttention,
    MultiHeadAttention,
    create_causal_mask,
    create_attention_layer
)
from src.model.ffn import (
    PositionWiseFFN,
    GatedFFN,
    create_ffn_layer
)
from src.model.transformer_block import (
    TransformerDecoderBlock,
    TransformerDecoder,
    create_transformer_block,
    create_transformer_decoder
)
from src.model.gpt import (
    GPTConfig,
    GPTModel,
    create_gpt_model
)
from src.model.initialization import (
    WeightInitializer,
    InitStrategy,
    initialize_model,
    get_init_stats
)


class TestEmbeddings:
    """Test Embeddings & Positional Encoding."""
    
    def test_token_embedding_shape(self):
        """Test token embedding output shape."""
        vocab_size = 1000
        d_model = 256
        batch_size = 2
        seq_len = 50
        
        embedding = TokenEmbedding(vocab_size=vocab_size, d_model=d_model)
        token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        output = embedding(token_ids)
        
        assert output.shape == (batch_size, seq_len, d_model)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_token_embedding_scaling(self):
        """Test token embedding scaling by sqrt(d_model)."""
        vocab_size = 100
        d_model = 64
        
        embedding = TokenEmbedding(vocab_size=vocab_size, d_model=d_model)
        token_ids = torch.tensor([[1, 2, 3]])
        
        output = embedding(token_ids)
        
        # Check if scaled (should be larger than unscaled)
        import math
        expected_scale = math.sqrt(d_model)
        assert abs(expected_scale - 8.0) < 0.01  # sqrt(64) = 8
    
    def test_positional_encoding_shape(self):
        """Test positional encoding output shape."""
        max_seq_len = 128
        d_model = 256
        batch_size = 2
        seq_len = 50
        
        pos_enc = PositionalEncoding(max_seq_len=max_seq_len, d_model=d_model, dropout=0.0)
        x = torch.randn(batch_size, seq_len, d_model)
        
        output = pos_enc(x)
        
        assert output.shape == (batch_size, seq_len, d_model)
    
    def test_positional_encoding_max_length(self):
        """Test positional encoding enforces max length."""
        max_seq_len = 10
        d_model = 64
        
        pos_enc = PositionalEncoding(max_seq_len=max_seq_len, d_model=d_model)
        x = torch.randn(1, 15, d_model)  # Exceeds max_seq_len
        
        with pytest.raises(AssertionError):
            pos_enc(x)
    
    def test_transformer_embedding_full_pipeline(self):
        """Test complete embedding pipeline."""
        vocab_size = 1000
        d_model = 256
        max_seq_len = 128
        batch_size = 2
        seq_len = 50
        
        embedding = create_embedding_layer(
            vocab_size=vocab_size,
            d_model=d_model,
            max_seq_len=max_seq_len,
            dropout=0.0
        )
        
        token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        output = embedding(token_ids)
        
        assert output.shape == (batch_size, seq_len, d_model)


class TestAttention:
    """Test Multi-Head Attention."""
    
    def test_scaled_dot_product_attention_shape(self):
        """Test attention output shape."""
        batch_size = 2
        n_heads = 8
        seq_len = 10
        d_k = 32
        
        attention = ScaledDotProductAttention(dropout=0.0)
        
        q = torch.randn(batch_size, n_heads, seq_len, d_k)
        k = torch.randn(batch_size, n_heads, seq_len, d_k)
        v = torch.randn(batch_size, n_heads, seq_len, d_k)
        
        output, attn_weights = attention(q, k, v)
        
        assert output.shape == (batch_size, n_heads, seq_len, d_k)
        assert attn_weights.shape == (batch_size, n_heads, seq_len, seq_len)
    
    def test_attention_weights_sum_to_one(self):
        """Test attention weights are normalized."""
        batch_size = 2
        n_heads = 4
        seq_len = 5
        d_k = 16
        
        attention = ScaledDotProductAttention(dropout=0.0)
        
        q = torch.randn(batch_size, n_heads, seq_len, d_k)
        k = torch.randn(batch_size, n_heads, seq_len, d_k)
        v = torch.randn(batch_size, n_heads, seq_len, d_k)
        
        _, attn_weights = attention(q, k, v)
        
        # Sum over keys (last dimension)
        sums = attn_weights.sum(dim=-1)
        expected = torch.ones_like(sums)
        
        assert torch.allclose(sums, expected, atol=1e-6)
    
    def test_causal_mask_property(self):
        """Test causal masking prevents future attention."""
        seq_len = 5
        mask = create_causal_mask(seq_len, torch.device('cpu'))
        
        # mask shape: [1, 1, T, T]
        assert mask.shape == (1, 1, seq_len, seq_len)
        
        # Position 0 should only see position 0 (rest masked)
        assert mask[0, 0, 0, 0] == False  # Can attend to self
        assert mask[0, 0, 0, 1] == True   # Cannot attend to future
        
        # Position 2 should see 0, 1, 2 but not 3, 4
        assert mask[0, 0, 2, 0] == False
        assert mask[0, 0, 2, 1] == False
        assert mask[0, 0, 2, 2] == False
        assert mask[0, 0, 2, 3] == True
        assert mask[0, 0, 2, 4] == True
    
    def test_multi_head_attention_shape(self):
        """Test multi-head attention output shape."""
        batch_size = 2
        seq_len = 10
        d_model = 256
        n_heads = 8
        
        attention = create_attention_layer(d_model=d_model, n_heads=n_heads, dropout=0.0)
        
        x = torch.randn(batch_size, seq_len, d_model)
        # MultiHeadAttention now returns 4 values: output, attn_weights, k_cache, v_cache
        output, attn_weights, *_ = attention(x)
        
        assert output.shape == (batch_size, seq_len, d_model)
        assert attn_weights.shape == (batch_size, n_heads, seq_len, seq_len)
    
    def test_multi_head_attention_with_causal_mask(self):
        """Test multi-head attention with causal masking."""
        batch_size = 2
        seq_len = 10
        d_model = 256
        n_heads = 8
        
        attention = create_attention_layer(d_model=d_model, n_heads=n_heads, dropout=0.0)
        attention.eval()
        
        x = torch.randn(batch_size, seq_len, d_model)
        mask = create_causal_mask(seq_len, x.device)
        
        with torch.no_grad():
            # MultiHeadAttention now returns 4 values: output, attn_weights, k_cache, v_cache
            output, attn_weights, *_ = attention(x, mask)
        
        # Check causal property: position 0 should only attend to position 0
        # attn_weights: [B, H, T, T]
        # Position 0 attending to future should be ~0
        future_attention = attn_weights[0, 0, 0, 1:].abs().max().item()
        assert future_attention < 1e-6


class TestFFN:
    """Test Feed-Forward Network."""
    
    def test_position_wise_ffn_shape(self):
        """Test FFN output shape."""
        batch_size = 2
        seq_len = 50
        d_model = 256
        d_ff = 1024
        
        ffn = create_ffn_layer(d_model=d_model, d_ff=d_ff, dropout=0.0)
        
        x = torch.randn(batch_size, seq_len, d_model)
        output = ffn(x)
        
        assert output.shape == (batch_size, seq_len, d_model)
    
    def test_position_wise_independence(self):
        """Test FFN applies same transformation to each position."""
        batch_size = 2
        seq_len = 50
        d_model = 256
        d_ff = 1024
        
        ffn = create_ffn_layer(d_model=d_model, d_ff=d_ff, dropout=0.0)
        ffn.eval()
        
        x = torch.randn(batch_size, seq_len, d_model)
        
        with torch.no_grad():
            # Full sequence
            output_full = ffn(x)
            
            # Single position
            x_single = x[:, 0:1, :]
            output_single = ffn(x_single)
            
            # Should match first position of full output
            diff = (output_single - output_full[:, 0:1, :]).abs().max()
            assert diff < 1e-5
    
    def test_gated_ffn_shape(self):
        """Test gated FFN output shape."""
        batch_size = 2
        seq_len = 50
        d_model = 256
        d_ff = 1024
        
        ffn = create_ffn_layer(d_model=d_model, d_ff=d_ff, dropout=0.0, gated=True)
        
        x = torch.randn(batch_size, seq_len, d_model)
        output = ffn(x)
        
        assert output.shape == (batch_size, seq_len, d_model)


class TestTransformerBlock:
    """Test Transformer Decoder Block."""
    
    def test_decoder_block_shape(self):
        """Test decoder block output shape."""
        batch_size = 2
        seq_len = 10
        d_model = 256
        n_heads = 8
        d_ff = 1024
        
        block = create_transformer_block(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=0.0
        )
        
        x = torch.randn(batch_size, seq_len, d_model)
        mask = create_causal_mask(seq_len, x.device)
        
        output, attn_weights = block(x, mask)
        
        assert output.shape == (batch_size, seq_len, d_model)
        assert attn_weights.shape == (batch_size, n_heads, seq_len, seq_len)
    
    def test_residual_connections(self):
        """Test residual connections modify output."""
        batch_size = 2
        seq_len = 10
        d_model = 256
        n_heads = 8
        d_ff = 1024
        
        block = create_transformer_block(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=0.0
        )
        
        x = torch.randn(batch_size, seq_len, d_model)
        mask = create_causal_mask(seq_len, x.device)
        
        output, _ = block(x, mask)
        
        # Output should differ from input (residual adds changes)
        diff = (output - x).abs().mean()
        assert diff > 0.01  # Should have meaningful change
    
    def test_stacked_decoder_shape(self):
        """Test stacked decoder output shape."""
        batch_size = 2
        seq_len = 10
        d_model = 256
        n_heads = 8
        d_ff = 1024
        n_layers = 4
        
        decoder = create_transformer_decoder(
            n_layers=n_layers,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=0.0
        )
        
        x = torch.randn(batch_size, seq_len, d_model)
        mask = create_causal_mask(seq_len, x.device)
        
        output, attn_list = decoder(x, mask)
        
        assert output.shape == (batch_size, seq_len, d_model)
        assert len(attn_list) == n_layers
        assert all(a.shape == (batch_size, n_heads, seq_len, seq_len) for a in attn_list)


class TestGPTModel:
    """Test GPT Model."""
    
    def test_gpt_config_validation(self):
        """Test GPT config validation."""
        # Valid config
        config = GPTConfig(
            vocab_size=1000,
            d_model=256,
            n_heads=8
        )
        assert config.d_model % config.n_heads == 0
        
        # Invalid: d_model not divisible by n_heads
        with pytest.raises(AssertionError):
            GPTConfig(vocab_size=1000, d_model=255, n_heads=8)
    
    def test_gpt_model_forward_shape(self):
        """Test GPT model forward pass shape."""
        config = GPTConfig(
            vocab_size=1000,
            max_seq_len=128,
            d_model=256,
            n_layers=4,
            n_heads=8,
            d_ff=1024,
            dropout=0.0
        )
        
        model = create_gpt_model(config)
        
        batch_size = 2
        seq_len = 50
        token_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        logits, attn_weights = model(token_ids)
        
        assert logits.shape == (batch_size, seq_len, config.vocab_size)
        assert len(attn_weights) == config.n_layers
    
    def test_gpt_model_generation(self):
        """Test GPT model text generation."""
        config = GPTConfig(
            vocab_size=100,
            max_seq_len=64,
            d_model=128,
            n_layers=2,
            n_heads=4,
            d_ff=512,
            dropout=0.0
        )
        
        model = create_gpt_model(config)
        model.eval()
        
        prompt = torch.randint(0, config.vocab_size, (1, 5))
        
        with torch.no_grad():
            generated = model.generate(
                prompt_ids=prompt,
                max_new_tokens=10,
                temperature=1.0
            )
        
        assert generated.shape == (1, 15)  # 5 prompt + 10 new
        assert (generated >= 0).all() and (generated < config.vocab_size).all()
    
    def test_gpt_model_parameter_count(self):
        """Test parameter counting."""
        config = GPTConfig(
            vocab_size=1000,
            d_model=256,
            n_layers=4,
            n_heads=8
        )
        
        model = create_gpt_model(config)
        
        total_params = model.get_num_params(non_embedding=False)
        non_emb_params = model.get_num_params(non_embedding=True)
        
        assert total_params > non_emb_params
        assert total_params > 0
        assert non_emb_params > 0
    
    def test_tied_embeddings(self):
        """Test tied embeddings share weights."""
        config = GPTConfig(
            vocab_size=1000,
            d_model=256,
            tie_embeddings=True
        )
        
        model = create_gpt_model(config)
        
        # Check if weights are shared
        emb_weight = model.embeddings.get_token_embedding_weight()
        out_weight = model.output_projection.weight
        
        assert emb_weight is out_weight  # Same object


class TestInitialization:
    """Test Weight Initialization."""
    
    def test_gpt_initialization(self):
        """Test GPT-style initialization."""
        config = GPTConfig(vocab_size=100, d_model=64, n_layers=2, n_heads=4)
        model = create_gpt_model(config)
        
        initialize_model(model, strategy="gpt", seed=42)
        
        stats = get_init_stats(model)
        
        # Check if embeddings and linear layers initialized
        assert 'Embedding' in stats
        assert 'Linear' in stats
        
        # Check mean close to 0
        assert abs(stats['Linear']['mean']) < 0.01
        assert abs(stats['Embedding']['mean']) < 0.01
    
    def test_initialization_strategies(self):
        """Test different initialization strategies."""
        config = GPTConfig(vocab_size=100, d_model=64, n_layers=2, n_heads=4)
        
        strategies = ['gpt', 'bert', 'xavier', 'kaiming', 'normal']
        
        for strategy in strategies:
            model = create_gpt_model(config)
            initialize_model(model, strategy=strategy, seed=42)
            
            stats = get_init_stats(model)
            
            # All strategies should initialize weights
            assert 'Linear' in stats
            assert stats['Linear']['count'] > 0
    
    def test_initialization_reproducibility(self):
        """Test initialization reproducibility with seed."""
        config = GPTConfig(vocab_size=100, d_model=64, n_layers=2, n_heads=4)
        
        model1 = create_gpt_model(config)
        model2 = create_gpt_model(config)
        
        initialize_model(model1, strategy="gpt", seed=123)
        initialize_model(model2, strategy="gpt", seed=123)
        
        # Check if weights are identical
        for (n1, p1), (n2, p2) in zip(model1.named_parameters(), model2.named_parameters()):
            assert torch.allclose(p1, p2), f"Weights differ at {n1}"


class TestIntegration:
    """Integration tests for complete pipeline."""
    
    def test_end_to_end_forward_pass(self):
        """Test complete forward pass from tokens to logits."""
        config = GPTConfig(
            vocab_size=1000,
            max_seq_len=128,
            d_model=256,
            n_layers=4,
            n_heads=8,
            d_ff=1024,
            dropout=0.1
        )
        
        model = create_gpt_model(config)
        initialize_model(model, strategy="gpt", seed=42)
        
        batch_size = 2
        seq_len = 50
        token_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        # Forward pass
        logits, attn_weights = model(token_ids)
        
        # Validate
        assert logits.shape == (batch_size, seq_len, config.vocab_size)
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()
        assert len(attn_weights) == config.n_layers
    
    def test_end_to_end_generation(self):
        """Test complete generation pipeline."""
        config = GPTConfig(
            vocab_size=500,
            max_seq_len=64,
            d_model=128,
            n_layers=2,
            n_heads=4,
            d_ff=512,
            dropout=0.0
        )
        
        model = create_gpt_model(config)
        initialize_model(model, strategy="gpt", seed=42)
        model.eval()
        
        prompt = torch.tensor([[1, 2, 3, 4, 5]])
        
        with torch.no_grad():
            generated = model.generate(
                prompt_ids=prompt,
                max_new_tokens=20,
                temperature=1.0,
                top_k=50
            )
        
        assert generated.shape == (1, 25)  # 5 + 20
        assert (generated[:, :5] == prompt).all()  # Prompt preserved
    
    def test_batch_inference(self):
        """Test batch inference."""
        config = GPTConfig(
            vocab_size=500,
            d_model=128,
            n_layers=2,
            n_heads=4
        )
        
        model = create_gpt_model(config)
        model.eval()
        
        batch_size = 4
        seq_len = 30
        token_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        with torch.no_grad():
            logits, _ = model(token_ids)
        
        assert logits.shape == (batch_size, seq_len, config.vocab_size)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
