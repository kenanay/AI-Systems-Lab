"""
tests/test_memory_optimizations.py

Unit and benchmark tests for PyTorch memory and training optimizations:
1. Scaled Dot-Product Attention (SDPA): FlashAttention/Cutlass fused kernels,
   numerical equivalence, causal & padding mask handling, backward compatibility.
2. Gradient Checkpointing: Activation recomputation, gradient equivalence with
   standard backprop, LoRA compatibility, dynamic enable/disable.
3. Memory efficiency and contract validation.
"""

import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.attention import (
    ScaledDotProductAttention,
    MultiHeadAttention,
    create_attention_layer,
    create_causal_mask,
)
from src.model.transformer_block import (
    TransformerDecoderBlock,
    TransformerDecoder,
    create_transformer_decoder,
)
from src.model.gpt import GPTModel, GPTConfig
from src.training.lora import LoRAConfig, add_lora_to_model


class TestSDPA:
    """Tests for PyTorch Scaled Dot-Product Attention (SDPA) integration."""

    def test_sdpa_numerical_equivalence_with_manual(self):
        """Verify that SDPA produces numerically equivalent outputs to manual O(T^2) attention."""
        torch.manual_seed(42)
        batch_size, n_heads, seq_len, d_k = 2, 4, 16, 32

        q = torch.randn(batch_size, n_heads, seq_len, d_k)
        k = torch.randn(batch_size, n_heads, seq_len, d_k)
        v = torch.randn(batch_size, n_heads, seq_len, d_k)

        attn_manual = ScaledDotProductAttention(dropout=0.0, use_sdpa=False)
        attn_sdpa = ScaledDotProductAttention(dropout=0.0, use_sdpa=True)

        out_manual, weights_manual = attn_manual(q, k, v, need_weights=True)
        out_sdpa, weights_sdpa = attn_sdpa(q, k, v, need_weights=False)

        assert weights_manual is not None
        assert weights_sdpa is None  # SDPA does not materialize attention matrix
        assert out_sdpa is not None
        assert torch.allclose(out_manual, out_sdpa, atol=1e-5), (
            f"Max absolute difference: {(out_manual - out_sdpa).abs().max().item()}"
        )

    def test_sdpa_causal_masking_equivalence(self):
        """Verify causal masking with hardware is_causal produces identical output to explicit mask."""
        torch.manual_seed(123)
        batch_size, n_heads, seq_len, d_k = 2, 4, 20, 32

        q = torch.randn(batch_size, n_heads, seq_len, d_k)
        k = torch.randn(batch_size, n_heads, seq_len, d_k)
        v = torch.randn(batch_size, n_heads, seq_len, d_k)

        # Causal mask with True for masked positions
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1).unsqueeze(0).unsqueeze(0)

        attn_manual = ScaledDotProductAttention(dropout=0.0, use_sdpa=False)
        attn_sdpa = ScaledDotProductAttention(dropout=0.0, use_sdpa=True)

        out_manual, _ = attn_manual(q, k, v, mask=mask, need_weights=True)
        out_sdpa, _ = attn_sdpa(q, k, v, mask=None, is_causal=True, need_weights=False)

        assert torch.allclose(out_manual, out_sdpa, atol=1e-5), (
            f"Max diff: {(out_manual - out_sdpa).abs().max().item()}"
        )

    def test_sdpa_custom_boolean_mask(self):
        """Verify SDPA with arbitrary boolean mask matches manual calculation."""
        torch.manual_seed(999)
        batch_size, n_heads, seq_len, d_k = 2, 2, 8, 16

        q = torch.randn(batch_size, n_heads, seq_len, d_k)
        k = torch.randn(batch_size, n_heads, seq_len, d_k)
        v = torch.randn(batch_size, n_heads, seq_len, d_k)

        # Arbitrary mask (e.g. padding tokens at end)
        mask = torch.zeros(batch_size, 1, seq_len, seq_len, dtype=torch.bool)
        mask[:, :, :, -2:] = True  # Mask last 2 positions

        attn_manual = ScaledDotProductAttention(dropout=0.0, use_sdpa=False)
        attn_sdpa = ScaledDotProductAttention(dropout=0.0, use_sdpa=True)

        out_manual, _ = attn_manual(q, k, v, mask=mask, need_weights=True)
        out_sdpa, _ = attn_sdpa(q, k, v, mask=mask, need_weights=False)

        assert torch.allclose(out_manual, out_sdpa, atol=1e-5)

    def test_multihead_attention_with_sdpa(self):
        """Verify MultiHeadAttention works seamlessly with SDPA enabled."""
        mha = create_attention_layer(d_model=64, n_heads=4, dropout=0.0, use_sdpa=True)
        x = torch.randn(2, 12, 64)

        # need_weights=False triggers SDPA
        out, weights, k_cache, v_cache = mha(x, need_weights=False, is_causal=True)
        assert out.shape == (2, 12, 64)
        assert weights is None

        # need_weights=True returns explicit weights
        out_w, weights_w, _, _ = mha(x, need_weights=True, is_causal=True)
        assert out_w.shape == (2, 12, 64)
        assert weights_w is not None
        assert weights_w.shape == (2, 4, 12, 12)
        assert torch.allclose(out, out_w, atol=1e-5)


class TestGradientCheckpointing:
    """Tests for activation gradient checkpointing in Transformer decoder."""

    def test_checkpointing_forward_equivalence(self):
        """Verify that enabling gradient checkpointing does not alter forward pass logits."""
        torch.manual_seed(42)
        config = GPTConfig(
            vocab_size=100,
            max_seq_len=64,
            d_model=64,
            n_layers=3,
            n_heads=4,
            d_ff=128,
            dropout=0.0,
            use_sdpa=True,
            gradient_checkpointing=False,
        )

        model = GPTModel(config)
        model.eval()

        token_ids = torch.randint(0, 100, (2, 16))

        with torch.no_grad():
            logits_no_chk, _ = model(token_ids)
            model.gradient_checkpointing_enable()
            logits_chk, _ = model(token_ids)

        assert torch.allclose(logits_no_chk, logits_chk, atol=1e-6)

    def test_checkpointing_exact_backward_gradients(self):
        """Verify that backward pass with checkpointing produces identical gradients to standard pass."""
        torch.manual_seed(42)
        config1 = GPTConfig(
            vocab_size=120,
            max_seq_len=32,
            d_model=64,
            n_layers=3,
            n_heads=4,
            d_ff=128,
            dropout=0.0,
            use_sdpa=True,
            gradient_checkpointing=False,
        )

        torch.manual_seed(42)
        model_standard = GPTModel(config1)
        model_standard.train()

        torch.manual_seed(42)
        config2 = GPTConfig(
            vocab_size=120,
            max_seq_len=32,
            d_model=64,
            n_layers=3,
            n_heads=4,
            d_ff=128,
            dropout=0.0,
            use_sdpa=True,
            gradient_checkpointing=True,
        )
        model_chk = GPTModel(config2)
        model_chk.train()

        token_ids = torch.randint(0, 120, (2, 16))
        targets = torch.randint(0, 120, (2, 16))

        criterion = nn.CrossEntropyLoss()

        # Forward + backward on standard model
        logits_std, _ = model_standard(token_ids)
        loss_std = criterion(logits_std.view(-1, 120), targets.view(-1))
        loss_std.backward()

        # Forward + backward on checkpointed model
        logits_chk, _ = model_chk(token_ids)
        loss_chk = criterion(logits_chk.view(-1, 120), targets.view(-1))
        loss_chk.backward()

        assert torch.allclose(loss_std, loss_chk, atol=1e-5)

        # Compare gradients across all parameters
        matched_params = 0
        for (n1, p1), (n2, p2) in zip(model_standard.named_parameters(), model_chk.named_parameters()):
            assert n1 == n2
            if p1.grad is not None and p2.grad is not None:
                assert torch.allclose(p1.grad, p2.grad, atol=1e-5), (
                    f"Gradient mismatch in parameter {n1}: max diff {(p1.grad - p2.grad).abs().max().item()}"
                )
                matched_params += 1

        assert matched_params > 0, "No parameter gradients were verified"

    def test_gradient_checkpointing_toggle(self):
        """Test enabling and disabling gradient checkpointing dynamically."""
        config = GPTConfig(vocab_size=50, max_seq_len=16, d_model=32, n_layers=2, n_heads=2, d_ff=64)
        model = GPTModel(config)

        assert not model.config.gradient_checkpointing
        assert not model.decoder.gradient_checkpointing

        model.gradient_checkpointing_enable()
        assert model.config.gradient_checkpointing
        assert model.decoder.gradient_checkpointing

        model.gradient_checkpointing_disable()
        assert not model.config.gradient_checkpointing
        assert not model.decoder.gradient_checkpointing

    def test_gradient_checkpointing_with_lora(self):
        """Verify gradient checkpointing works seamlessly when LoRA adapter is applied."""
        config = GPTConfig(
            vocab_size=80,
            max_seq_len=24,
            d_model=64,
            n_layers=2,
            n_heads=4,
            d_ff=128,
            dropout=0.0,
            gradient_checkpointing=True,
        )
        model = GPTModel(config)
        lora_cfg = LoRAConfig(rank=4, alpha=8.0, r=4, lora_alpha=8.0, target_modules=['w_q', 'w_v'])
        model = add_lora_to_model(model, lora_cfg, verbose=False)
        model.train()

        tokens = torch.randint(0, 80, (2, 12))
        logits, _ = model(tokens)
        loss = logits.sum()
        loss.backward()

        # Verify LoRA parameters received valid gradients
        lora_grads = [p.grad for n, p in model.named_parameters() if "lora" in n.lower() and p.requires_grad]
        assert len(lora_grads) > 0
        assert all(g is not None and not torch.isnan(g).any() for g in lora_grads)


class TestMemoryAndContract:
    """Verify architectural contracts and memory savings."""

    def test_gpt_forward_attention_weights_contract(self):
        """Verify that model returns a list of length n_layers in all modes."""
        config = GPTConfig(vocab_size=60, max_seq_len=16, d_model=32, n_layers=4, n_heads=2, d_ff=64, use_sdpa=True)
        model = GPTModel(config)
        tokens = torch.randint(0, 60, (2, 8))

        # Default mode (SDPA enabled): attention_weights list of length n_layers with None elements
        logits, attn_weights = model(tokens)
        assert logits.shape == (2, 8, 60)
        assert len(attn_weights) == 4
        assert all(w is None for w in attn_weights)

        # Explicit need_weights=True: returns list of length n_layers with [B, H, T, T] tensors
        logits_w, attn_weights_w = model(tokens, need_weights=True)
        assert len(attn_weights_w) == 4
        assert all(w is not None for w in attn_weights_w)
        assert attn_weights_w[0].shape == (2, 2, 8, 8)

    def test_config_to_dict_includes_new_optimizations(self):
        """Verify to_dict exposes use_sdpa and gradient_checkpointing."""
        config = GPTConfig(use_sdpa=True, gradient_checkpointing=True)
        d = config.to_dict()
        assert "use_sdpa" in d and d["use_sdpa"] is True
        assert "gradient_checkpointing" in d and d["gradient_checkpointing"] is True

    def test_training_step_with_both_optimizations(self):
        """Verify an entire training step (forward + loss + backward + optimizer step) succeeds with both SDPA and checkpointing."""
        config = GPTConfig(
            vocab_size=100,
            max_seq_len=64,
            d_model=64,
            n_layers=4,
            n_heads=4,
            d_ff=128,
            use_sdpa=True,
            gradient_checkpointing=True,
        )
        model = GPTModel(config)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        tokens = torch.randint(0, 100, (4, 32))
        targets = torch.randint(0, 100, (4, 32))

        optimizer.zero_grad()
        logits, attn = model(tokens)
        loss = criterion(logits.view(-1, 100), targets.view(-1))
        loss.backward()
        optimizer.step()

        assert loss.item() > 0.0
        assert not torch.isnan(loss)
        assert len(attn) == 4
        assert all(w is None for w in attn)
