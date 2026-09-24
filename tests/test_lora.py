"""
Tests for LoRA (Low-Rank Adaptation) implementation.

Bu test suite, LoRA implementasyonunun doğruluğunu test eder:
- LoRALayer initialization ve forward pass
- LinearWithLoRA wrapping ve merge/unmerge
- Model integration (add_lora_to_model)
- Save/load functionality
- Utility functions

Test Coverage:
- LoRALayer: initialization, forward, shapes
- LinearWithLoRA: forward, merge, unmerge, freezing
- Model integration: add_lora_to_model, parameter filtering
- Save/Load: save_lora_weights, load_lora_weights
- Utilities: get_lora_parameters, merge/unmerge helpers

Version: 1.0.0
"""

import pytest
import torch
import torch.nn as nn
import tempfile
import os
from pathlib import Path

# Import LoRA components
from src.training.lora import (
    LoRALayer,
    LinearWithLoRA,
    LoRAConfig,
    add_lora_to_model,
    get_lora_parameters,
    merge_lora_weights,
    unmerge_lora_weights,
    save_lora_weights,
    load_lora_weights,
    print_trainable_parameters
)


# ============================================================================
# Test LoRALayer
# ============================================================================

class TestLoRALayer:
    """Test LoRALayer class."""
    
    def test_lora_layer_initialization(self):
        """Test LoRALayer initialization."""
        in_features = 128
        out_features = 256
        rank = 8
        alpha = 16.0
        
        layer = LoRALayer(
            in_features=in_features,
            out_features=out_features,
            rank=rank,
            alpha=alpha
        )
        
        # Check attributes
        assert layer.in_features == in_features
        assert layer.out_features == out_features
        assert layer.rank == rank
        assert layer.alpha == alpha
        assert layer.scaling == alpha / rank  # 16 / 8 = 2.0
        
        # Check parameter shapes
        assert layer.lora_A.shape == (rank, in_features)  # [8, 128]
        assert layer.lora_B.shape == (out_features, rank)  # [256, 8]
        
        # Check B is initialized to zeros
        assert torch.allclose(layer.lora_B, torch.zeros_like(layer.lora_B))
        
        # Check A is not zeros (Kaiming init)
        assert not torch.allclose(layer.lora_A, torch.zeros_like(layer.lora_A))
    
    def test_lora_layer_forward_shape(self):
        """Test LoRALayer forward pass shape."""
        batch_size = 4
        seq_len = 10
        in_features = 128
        out_features = 256
        rank = 8
        
        layer = LoRALayer(
            in_features=in_features,
            out_features=out_features,
            rank=rank
        )
        
        # Input: [batch, seq_len, in_features]
        x = torch.randn(batch_size, seq_len, in_features)
        
        # Forward
        output = layer(x)
        
        # Check output shape: [batch, seq_len, out_features]
        assert output.shape == (batch_size, seq_len, out_features)
    
    def test_lora_layer_forward_computation(self):
        """Test LoRALayer forward computation correctness."""
        batch_size = 2
        seq_len = 3
        in_features = 4
        out_features = 5
        rank = 2
        alpha = 8.0
        
        layer = LoRALayer(
            in_features=in_features,
            out_features=out_features,
            rank=rank,
            alpha=alpha,
            dropout=0.0  # No dropout for deterministic test
        )
        
        # Set known values for testing
        layer.lora_A.data = torch.ones(rank, in_features)  # [2, 4]
        layer.lora_B.data = torch.ones(out_features, rank)  # [5, 2]
        
        # Input
        x = torch.ones(batch_size, seq_len, in_features)  # [2, 3, 4]
        
        # Forward
        output = layer(x)
        
        # Manual computation:
        # A projection: x @ A.T = [2,3,4] @ [4,2] = [2,3,2]
        #   Each element = sum of 4 ones = 4
        # B projection: result @ B.T = [2,3,2] @ [2,5] = [2,3,5]
        #   Each element = 2 * 4 = 8
        # Scaling: 8 * (alpha/rank) = 8 * (8/2) = 8 * 4 = 32
        
        expected_value = 32.0
        assert torch.allclose(output, torch.full_like(output, expected_value))
    
    def test_lora_layer_dropout(self):
        """Test LoRALayer with dropout."""
        layer = LoRALayer(
            in_features=128,
            out_features=256,
            rank=8,
            dropout=0.5
        )
        
        # Check dropout layer exists
        assert isinstance(layer.dropout, nn.Dropout)
        
        # With dropout=0, should be Identity
        layer_no_dropout = LoRALayer(
            in_features=128,
            out_features=256,
            rank=8,
            dropout=0.0
        )
        assert isinstance(layer_no_dropout.dropout, nn.Identity)


# ============================================================================
# Test LinearWithLoRA
# ============================================================================

class TestLinearWithLoRA:
    """Test LinearWithLoRA class."""
    
    def test_linear_with_lora_initialization(self):
        """Test LinearWithLoRA initialization."""
        in_features = 128
        out_features = 256
        
        # Create base linear layer
        base_layer = nn.Linear(in_features, out_features)
        
        # Wrap with LoRA
        lora_layer = LinearWithLoRA(
            base_layer=base_layer,
            rank=8,
            alpha=16.0
        )
        
        # Check base layer is frozen
        for param in lora_layer.base_layer.parameters():
            assert not param.requires_grad
        
        # Check LoRA adapter exists
        assert isinstance(lora_layer.lora, LoRALayer)
        assert lora_layer.lora.in_features == in_features
        assert lora_layer.lora.out_features == out_features
        
        # Check merge state
        assert lora_layer.merged == False
    
    def test_linear_with_lora_forward_unmerged(self):
        """Test LinearWithLoRA forward pass (unmerged)."""
        batch_size = 4
        seq_len = 10
        in_features = 128
        out_features = 256
        
        base_layer = nn.Linear(in_features, out_features)
        lora_layer = LinearWithLoRA(base_layer, rank=8)
        
        # Input
        x = torch.randn(batch_size, seq_len, in_features)
        
        # Forward (unmerged: W₀x + ΔWx)
        output = lora_layer(x)
        
        # Check shape
        assert output.shape == (batch_size, seq_len, out_features)
        
        # Verify computation: output = base_output + lora_output
        with torch.no_grad():
            base_output = base_layer(x)
            lora_output = lora_layer.lora(x)
            expected_output = base_output + lora_output
        
        assert torch.allclose(output, expected_output)
    
    def test_linear_with_lora_merge_unmerge(self):
        """Test merge and unmerge functionality."""
        in_features = 64
        out_features = 128
        
        base_layer = nn.Linear(in_features, out_features)
        
        # Save original weights
        original_weight = base_layer.weight.data.clone()
        
        lora_layer = LinearWithLoRA(base_layer, rank=4, alpha=8.0)
        
        # Set non-zero LoRA weights for meaningful test
        # (B is initialized to zeros by default)
        lora_layer.lora.lora_B.data = torch.randn_like(lora_layer.lora.lora_B.data) * 0.01
        
        # Test input
        x = torch.randn(2, 5, in_features)
        
        # Get output before merge
        output_before = lora_layer(x)
        
        # Merge weights
        lora_layer.merge()
        assert lora_layer.merged == True
        
        # After merge, weight should be different (now we have non-zero LoRA)
        assert not torch.allclose(base_layer.weight.data, original_weight)
        
        # Get output after merge
        output_after_merge = lora_layer(x)
        
        # Outputs should be similar (forward behavior same)
        assert torch.allclose(output_before, output_after_merge, atol=1e-5)
        
        # Unmerge weights
        lora_layer.unmerge()
        assert lora_layer.merged == False
        
        # After unmerge, weight should be back to original
        assert torch.allclose(base_layer.weight.data, original_weight)
        
        # Output after unmerge should match original
        output_after_unmerge = lora_layer(x)
        assert torch.allclose(output_before, output_after_unmerge, atol=1e-5)
    
    def test_linear_with_lora_double_merge_warning(self):
        """Test warning on double merge."""
        base_layer = nn.Linear(64, 128)
        lora_layer = LinearWithLoRA(base_layer, rank=4)
        
        # First merge - OK
        lora_layer.merge()
        
        # Second merge - should log warning (no exception)
        lora_layer.merge()  # Should not raise
        
        assert lora_layer.merged == True
    
    def test_linear_with_lora_unmerge_without_merge(self):
        """Test unmerge without prior merge."""
        base_layer = nn.Linear(64, 128)
        lora_layer = LinearWithLoRA(base_layer, rank=4)
        
        # Unmerge without merge - should log warning
        lora_layer.unmerge()  # Should not raise
        
        assert lora_layer.merged == False


# ============================================================================
# Test LoRAConfig
# ============================================================================

class TestLoRAConfig:
    """Test LoRAConfig class."""
    
    def test_lora_config_initialization(self):
        """Test LoRAConfig initialization."""
        config = LoRAConfig(
            rank=16,
            alpha=32.0,
            dropout=0.1,
            target_modules=['q_proj', 'v_proj', 'k_proj'],
            merge_weights=True
        )
        
        assert config.rank == 16
        assert config.alpha == 32.0
        assert config.dropout == 0.1
        assert config.target_modules == ['q_proj', 'v_proj', 'k_proj']
        assert config.merge_weights == True
    
    def test_lora_config_defaults(self):
        """Test LoRAConfig default values."""
        config = LoRAConfig()
        
        assert config.rank == 8
        assert config.alpha == 16.0
        assert config.dropout == 0.0
        assert config.target_modules == ['q_proj', 'v_proj']
        assert config.merge_weights == False
    
    def test_lora_config_repr(self):
        """Test LoRAConfig string representation."""
        config = LoRAConfig(rank=8, alpha=16.0)
        
        repr_str = repr(config)
        assert 'LoRAConfig' in repr_str
        assert 'rank=8' in repr_str
        assert 'alpha=16.0' in repr_str


# ============================================================================
# Test Model Integration
# ============================================================================

class TestModelIntegration:
    """Test LoRA integration with models."""
    
    def create_simple_model(self):
        """Create a simple model for testing."""
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(128, 128)
                self.k_proj = nn.Linear(128, 128)
                self.v_proj = nn.Linear(128, 128)
                self.out_proj = nn.Linear(128, 128)
            
            def forward(self, x):
                q = self.q_proj(x)
                k = self.k_proj(x)
                v = self.v_proj(x)
                return self.out_proj(q + k + v)
        
        return SimpleModel()
    
    def test_add_lora_to_model(self):
        """Test adding LoRA to model."""
        model = self.create_simple_model()
        
        # Count original parameters
        original_trainable = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        
        # Add LoRA
        config = LoRAConfig(
            rank=4,
            alpha=8.0,
            target_modules=['q_proj', 'v_proj']
        )
        model = add_lora_to_model(model, config, verbose=False)
        
        # Check q_proj and v_proj are wrapped
        assert isinstance(model.q_proj, LinearWithLoRA)
        assert isinstance(model.v_proj, LinearWithLoRA)
        
        # Check k_proj and out_proj are NOT wrapped (not in target_modules)
        assert isinstance(model.k_proj, nn.Linear)
        assert isinstance(model.out_proj, nn.Linear)
        
        # Count trainable parameters after LoRA
        lora_trainable = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        
        # LoRA should have fewer trainable params
        assert lora_trainable < original_trainable
    
    def test_add_lora_parameter_reduction(self):
        """Test parameter reduction with LoRA."""
        model = self.create_simple_model()
        
        # Total params before
        total_before = sum(p.numel() for p in model.parameters())
        trainable_before = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        
        # Add LoRA to all projections
        config = LoRAConfig(
            rank=4,
            target_modules=['q_proj', 'k_proj', 'v_proj', 'out_proj']
        )
        model = add_lora_to_model(model, config, verbose=False)
        
        # Total params after (should be more due to LoRA matrices)
        total_after = sum(p.numel() for p in model.parameters())
        trainable_after = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        
        # Trainable params should be much less
        # Each layer: 128x128 = 16384 params
        # LoRA: (128x4) + (4x128) = 512 + 512 = 1024 params
        # Reduction: ~16x per layer
        
        assert trainable_after < trainable_before * 0.2  # At least 5x reduction
        
        # But total params increased (LoRA params added)
        assert total_after > total_before
    
    def test_add_lora_forward_pass(self):
        """Test model forward pass after adding LoRA."""
        model = self.create_simple_model()
        
        config = LoRAConfig(rank=4, target_modules=['q_proj', 'v_proj'])
        model = add_lora_to_model(model, config, verbose=False)
        
        # Test forward pass
        x = torch.randn(2, 10, 128)  # [batch, seq_len, features]
        
        output = model(x)
        
        # Check output shape
        assert output.shape == (2, 10, 128)
        
        # Check no NaN or Inf
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


# ============================================================================
# Test Get LoRA Parameters
# ============================================================================

class TestGetLoRAParameters:
    """Test get_lora_parameters function."""
    
    def test_get_lora_parameters(self):
        """Test getting LoRA parameters from model."""
        # Create model
        model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        # Add LoRA to first linear layer only
        base_layer = model[0]
        model[0] = LinearWithLoRA(base_layer, rank=4)
        
        # Get LoRA parameters
        lora_params = get_lora_parameters(model)
        
        # Should have 2 parameters (A and B)
        assert len(lora_params) == 2
        
        # Check they are the correct parameters
        assert lora_params[0].shape == (4, 64)   # lora_A: [rank, in_features]
        assert lora_params[1].shape == (128, 4)  # lora_B: [out_features, rank]
    
    def test_get_lora_parameters_multiple_layers(self):
        """Test getting LoRA parameters from model with multiple LoRA layers."""
        model = nn.Sequential(
            nn.Linear(64, 128),
            nn.Linear(128, 128),
            nn.Linear(128, 64)
        )
        
        # Add LoRA to all layers
        for i in range(3):
            base_layer = model[i]
            model[i] = LinearWithLoRA(base_layer, rank=4)
        
        # Get LoRA parameters
        lora_params = get_lora_parameters(model)
        
        # Should have 6 parameters (2 per layer × 3 layers)
        assert len(lora_params) == 6


# ============================================================================
# Test Merge/Unmerge Weights
# ============================================================================

class TestMergeUnmergeWeights:
    """Test merge_lora_weights and unmerge_lora_weights functions."""
    
    def test_merge_lora_weights(self):
        """Test merging all LoRA weights in model."""
        model = nn.Sequential(
            nn.Linear(64, 128),
            nn.Linear(128, 64)
        )
        
        # Add LoRA
        for i in range(2):
            base_layer = model[i]
            model[i] = LinearWithLoRA(base_layer, rank=4)
        
        # Initially not merged
        assert model[0].merged == False
        assert model[1].merged == False
        
        # Merge all
        merge_lora_weights(model)
        
        # Now merged
        assert model[0].merged == True
        assert model[1].merged == True
    
    def test_unmerge_lora_weights(self):
        """Test unmerging all LoRA weights in model."""
        model = nn.Sequential(
            nn.Linear(64, 128),
            nn.Linear(128, 64)
        )
        
        # Add LoRA and merge
        for i in range(2):
            base_layer = model[i]
            model[i] = LinearWithLoRA(base_layer, rank=4)
        
        merge_lora_weights(model)
        
        # Unmerge all
        unmerge_lora_weights(model)
        
        # Now unmerged
        assert model[0].merged == False
        assert model[1].merged == False


# ============================================================================
# Test Save/Load Weights
# ============================================================================

class TestSaveLoadWeights:
    """Test save_lora_weights and load_lora_weights functions."""
    
    def test_save_lora_weights(self):
        """Test saving LoRA weights."""
        model = nn.Sequential(
            nn.Linear(64, 128),
            nn.Linear(128, 64)
        )
        
        # Add LoRA
        for i in range(2):
            base_layer = model[i]
            model[i] = LinearWithLoRA(base_layer, rank=4)
        
        # Save to temp file
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, 'lora_weights.pt')
            
            save_lora_weights(model, save_path)
            
            # Check file exists
            assert os.path.exists(save_path)
            
            # Load and check content
            state_dict = torch.load(save_path, weights_only=True)
            
            # Should have 4 keys (2 layers × 2 matrices)
            assert len(state_dict) == 4
            assert '0.lora_A' in state_dict
            assert '0.lora_B' in state_dict
            assert '1.lora_A' in state_dict
            assert '1.lora_B' in state_dict
    
    def test_load_lora_weights(self):
        """Test loading LoRA weights."""
        # Create original model
        model1 = nn.Sequential(
            nn.Linear(64, 128),
            nn.Linear(128, 64)
        )
        
        # Add LoRA
        for i in range(2):
            base_layer = model1[i]
            model1[i] = LinearWithLoRA(base_layer, rank=4)
        
        # Save weights
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, 'lora_weights.pt')
            save_lora_weights(model1, save_path)
            
            # Create new model with different LoRA weights
            model2 = nn.Sequential(
                nn.Linear(64, 128),
                nn.Linear(128, 64)
            )
            
            for i in range(2):
                base_layer = model2[i]
                model2[i] = LinearWithLoRA(base_layer, rank=4)
            
            # Verify weights are different initially
            assert not torch.allclose(
                model1[0].lora.lora_A,
                model2[0].lora.lora_A
            )
            
            # Load weights from model1 into model2
            load_lora_weights(model2, save_path)
            
            # Now weights should match
            assert torch.allclose(
                model1[0].lora.lora_A,
                model2[0].lora.lora_A
            )
            assert torch.allclose(
                model1[0].lora.lora_B,
                model2[0].lora.lora_B
            )
            assert torch.allclose(
                model1[1].lora.lora_A,
                model2[1].lora.lora_A
            )
            assert torch.allclose(
                model1[1].lora.lora_B,
                model2[1].lora.lora_B
            )
    
    def test_save_load_weights_size_efficiency(self):
        """Test that saving only LoRA weights is storage efficient."""
        model = nn.Sequential(
            nn.Linear(512, 512),  # Large layer: 262K params
            nn.Linear(512, 512)
        )
        
        # Add LoRA with small rank
        for i in range(2):
            base_layer = model[i]
            model[i] = LinearWithLoRA(base_layer, rank=8)  # Only 8K params per layer
        
        with tempfile.TemporaryDirectory() as tmpdir:
            lora_path = os.path.join(tmpdir, 'lora_only.pt')
            full_path = os.path.join(tmpdir, 'full_model.pt')
            
            # Save only LoRA weights
            save_lora_weights(model, lora_path)
            
            # Save full model
            torch.save(model.state_dict(), full_path)
            
            # LoRA file should be much smaller
            lora_size = os.path.getsize(lora_path)
            full_size = os.path.getsize(full_path)
            
            # LoRA should be < 10% of full model size
            assert lora_size < full_size * 0.1


# ============================================================================
# Test Print Trainable Parameters
# ============================================================================

class TestPrintTrainableParameters:
    """Test print_trainable_parameters function."""
    
    def test_print_trainable_parameters(self, capsys):
        """Test printing trainable parameter statistics."""
        model = nn.Sequential(
            nn.Linear(64, 128),
            nn.Linear(128, 64)
        )
        
        # Add LoRA
        for i in range(2):
            base_layer = model[i]
            model[i] = LinearWithLoRA(base_layer, rank=4)
        
        # Print stats
        print_trainable_parameters(model)
        
        # Capture output
        captured = capsys.readouterr()
        
        # Check output contains expected strings
        assert 'Trainable params:' in captured.out
        assert 'All params:' in captured.out
        assert 'Trainable %:' in captured.out


# ============================================================================
# Integration Test
# ============================================================================

class TestLoRAIntegration:
    """End-to-end integration test."""
    
    def test_full_lora_workflow(self):
        """Test complete LoRA workflow."""
        # 1. Create model
        model = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
        # 2. Add LoRA
        config = LoRAConfig(rank=8, target_modules=['0', '2'])
        model = add_lora_to_model(model, config, verbose=False)
        
        # 3. Check trainable params reduced
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        assert trainable < total * 0.3  # <30% trainable
        
        # 4. Forward pass
        x = torch.randn(4, 10, 128)
        output = model(x)
        assert output.shape == (4, 10, 128)
        
        # 5. Save LoRA weights
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, 'lora.pt')
            save_lora_weights(model, save_path)
            assert os.path.exists(save_path)
            
            # 6. Create new model with SAME base weights
            model2 = nn.Sequential(
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, 128)
            )
            
            # Copy base weights from model1 to model2 for consistent test
            model2[0].weight.data = model[0].base_layer.weight.data.clone()
            model2[0].bias.data = model[0].base_layer.bias.data.clone()
            model2[2].weight.data = model[2].base_layer.weight.data.clone()
            model2[2].bias.data = model[2].base_layer.bias.data.clone()
            
            # Add LoRA to model2
            model2 = add_lora_to_model(model2, config, verbose=False)
            
            # Load LoRA weights from model1
            load_lora_weights(model2, save_path)
            
            # 7. Verify outputs match (same base + same LoRA = same output)
            output2 = model2(x)
            assert torch.allclose(output, output2, atol=1e-5)
        
        # 8. Test merge/unmerge
        merge_lora_weights(model)
        output_merged = model(x)
        assert torch.allclose(output, output_merged, atol=1e-5)
        
        unmerge_lora_weights(model)
        output_unmerged = model(x)
        assert torch.allclose(output, output_unmerged, atol=1e-5)


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
