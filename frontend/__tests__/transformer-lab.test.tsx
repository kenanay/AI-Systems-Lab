import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import TransformerLabPage from '@/app/transformer-lab/page';
import { transformerLabApi } from '@/lib/api';

// Mock transformerLabApi
jest.mock('@/lib/api', () => {
  const original = jest.requireActual('@/lib/api');
  return {
    ...original,
    transformerLabApi: {
      getRoPE: jest.fn(() =>
        Promise.resolve({
          dim: 16,
          max_seq_len: 8,
          base: 10000.0,
          theta_values: [1.0, 0.316],
          positions_data: [
            { position: 0, angle_rad: 0, angle_deg: 0, q_rotated: [1, 0], k_rotated: [0.8, 0.6] },
            { position: 1, angle_rad: 1, angle_deg: 57.3, q_rotated: [0.54, 0.84], k_rotated: [0.3, 0.9] },
          ],
          relative_dot_matrix: [[1.0, 0.8], [0.8, 1.0]],
          explanation: 'RoPE preserves relative distance',
        })
      ),
      getSinusoidalPE: jest.fn(() =>
        Promise.resolve({
          seq_len: 16,
          d_model: 32,
          pe_matrix: [[0, 1], [0.8, 0.5]],
          similarity_matrix: [[1, 0.5], [0.5, 1]],
          waves: [
            { dimension_index: 0, values: [0, 0.8], type: 'sin', wavelength_approx: 6.3 },
          ],
          description: 'Sinusoidal encoding',
        })
      ),
      getALiBi: jest.fn(() =>
        Promise.resolve({
          num_heads: 8,
          seq_len: 8,
          slopes: [0.5, 0.25],
          heads: [
            { head_index: 0, slope: 0.5, bias_matrix: [[0, -0.5], [-0.5, 0]] },
          ],
          description: 'ALiBi slopes',
        })
      ),
      comparePE: jest.fn(() =>
        Promise.resolve([
          {
            name: 'RoPE',
            authors: 'Su et al.',
            modern_usage: 'LLaMA',
            type: 'Relative',
            parameter_overhead: '0',
            extrapolation: 'High',
            mechanism: '2D Rotation',
          },
        ])
      ),
      analyzeAttentionVariants: jest.fn(() =>
        Promise.resolve({
          variant_type: 'GQA (Grouped-Query Attention, 4:1)',
          description: 'Modern standard',
          batch_size: 2,
          seq_len: 4096,
          num_query_heads: 32,
          num_kv_heads: 8,
          head_dim: 128,
          num_layers: 32,
          dtype: 'float16',
          bytes_per_elem: 2,
          queries_per_kv_head: 4,
          current_kv_cache: {
            total_bytes: 536870912,
            formatted: '512.00 MB',
            bytes_per_layer: 16777216,
            formatted_per_layer: '16.00 MB',
          },
          comparison: {
            mha_baseline_formatted: '2.00 GB',
            mha_baseline_bytes: 2147483648,
            mqa_formatted: '64.00 MB',
            mqa_bytes: 67108864,
            savings_vs_mha_pct: 75.0,
            savings_multiplier: '4.0x',
          },
          memory_bandwidth: {
            read_per_step_formatted: '512.00 MB',
            theoretical_throughput_a100: '1953.1 tok/s',
          },
          head_groups: [
            { kv_head_index: 0, shared_query_heads: [0, 1, 2, 3], query_count: 4 },
          ],
        })
      ),
      simulateBlock: jest.fn(() =>
        Promise.resolve({
          norm_type: 'rmsnorm',
          ffn_type: 'swiglu',
          norm_placement: 'pre_ln',
          d_model: 8,
          d_ff: 16,
          stages: [
            { sublayer_name: 'Block Input x', shape: [1, 4, 8], mean: 0.05, std: 0.98, min: -1.5, max: 1.8, description: 'Input' },
            { sublayer_name: 'SwiGLU FFN Layer', shape: [1, 4, 8], mean: 0.02, std: 0.85, min: -1.2, max: 1.5, description: 'FFN' },
          ],
        })
      ),
      calculateParams: jest.fn(() =>
        Promise.resolve({
          total_parameters: 8030261248,
          total_millions: 8030.26,
          total_billions: 8.03,
          breakdown: {
            token_embeddings: 525336576,
            attention_all_layers: 2147483648,
            attention_per_layer: 67108864,
            ffn_all_layers: 5368709120,
            ffn_per_layer: 167772160,
            norms_all_layers: 266240,
            lm_head: 0,
          },
          percentages: {
            attention_pct: 26.7,
            ffn_pct: 66.8,
            embeddings_pct: 6.5,
          },
          vram_inference: {
            fp16: '14.96 GB',
            fp32: '29.91 GB',
            int8_quantized: '7.48 GB',
            int4_quantized: '3.74 GB',
          },
          vram_training_adamw: '119.66 GB',
        })
      ),
      getPresets: jest.fn(() =>
        Promise.resolve([
          {
            name: 'LLaMA-3 8B',
            vocab_size: 128256,
            d_model: 4096,
            n_layers: 32,
            n_heads: 32,
            n_kv_heads: 8,
            d_ff: 14336,
            norm_type: 'rmsnorm',
            ffn_type: 'swiglu',
            tie_word_embeddings: false,
          },
        ])
      ),
    },
  };
});

describe('TransformerLabPage', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('renders page title and initial RoPE visualization', async () => {
    render(<TransformerLabPage />);

    expect(screen.getByText('Transformer Architecture Lab')).toBeInTheDocument();
    expect(screen.getByText(/Positional Encodings \(RoPE, Sinusoidal, ALiBi\)/i)).toBeInTheDocument();
    expect(screen.getByText('1. Positional Encoding')).toBeInTheDocument();

    await waitFor(() => {
      expect(transformerLabApi.getRoPE).toHaveBeenCalled();
      expect(screen.getByText(/2D Karmaşık Düzlemde Vektör Rotasyonu/i)).toBeInTheDocument();
    });
  });

  test('switches to MHA vs GQA vs MQA tab and displays KV-cache savings', async () => {
    render(<TransformerLabPage />);

    const gqaTabBtn = screen.getByText('2. MHA vs GQA vs MQA');
    fireEvent.click(gqaTabBtn);

    await waitFor(() => {
      expect(screen.getByText(/Grouped-Query Attention \(GQA\)/i)).toBeInTheDocument();
      expect(screen.getAllByText(/512.00 MB/i).length).toBeGreaterThan(0);
      expect(screen.getByText(/%75 \(4.0x\)/i)).toBeInTheDocument();
    });
  });

  test('switches to Transformer Block tab and displays stage cards', async () => {
    render(<TransformerLabPage />);

    const blockTabBtn = screen.getByText('3. Transformer Bloğu');
    fireEvent.click(blockTabBtn);

    await waitFor(() => {
      expect(screen.getByText(/Alt Katman Akışı/i)).toBeInTheDocument();
      expect(screen.getByText('Block Input x')).toBeInTheDocument();
      expect(screen.getByText('SwiGLU FFN Layer')).toBeInTheDocument();
    });
  });

  test('switches to Model Parameters tab and displays parameter metrics', async () => {
    render(<TransformerLabPage />);

    const paramsTabBtn = screen.getByText('4. Model Parametreleri');
    fireEvent.click(paramsTabBtn);

    await waitFor(() => {
      expect(screen.getByText(/Model Parametre Dağılımı/i)).toBeInTheDocument();
      expect(screen.getByText('8.03 Milyar (B)')).toBeInTheDocument();
      expect(screen.getByText('14.96 GB')).toBeInTheDocument();
    });
  });
});
