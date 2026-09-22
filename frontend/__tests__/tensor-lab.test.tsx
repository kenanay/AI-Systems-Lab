import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import TensorLabPage from '@/app/tensor-lab/page';
import { tensorLabApi } from '@/lib/api';

// Mock tensorLabApi
jest.mock('@/lib/api', () => {
  const original = jest.requireActual('@/lib/api');
  return {
    ...original,
    tensorLabApi: {
      analyzeShape: jest.fn(() =>
        Promise.resolve({
          shape: [1, 32, 512, 64],
          rank: 4,
          total_elements: 1048576,
          dtype: 'float32',
          element_bytes: 4,
          total_bytes: 4194304,
          formatted_memory: '4.00 MB',
          element_strides: [1048576, 32768, 64, 1],
          byte_strides: [4194304, 131072, 256, 4],
          is_contiguous: true,
          semantic_interpretation: '4D Attention Tensor',
          memory_comparison: {
            float32: { bytes: 4194304, formatted: '4.00 MB', ratio_vs_fp32: 1.0 },
            float16: { bytes: 2097152, formatted: '2.00 MB', ratio_vs_fp32: 0.5 },
            int8: { bytes: 1048576, formatted: '1.00 MB', ratio_vs_fp32: 0.25 },
          },
          dtype_description: 'Standard single-precision floating point',
        })
      ),
      reshape: jest.fn(() =>
        Promise.resolve({
          success: true,
          original_shape: [1, 32, 512, 64],
          target_shape: [32, -1],
          resolved_shape: [32, 32768],
          total_elements: 1048576,
          new_strides: [32768, 1],
          contiguous: true,
          requires_copy: false,
          explanation: 'Successfully reshaped',
        })
      ),
      transpose: jest.fn(() =>
        Promise.resolve({
          success: true,
          original_shape: [1, 32, 512, 64],
          permutation: [0, 2, 1, 3],
          transposed_shape: [1, 512, 32, 64],
          transposed_strides: [1048576, 64, 32768, 1],
          is_contiguous: false,
          requires_contiguous_call: true,
          note: 'Tensor is non-contiguous',
        })
      ),
      checkBroadcast: jest.fn(() =>
        Promise.resolve({
          compatible: true,
          shape_a: [3, 1],
          shape_b: [1, 4],
          padded_a: [3, 1],
          padded_b: [1, 4],
          result_shape: [3, 4],
          steps: [
            {
              axis: -2,
              dim_a: 3,
              dim_b: 1,
              result_dim: 3,
              action: 'B has dimension 1; broadcasted',
              expansion_factor_a: 1,
              expansion_factor_b: 3,
              compatible: true,
            },
            {
              axis: -1,
              dim_a: 1,
              dim_b: 4,
              result_dim: 4,
              action: 'A has dimension 1; broadcasted',
              expansion_factor_a: 4,
              expansion_factor_b: 1,
              compatible: true,
            },
          ],
          error: null,
        })
      ),
      simulateBroadcast: jest.fn(() =>
        Promise.resolve({
          success: true,
          operation: 'add',
          matrix_a: [[1], [2], [3]],
          matrix_b: [[10, 20, 30, 40]],
          broadcasted_a: [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]],
          broadcasted_b: [[10, 20, 30, 40], [10, 20, 30, 40], [10, 20, 30, 40]],
          result_matrix: [[11, 21, 31, 41], [12, 22, 32, 42], [13, 23, 33, 43]],
        })
      ),
      getMatmulSample: jest.fn(() =>
        Promise.resolve({
          matrix_a: [[1, 2], [3, 4]],
          matrix_b: [[5, 6], [7, 8]],
        })
      ),
      matmul: jest.fn(() =>
        Promise.resolve({
          success: true,
          m: 2,
          k: 2,
          n: 2,
          shape_a: [2, 2],
          shape_b: [2, 2],
          shape_c: [2, 2],
          matrix_a: [[1, 2], [3, 4]],
          matrix_b: [[5, 6], [7, 8]],
          matrix_c: [[19, 22], [43, 50]],
          selected_cell: { row: 0, col: 0 },
          cell_value: 19,
          row_vector: [1, 2],
          col_vector: [5, 7],
          pairwise_terms: [
            { k_index: 0, a_val: 1, b_val: 5, product: 5 },
            { k_index: 1, a_val: 2, b_val: 7, product: 14 },
          ],
          formula_string: 'C[0,0] = (1.00 × 5.00) + (2.00 × 7.00) = 19.0000',
          hardware_metrics: {
            total_flops: 16,
            read_bytes_fp32: 32,
            write_bytes_fp32: 16,
            total_memory_bytes: 48,
            formatted_memory: '48 B',
            arithmetic_intensity_flops_per_byte: 0.33,
          },
        })
      ),
      getActivations: jest.fn(() =>
        Promise.resolve([
          { id: 'gelu', name: 'GELU', llm_usage: 'BERT, GPT' },
          { id: 'silu', name: 'SiLU', llm_usage: 'LLaMA' },
        ])
      ),
      getActivationCurve: jest.fn(() =>
        Promise.resolve({
          activation: 'gelu',
          formula: 'f(x) = 0.5 * x * (1 + erf(x / sqrt(2)))',
          latex: 'f(x)',
          description: 'GELU activation curve',
          temperature: 1.0,
          points: [
            { x: -2.0, y: -0.045, derivative: 0.05 },
            { x: 0.0, y: 0.0, derivative: 0.5 },
            { x: 2.0, y: 1.954, derivative: 1.05 },
          ],
        })
      ),
      computeSoftmaxTemperature: jest.fn(() =>
        Promise.resolve({
          temperature: 1.0,
          logits: [3.2, 1.8],
          elements: [
            { index: 0, raw_logit: 3.2, scaled_logit: 3.2, probability: 0.802, percentage: 80.2 },
            { index: 1, raw_logit: 1.8, scaled_logit: 1.8, probability: 0.198, percentage: 19.8 },
          ],
          entropy_bits: 0.72,
          max_probability: 0.802,
          argmax_index: 0,
          interpretation: 'Balanced temperature',
        })
      ),
      simulateAutograd: jest.fn(() =>
        Promise.resolve({
          success: true,
          architecture: 'MLP: 2 -> 3 -> 1 (RELU)',
          loss: 0.25,
          y_target: [1.0],
          y_pred: [0.5],
          nodes: [
            { id: 'input_x', label: 'Input x', type: 'input', shape: [1, 2], forward_val: [1.0, -0.5], grad_val: null, layer: 0 },
            { id: 'loss', label: 'MSE Loss', type: 'loss', shape: [], forward_val: 0.25, grad_val: 1.0, layer: 6 },
          ],
          edges: [],
        })
      ),
    },
  };
});

describe('TensorLabPage', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('renders page title and initial shape analysis', async () => {
    render(<TensorLabPage />);

    expect(screen.getByText('Tensor & Math Lab')).toBeInTheDocument();
    expect(screen.getByText(/Tensör Boyutları, Bellek Ayak İzi/i)).toBeInTheDocument();
    expect(screen.getByText('1. Şekil & Bellek')).toBeInTheDocument();

    await waitFor(() => {
      expect(tensorLabApi.analyzeShape).toHaveBeenCalled();
    });

    // Check rendered metric cards
    await waitFor(() => {
      expect(screen.getAllByText('4.00 MB').length).toBeGreaterThan(0);
      expect(screen.getByText('1,048,576')).toBeInTheDocument();
    });
  });

  test('switches to broadcasting tab and triggers broadcast check', async () => {
    render(<TensorLabPage />);

    const broadcastTabBtn = screen.getByText('2. Broadcasting');
    fireEvent.click(broadcastTabBtn);

    await waitFor(() => {
      expect(screen.getByText(/Broadcasting Kuralları/i)).toBeInTheDocument();
      expect(screen.getByText(/Uyumlu: Sonuç/i)).toBeInTheDocument();
    });
  });

  test('switches to GEMM MatMul tab and displays matrix cards and formula', async () => {
    render(<TensorLabPage />);

    const matmulTabBtn = screen.getByText('3. GEMM MatMul');
    fireEvent.click(matmulTabBtn);

    await waitFor(() => {
      expect(screen.getByText(/Genel Matris Çarpımı/i)).toBeInTheDocument();
      expect(screen.getByText(/16 FLOPs/i)).toBeInTheDocument();
    });
  });

  test('switches to Activation & Autograd tab', async () => {
    render(<TensorLabPage />);

    const actTabBtn = screen.getByText('4. Aktivasyon & Autograd');
    fireEvent.click(actTabBtn);

    await waitFor(() => {
      expect(screen.getByText(/Aktivasyon Fonksiyonları & Türev Eğrileri/i)).toBeInTheDocument();
      expect(screen.getByText(/Shannon Entropisi/i)).toBeInTheDocument();
    });
  });
});
