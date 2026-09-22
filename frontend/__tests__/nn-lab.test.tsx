import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import NNLabPage from '@/app/nn-lab/page';
import { api } from '@/lib/api';

// Mock api
jest.mock('@/lib/api', () => {
  const original = jest.requireActual('@/lib/api');
  return {
    ...original,
    api: {
      ...original.api,
      nnLab: {
        simulateMLP: jest.fn(() =>
          Promise.resolve({
            loss: 0.1234,
            loss_function: 'mse',
            activation: 'gelu',
            learning_rate: 0.05,
            architecture: [2, 4, 4, 2],
            predictions: [0.92, 0.08],
            targets: [1.0, 0.0],
            nodes: [
              {
                id: 'in_0',
                label: 'x_0',
                layer_idx: 0,
                neuron_idx: 0,
                layer_type: 'input',
                pre_activation_z: null,
                post_activation_a: 0.8,
                gradient_z: 0.012,
                gradient_a: 0.012,
                is_dead: false,
              },
              {
                id: 'h0_0',
                label: 'h0_0',
                layer_idx: 1,
                neuron_idx: 0,
                layer_type: 'hidden',
                pre_activation_z: 0.45,
                post_activation_a: 0.48,
                gradient_z: 0.034,
                gradient_a: 0.034,
                is_dead: false,
              },
              {
                id: 'out_0',
                label: 'y_0',
                layer_idx: 3,
                neuron_idx: 0,
                layer_type: 'output',
                pre_activation_z: 0.92,
                post_activation_a: 0.92,
                gradient_z: -0.08,
                gradient_a: -0.08,
                is_dead: false,
              },
            ],
            synapses: [
              {
                id: 'syn_in0_h00',
                source_id: 'in_0',
                target_id: 'h0_0',
                source_layer: 0,
                target_layer: 1,
                weight: 0.65,
                gradient_w: 0.024,
                weight_update: -0.0012,
              },
            ],
            chain_rule_steps: [
              {
                step: 1,
                layer: 'Layer 3 (Output)',
                formula: 'dL / dy_pred = 2 * (y_pred - y_true) / N',
                latex: '\\frac{\\partial L}{\\partial y}',
                description: 'MSE kaybının ağ çıkışına göre kısmi türevi',
              },
            ],
            layer_health: [
              {
                layer_idx: 1,
                max_grad: 0.045,
                mean_grad: 0.021,
                status: 'HEALTHY',
              },
            ],
            dead_neurons: [],
            overall_health: 'HEALTHY',
            activation_derivative_formula: 'Φ(x) + x * φ(x)',
          })
        ),
        getLandscapes: jest.fn(() =>
          Promise.resolve({
            landscapes: {
              quadratic_bowl: {
                key: 'quadratic_bowl',
                name: 'Kuadratik Çanak (Quadratic Bowl)',
                formula: 'f(x, y) = 0.5 * x² + 2.0 * y²',
                description: 'Konveks kuyu optimizasyonu',
                default_start: [3.5, 3.5],
                optimum: [0.0, 0.0],
                x_range: [-4.0, 4.0],
                y_range: [-4.0, 4.0],
              },
              saddle: {
                key: 'saddle',
                name: 'Eyer Noktası (Saddle Point)',
                formula: 'f(x, y) = x² - y²',
                description: 'Eyer noktası kaçışı',
                default_start: [-0.1, 1.8],
                optimum: [0.0, 0.0],
                x_range: [-3.0, 3.0],
                y_range: [-3.0, 3.0],
              },
            },
            optimizers: {
              adamw: {
                name: 'AdamW',
                color: '#6366f1',
                description: 'Decoupled weight decay ile adaptif momentum',
              },
              sgd: {
                name: 'SGD',
                color: '#ef4444',
                description: 'Klasik birinci derece türev güncellemesi',
              },
            },
          })
        ),
        raceOptimizers: jest.fn(() =>
          Promise.resolve({
            landscape: {
              key: 'quadratic_bowl',
              name: 'Kuadratik Çanak (Quadratic Bowl)',
              formula: 'f(x, y) = 0.5 * x² + 2.0 * y²',
              description: 'Konveks kuyu optimizasyonu',
              optimum: [0.0, 0.0],
              start_position: [3.5, 3.5],
            },
            parameters: {
              learning_rate: 0.03,
              momentum: 0.9,
              weight_decay: 0.01,
              steps: 50,
            },
            optimizers: {
              adamw: {
                name: 'AdamW',
                color: '#6366f1',
                description: 'Decoupled weight decay ile adaptif momentum',
                trajectory: [
                  { step: 0, x: 3.5, y: 3.5, loss: 24.5, grad_norm: 7.2 },
                  { step: 1, x: 2.1, y: 1.8, loss: 8.4, grad_norm: 4.1 },
                  { step: 50, x: 0.05, y: 0.02, loss: 0.002, grad_norm: 0.05 },
                ],
                final_position: [0.05, 0.02],
                final_loss: 0.002,
                initial_loss: 24.5,
                loss_reduction_pct: 99.99,
                total_distance: 4.9,
                distance_to_optimum: 0.054,
              },
              sgd: {
                name: 'SGD',
                color: '#ef4444',
                description: 'Klasik birinci derece türev güncellemesi',
                trajectory: [
                  { step: 0, x: 3.5, y: 3.5, loss: 24.5, grad_norm: 7.2 },
                  { step: 1, x: 3.2, y: 3.0, loss: 19.5, grad_norm: 6.5 },
                  { step: 50, x: 1.1, y: 0.8, loss: 1.89, grad_norm: 1.9 },
                ],
                final_position: [1.1, 0.8],
                final_loss: 1.89,
                initial_loss: 24.5,
                loss_reduction_pct: 92.28,
                total_distance: 3.5,
                distance_to_optimum: 1.36,
              },
            },
            winner: 'adamw',
            winner_name: 'AdamW',
            contours: {
              landscape_key: 'quadratic_bowl',
              x_range: [-4.0, 4.0],
              y_range: [-4.0, 4.0],
              x_vals: [-4.0, 0.0, 4.0],
              y_vals: [-4.0, 0.0, 4.0],
              z_grid: [
                [20.0, 10.0, 20.0],
                [10.0, 0.0, 10.0],
                [20.0, 10.0, 20.0],
              ],
              z_min: 0.0,
              z_max: 20.0,
              optimum: [0.0, 0.0],
            },
          })
        ),
        getActivations: jest.fn(() =>
          Promise.resolve({
            x_vals: [-3.0, -1.5, 0.0, 1.5, 3.0],
            activations: {
              gelu: {
                name: 'GELU (Gaussian Error Linear Unit)',
                derivative_formula: 'Φ(x) + x * φ(x)',
                fx: [-0.004, -0.1, 0.0, 1.4, 2.99],
                dfx: [-0.01, -0.05, 0.5, 1.05, 1.0],
                max_derivative: 1.1,
                vanishing_notes: ['Negatif bölgede yumuşak sönüm'],
              },
              relu: {
                name: 'ReLU (Rectified Linear Unit)',
                derivative_formula: '1 if x > 0 else 0',
                fx: [0.0, 0.0, 0.0, 1.5, 3.0],
                dfx: [0.0, 0.0, 0.0, 1.0, 1.0],
                max_derivative: 1.0,
                vanishing_notes: ['x < 0 iken tamamen sıfır türev'],
              },
            },
          })
        ),
      },
    },
  };
});

describe('NNLabPage Component', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('renders page header and initial MLP simulation data', async () => {
    render(<NNLabPage />);

    expect(screen.getByText('Neural Network & Backprop Lab')).toBeInTheDocument();
    expect(screen.getByText(/Yapay sinir ağlarında ileri ve geri geçiş/i)).toBeInTheDocument();
    expect(screen.getByText(/Bölüm 4 • Derin Öğrenme Temelleri/i)).toBeInTheDocument();

    await waitFor(() => {
      expect(api.nnLab.simulateMLP).toHaveBeenCalled();
      expect(screen.getByText(/MLP Mimari & Hesaplama Parametreleri/i)).toBeInTheDocument();
      expect(screen.getByText(/İnteraktif Hesaplama Grafiği/i)).toBeInTheDocument();
      expect(screen.getByText(/Adım Adım Geriye Yayılım/i)).toBeInTheDocument();
    });
  });

  test('displays MLP loss, predictions, and chain rule derivation steps', async () => {
    render(<NNLabPage />);

    await waitFor(() => {
      expect(screen.getByText('0.1234')).toBeInTheDocument(); // Loss
      expect(screen.getByText(/Tahmin: \[0.92, 0.08\]/i)).toBeInTheDocument();
      expect(screen.getByText(/Hedef: \[1, 0\]/i)).toBeInTheDocument();
      expect(screen.getByText('Layer 3 (Output)')).toBeInTheDocument();
      expect(screen.getByText(/dL \/ dy_pred = 2 \* \(y_pred - y_true\) \/ N/i)).toBeInTheDocument();
      expect(screen.getByText('Katman 1')).toBeInTheDocument();
    });
  });

  test('applies MLP preset and triggers new simulation', async () => {
    render(<NNLabPage />);

    await waitFor(() => {
      expect(api.nnLab.simulateMLP).toHaveBeenCalledTimes(1);
    });

    const dyingReluPresetBtn = screen.getByText('Ölü ReLU Senaryosu (Dying ReLU)');
    fireEvent.click(dyingReluPresetBtn);

    const runBtn = screen.getByRole('button', { name: /Ağı Simüle Et/i });
    fireEvent.click(runBtn);

    await waitFor(() => {
      expect(api.nnLab.simulateMLP).toHaveBeenCalledTimes(2);
    });
  });

  test('switches to Optimizer Race tab and runs race simulation', async () => {
    render(<NNLabPage />);

    const raceTabBtn = screen.getByRole('button', { name: /2D Kayıp Manzarası & Optimizer Yarışı/i });
    fireEvent.click(raceTabBtn);

    await waitFor(() => {
      expect(api.nnLab.raceOptimizers).toHaveBeenCalled();
      expect(screen.getByText(/2D Kayıp Yüzeyi & Çoklu Optimizer Yarışı/i)).toBeInTheDocument();
      expect(screen.getByText(/Optimizer Skor Tablosu/i)).toBeInTheDocument();
      expect(screen.getAllByText('AdamW').length).toBeGreaterThan(0);
      expect(screen.getAllByText('SGD').length).toBeGreaterThan(0);
    });
  });

  test('switches to Activation Dynamics tab and displays activation curves and probe', async () => {
    render(<NNLabPage />);

    const actTabBtn = screen.getByRole('button', { name: /Aktivasyon Dinamikleri & Gradiyen Doyumu/i });
    fireEvent.click(actTabBtn);

    await waitFor(() => {
      expect(api.nnLab.getActivations).toHaveBeenCalled();
      expect(screen.getByText(/Aktivasyon Fonksiyonları & Doygunluk \(Saturation\) Analizi/i)).toBeInTheDocument();
      expect(screen.getByText('GELU (Gaussian Error Linear Unit)')).toBeInTheDocument();
      expect(screen.getByText('ReLU (Rectified Linear Unit)')).toBeInTheDocument();
    });
  });
});
