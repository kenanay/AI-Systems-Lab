import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import SystemsLabPage from '@/app/systems-lab/page';
import { api } from '@/lib/api';

// Mock Systems Lab API
jest.mock('@/lib/api', () => {
  const original = jest.requireActual('@/lib/api');
  return {
    ...original,
    api: {
      ...original.api,
      systemsLab: {
        getPresets: jest.fn(() =>
          Promise.resolve([
            {
              name: 'NVIDIA H100 SXM5',
              category: 'Datacenter GPU',
              peak_tflops: 1000.0,
              peak_bandwidth_gbs: 3350.0,
              vram_gb: 80.0,
              description: 'Hopper mimarisi test donanımı.',
            },
            {
              name: 'NVIDIA RTX 4090',
              category: 'Consumer GPU',
              peak_tflops: 330.0,
              peak_bandwidth_gbs: 1008.0,
              vram_gb: 24.0,
              description: 'Ada Lovelace tüketici kartı.',
            },
          ])
        ),
        analyzeRoofline: jest.fn(() =>
          Promise.resolve({
            hardware: {
              name: 'NVIDIA H100 SXM5',
              category: 'Datacenter GPU',
              peak_tflops: 1000.0,
              peak_bandwidth_gbs: 3350.0,
              vram_gb: 80.0,
              description: 'Hopper test donanımı.',
            },
            workload_name: 'Autoregressive LLM Token Generation',
            operational_intensity: 1.0,
            knee_point_intensity: 298.51,
            regime: 'Memory-Bound',
            attainable_tflops: 3.35,
            efficiency_pct: 0.335,
            curve_points: [
              { operational_intensity: 0.1, attainable_tflops: 0.335 },
              { operational_intensity: 1.0, attainable_tflops: 3.35 },
            ],
            diagnosis: 'İş yükü belirgin bir şekilde Bellek Bant Genişliği Darboğazında (Memory-Bound).',
          })
        ),
        simulateQuantization: jest.fn(() =>
          Promise.resolve({
            distribution_type: 'outlier',
            num_elements: 2000,
            original_stats: { min: -15.2, max: 18.4, mean: 0.02, std: 1.15 },
            formats: {
              fp32: {
                bits: 32,
                dtype: 'float32',
                mse: 0.0,
                mae: 0.0,
                snr_db: Infinity,
                compression_ratio: 1.0,
                original_memory_kb: 7.81,
                quantized_memory_kb: 7.81,
              },
              fp16: {
                bits: 16,
                dtype: 'float16',
                mse: 1.2e-7,
                mae: 2.1e-4,
                snr_db: 72.4,
                compression_ratio: 2.0,
                original_memory_kb: 7.81,
                quantized_memory_kb: 3.91,
              },
              int8: {
                bits: 8,
                dtype: 'int8',
                mse: 0.0084,
                mae: 0.065,
                snr_db: 22.3,
                compression_ratio: 4.0,
                original_memory_kb: 7.81,
                quantized_memory_kb: 1.95,
                scale: 0.1317,
                zero_point: 115,
              },
              int4: {
                bits: 4,
                dtype: 'int4',
                mse: 0.145,
                mae: 0.285,
                snr_db: 9.8,
                compression_ratio: 8.0,
                original_memory_kb: 7.81,
                quantized_memory_kb: 0.98,
                scale: 2.24,
                zero_point: 7,
              },
            },
            histogram_original: [
              { bin_center: -10, count: 10, density: 0.005 },
              { bin_center: 0, count: 1800, density: 0.9 },
              { bin_center: 10, count: 10, density: 0.005 },
            ],
            histogram_int8: [
              { bin_center: -10, count: 12, density: 0.006 },
              { bin_center: 0, count: 1790, density: 0.895 },
              { bin_center: 10, count: 12, density: 0.006 },
            ],
            histogram_int4: [
              { bin_center: -10, count: 20, density: 0.01 },
              { bin_center: 0, count: 1750, density: 0.875 },
              { bin_center: 10, count: 20, density: 0.01 },
            ],
            insights: [
              'Aykırı değerler (outliers) tüm ölçekleme faktörünü (Scale S) genişletir.',
              'INT8 kuantizasyon 4x bellek tasarrufu sağlar.',
            ],
          })
        ),
        simulateMemory: jest.fn(() =>
          Promise.resolve({
            mode: 'training',
            param_count_billions: 8.0,
            precision: 'fp16',
            gpu_vram_gb: 80.0,
            components: {
              weights_gb: 16.0,
              gradients_gb: 16.0,
              optimizer_gb: 32.0,
              activations_gb: 4.5,
              kv_cache_gb: 0.0,
              cuda_overhead_gb: 1.5,
            },
            total_memory_gb: 70.0,
            memory_utilization_pct: 87.5,
            fits_in_gpu: true,
            oom_risk: 'WARNING',
            recommendations: [
              'Gradient checkpointing aktif, aktivasyon belleği minimize edildi.',
              'ZeRO-1 veya ZeRO-2 kullanarak optimizatör durumlarını dağıtmayı değerlendirin.',
            ],
          })
        ),
      },
    },
  };
});

describe('SystemsLabPage Component', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('renders the header title and main tab controls', async () => {
    render(<SystemsLabPage />);

    expect(screen.getByText('SYSTEMS FOR AI & HARDWARE')).toBeInTheDocument();
    expect(
      screen.getByText('Roofline Modeli, Kuantizasyon & GPU VRAM Laboratuvarı')
    ).toBeInTheDocument();

    expect(screen.getByText('Roofline & Memory Wall')).toBeInTheDocument();
    expect(screen.getByText('Kuantizasyon Simülatörü')).toBeInTheDocument();
    expect(screen.getByText('GPU VRAM & Bellek Ayrıştırması')).toBeInTheDocument();
  });

  it('loads presets and renders Roofline analysis results', async () => {
    render(<SystemsLabPage />);

    await waitFor(() => {
      expect(api.systemsLab.getPresets).toHaveBeenCalled();
      expect(api.systemsLab.analyzeRoofline).toHaveBeenCalled();
    });

    // Check that regime badge and diagnosis appear
    await waitFor(() => {
      expect(screen.getAllByText('Memory-Bound').length).toBeGreaterThanOrEqual(1);
      expect(
        screen.getByText(/İş yükü belirgin bir şekilde Bellek Bant Genişliği Darboğazında/i)
      ).toBeInTheDocument();
      expect(screen.getByText('Erişilebilir Güç')).toBeInTheDocument();
      expect(screen.getByText('Donanım Verimi')).toBeInTheDocument();
    });
  });

  it('switches to Quantization tab and displays format cards and insights', async () => {
    render(<SystemsLabPage />);

    const quantTabBtn = screen.getByRole('button', { name: /Kuantizasyon Simülatörü/i });
    fireEvent.click(quantTabBtn);

    await waitFor(() => {
      expect(api.systemsLab.simulateQuantization).toHaveBeenCalled();
    });

    await waitFor(() => {
      expect(screen.getByText(/FP32 \(32-bit\)/i)).toBeInTheDocument();
      expect(screen.getByText(/INT8 \(8-bit\)/i)).toBeInTheDocument();
      expect(screen.getByText(/INT4 \(4-bit\)/i)).toBeInTheDocument();
      expect(screen.getByText('4.0x Sıkıştırma')).toBeInTheDocument();
      expect(screen.getByText('8.0x Sıkıştırma')).toBeInTheDocument();
      expect(
        screen.getByText(/Aykırı değerler \(outliers\) tüm ölçekleme faktörünü/i)
      ).toBeInTheDocument();
    });
  });

  it('switches to GPU VRAM tab and displays memory decomposition', async () => {
    render(<SystemsLabPage />);

    const vramTabBtn = screen.getByRole('button', { name: /GPU VRAM & Bellek Ayrıştırması/i });
    fireEvent.click(vramTabBtn);

    await waitFor(() => {
      expect(api.systemsLab.simulateMemory).toHaveBeenCalled();
    });

    await waitFor(() => {
      expect(screen.getByText('Model Parametreleri')).toBeInTheDocument();
      expect(screen.getByText('Toplam Gerekli VRAM')).toBeInTheDocument();
      expect(screen.getByText('KRİTİK SINIR')).toBeInTheDocument();
      expect(screen.getByText('Ağırlıklar (Weights)')).toBeInTheDocument();
      expect(screen.getByText('Gradyanlar (Gradients)')).toBeInTheDocument();
      expect(screen.getByText('Optimizatör Durumu')).toBeInTheDocument();
    });
  });

  it('handles training vs inference mode toggle in VRAM tab', async () => {
    render(<SystemsLabPage />);

    const vramTabBtn = screen.getByRole('button', { name: /GPU VRAM & Bellek Ayrıştırması/i });
    fireEvent.click(vramTabBtn);

    const infModeBtn = screen.getByRole('button', { name: /Çıkarım \(Inference\)/i });
    fireEvent.click(infModeBtn);

    await waitFor(() => {
      expect(api.systemsLab.simulateMemory).toHaveBeenCalledWith(
        expect.objectContaining({
          mode: 'inference',
        })
      );
    });
  });
});
