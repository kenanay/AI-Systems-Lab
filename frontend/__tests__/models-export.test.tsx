import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import ModelsPage from '@/app/models/page';
import { modelsApi } from '@/lib/api';

// Mock next/navigation
jest.mock('next/navigation', () => ({
  usePathname: () => '/models',
}));

// Mock modelsApi
jest.mock('@/lib/api', () => {
  const actual = jest.requireActual('@/lib/api');
  return {
    ...actual,
    modelsApi: {
      list: jest.fn(),
      get: jest.fn(),
      verify: jest.fn(),
      delete: jest.fn(),
      exportModel: jest.fn(),
      listExports: jest.fn(),
      getDownloadUrl: jest.fn((modelName: string, fileName: string, version?: string) => {
        return `http://localhost:8000/api/v1/models/${modelName}/download/${fileName}?version=${version || '1.0.0'}`;
      }),
    },
  };
});

describe('ModelsPage - Model Export & Quantization Pipeline', () => {
  const mockModels = [
    {
      model_name: 'turkish-gpt-mini',
      version: '1.0.0',
      description: 'Türkçe mini dil modeli',
      architecture: 'GPTModel (Transformer Decoder)',
      parameters: 12500000,
      metrics: { loss: 2.14, perplexity: 8.5 },
      training_config: { epochs: 3, lr: 0.0003 },
      tags: ['turkce', 'gpt', 'demo'],
      created_at: '2026-09-22T12:00:00Z',
      file_size_mb: 50.0,
      model_hash: 'a1b2c3d4e5f678901234567890abcdef1234567890abcdef1234567890abcdef',
    },
  ];

  beforeEach(() => {
    jest.clearAllMocks();
    (modelsApi.list as jest.Mock).mockResolvedValue(mockModels);
    (modelsApi.listExports as jest.Mock).mockResolvedValue([]);
  });

  it('renders model cards with Export & Quantize button', async () => {
    render(<ModelsPage />);

    expect(await screen.findByText('turkish-gpt-mini')).toBeInTheDocument();
    expect(screen.getByText('📦 Dışa Aktar & Kuantize Et')).toBeInTheDocument();
  });

  it('opens Export & Quantization modal when clicking export button', async () => {
    render(<ModelsPage />);

    const exportBtn = await screen.findByText('📦 Dışa Aktar & Kuantize Et');
    fireEvent.click(exportBtn);

    // Modal elements should be visible
    expect(screen.getByText('Model Dışa Aktarma & Kuantizasyon')).toBeInTheDocument();
    expect(screen.getByText('1. Dışa Aktarma Formatı')).toBeInTheDocument();
    expect(screen.getByText('2. Kuantizasyon & Hassasiyet')).toBeInTheDocument();
    expect(screen.getByText('Tahmini Bellek & Dosya Boyutu Etkisi')).toBeInTheDocument();

    // Check format choices
    expect(screen.getByText('GGUF v3')).toBeInTheDocument();
    expect(screen.getByText('TorchScript')).toBeInTheDocument();

    // Check quantization choices
    expect(screen.getByText('FP16 (Yarı Hassasiyet)')).toBeInTheDocument();
    expect(screen.getByText('INT8 (Dinamik)')).toBeInTheDocument();
    expect(screen.getByText('INT4 (Blok Ağırlık)')).toBeInTheDocument();
  });

  it('updates live RAM & size estimation when changing quantization', async () => {
    render(<ModelsPage />);

    const exportBtn = await screen.findByText('📦 Dışa Aktar & Kuantize Et');
    fireEvent.click(exportBtn);

    // Select INT8
    const int8Btn = screen.getByText('INT8 (Dinamik)');
    fireEvent.click(int8Btn);

    // Live savings should display INT8 percentage
    expect(screen.getByText('%73.5 RAM Tasarrufu')).toBeInTheDocument();

    // Select INT4
    const int4Btn = screen.getByText('INT4 (Blok Ağırlık)');
    fireEvent.click(int4Btn);

    // Live savings should display INT4 percentage
    expect(screen.getByText('%86.2 RAM Tasarrufu')).toBeInTheDocument();
  });

  it('triggers export API and displays download link, checksum, and code snippet', async () => {
    const mockExportResponse = {
      success: true,
      model_name: 'turkish-gpt-mini',
      version: '1.0.0',
      export_format: 'gguf',
      quantization: 'int8',
      file_name: 'turkish-gpt-mini_1.0.0_int8.gguf',
      file_path: 'models/turkish-gpt-mini/1.0.0/exports/turkish-gpt-mini_1.0.0_int8.gguf',
      file_size_mb: 13.25,
      compression_ratio: 3.77,
      sha256: '99887766554433221100aabbccddeeff99887766554433221100aabbccddeeff',
      export_duration_sec: 1.45,
      download_url: '/api/v1/models/turkish-gpt-mini/download/turkish-gpt-mini_1.0.0_int8.gguf',
      deployment_snippet: 'FROM ./turkish-gpt-mini_1.0.0_int8.gguf\nPARAMETER temperature 0.7',
      created_at: '2026-09-22T21:10:00Z',
    };

    (modelsApi.exportModel as jest.Mock).mockResolvedValueOnce(mockExportResponse);

    render(<ModelsPage />);

    const exportBtn = await screen.findByText('📦 Dışa Aktar & Kuantize Et');
    fireEvent.click(exportBtn);

    // Select GGUF and INT8
    fireEvent.click(screen.getByText('GGUF v3'));
    fireEvent.click(screen.getByText('INT8 (Dinamik)'));

    // Trigger export
    const submitBtn = screen.getByText(/Dışa Aktarmayı Başlat/i);
    fireEvent.click(submitBtn);

    await waitFor(() => {
      expect(modelsApi.exportModel).toHaveBeenCalledWith('turkish-gpt-mini', {
        version: '1.0.0',
        export_format: 'gguf',
        quantization: 'int8',
      });
    });

    // Check completion UI
    expect(await screen.findByText('✓ Dışa Aktarma Tamamlandı')).toBeInTheDocument();
    expect(screen.getByText(/3.77x küçültüldü/i)).toBeInTheDocument();
    expect(screen.getByText(/99887766554433221100aabbccddeeff/)).toBeInTheDocument();
    expect(screen.getByText('Dağıtım & Çalıştırma Kodu')).toBeInTheDocument();
    expect(screen.getByText(/FROM \.\/turkish-gpt-mini_1\.0\.0_int8\.gguf/)).toBeInTheDocument();
  });

  it('can close the export modal via close button', async () => {
    render(<ModelsPage />);

    const exportBtn = await screen.findByText('📦 Dışa Aktar & Kuantize Et');
    fireEvent.click(exportBtn);

    expect(screen.getByText('Model Dışa Aktarma & Kuantizasyon')).toBeInTheDocument();

    const closeBtn = screen.getByTitle('Kapat');
    fireEvent.click(closeBtn);

    expect(screen.queryByText('Model Dışa Aktarma & Kuantizasyon')).not.toBeInTheDocument();
  });
});
