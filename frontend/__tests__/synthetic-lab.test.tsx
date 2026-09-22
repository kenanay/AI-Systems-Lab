import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import SyntheticLabPage from '@/app/synthetic-lab/page';
import { api } from '@/lib/api';

// Mock Synthetic Lab API
jest.mock('@/lib/api', () => {
  const original = jest.requireActual('@/lib/api');
  return {
    ...original,
    api: {
      ...original.api,
      syntheticLab: {
        getTemplates: jest.fn(() =>
          Promise.resolve({
            paradigms: [
              {
                id: 'self_instruct',
                name: 'Self-Instruct (Alpaca Tarzı)',
                description: 'Görev tohumlarından çok yönlü talimat-yanıt çiftleri türetme.',
                icon: '⚡',
              },
              {
                id: 'chain_of_thought',
                name: 'Düşünce Zinciri (Chain-of-Thought)',
                description: 'Adım adım mantıksal akıl yürütme ve problem çözme çiftleri.',
                icon: '🧠',
              },
            ],
            domains: [
              { id: 'computer_science', name: 'Bilgisayar Bilimleri & Algoritmalar' },
              { id: 'mathematics', name: 'Matematik & Olasılık' },
            ],
            complexities: ['basic', 'intermediate', 'advanced'],
            preset_profiles: {
              balanced: {
                name: 'Dengeli Filtre',
                min_perplexity: 5.0,
                max_perplexity: 120.0,
                min_words: 15,
                max_words: 1500,
                max_repetition_ratio: 0.18,
                min_quality_score: 65.0,
                pii_action: 'mask',
                max_jaccard_similarity: 0.82,
              },
            },
          })
        ),
        generateSamples: jest.fn(() =>
          Promise.resolve({
            total_generated: 2,
            paradigm: 'self_instruct',
            domain: 'computer_science',
            complexity: 'intermediate',
            samples: [
              {
                id: 'SYNTH-SELF-101',
                instruction: 'İkili arama algoritması nedir?',
                input_context: '',
                response: 'İkili arama sıralı dizide O(log n) zamanda arama yapar.',
                paradigm: 'self_instruct',
                domain: 'computer_science',
                complexity: 'intermediate',
                metrics: {
                  composite_score: 85.0,
                  verdict: 'ACCEPT',
                  perplexity: 28.5,
                  repetition_ratio_2g: 0.05,
                  repetition_ratio_4g: 0.0,
                  lexical_diversity_ttr: 0.85,
                  word_count: 24,
                  char_count: 140,
                  pii_count: 0,
                  subscores: {
                    fluency: 95.0,
                    repetition: 100.0,
                    length: 90.0,
                    formatting: 80.0,
                    safety: 100.0,
                  },
                },
              },
            ],
          })
        ),
        filterSamples: jest.fn(() =>
          Promise.resolve({
            total_input_count: 5,
            passed_count: 4,
            rejected_count: 1,
            final_yield_pct: 80.0,
            avg_initial_quality: 68.2,
            avg_final_quality: 84.5,
            funnel_stages: [
              {
                stage_name: '1. Uzunluk Filtresi',
                input_count: 5,
                passed_count: 5,
                rejected_count: 0,
                retention_rate: 100.0,
              },
              {
                stage_name: '2. Tekrarlama & Döngü Filtresi',
                input_count: 5,
                passed_count: 4,
                rejected_count: 1,
                retention_rate: 80.0,
              },
            ],
            passed_samples: [
              {
                id: 'SYNTH-SELF-101',
                instruction: 'İkili arama algoritması nedir?',
                response: 'İkili arama sıralı dizide O(log n) zamanda arama yapar.',
                metrics: { composite_score: 85.0, verdict: 'ACCEPT' },
              },
            ],
            rejected_samples: [
              {
                id: 'SYNTH-EDGE-001',
                instruction: 'Tekrar testi',
                response: 'model model model',
                rejection_stage: 'Repetition Loop Filter',
                rejection_reason: 'Aşırı 2-gram tekrarlama döngüsü',
                metrics: { composite_score: 30.0, verdict: 'REJECT' },
              },
            ],
          })
        ),
        scoreSample: jest.fn(() =>
          Promise.resolve({
            composite_score: 88.0,
            verdict: 'ACCEPT',
            verdict_reasons: [],
            perplexity: 24.2,
            repetition_ratio_2g: 0.04,
            repetition_ratio_4g: 0.0,
            lexical_diversity_ttr: 0.88,
            word_count: 35,
            char_count: 220,
            pii_count: 0,
            subscores: {
              fluency: 98.0,
              repetition: 100.0,
              length: 95.0,
              formatting: 85.0,
              safety: 100.0,
            },
          })
        ),
        exportDataset: jest.fn(() =>
          Promise.resolve({
            success: true,
            dataset_name: 'synthetic_curated_v1',
            export_format: 'jsonl',
            total_samples: 4,
            approx_tokens: 350,
            preview_jsonl: ['{"id": "SYNTH-1"}'],
            message: '4 adet sentetik örnek başarıyla dışa aktarıldı.',
          })
        ),
        ingestToDataset: jest.fn(() =>
          Promise.resolve({
            success: true,
            file_id: 'file_synth_101',
            document_count: 4,
            dataset_name: 'synthetic_curated_v1',
            avg_quality_score: 84.5,
            message: '4 sentetik doküman başarıyla derleyici havuzuna aktarıldı (file_synth_101).',
          })
        ),
      },
    },
  };
});

describe('SyntheticLabPage Component', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('renders header, title and tab controls', async () => {
    render(<SyntheticLabPage />);

    expect(screen.getByText('SYNTHETIC DATA & QUALITY FILTERING')).toBeInTheDocument();
    expect(
      screen.getByText('Sentetik Veri Üretimi, Perplexity & Kalite Filtreleme Laboratuvarı')
    ).toBeInTheDocument();

    expect(screen.getByText('Sentetik Veri Üreticisi')).toBeInTheDocument();
    expect(screen.getByText('Kalite Filtreleme & Huni Analizi')).toBeInTheDocument();
    expect(screen.getByText('Tekil Örnek Teşhisi & Skorlayıcı')).toBeInTheDocument();
  });

  it('loads templates and renders generated synthetic sample cards', async () => {
    render(<SyntheticLabPage />);

    await waitFor(() => {
      expect(api.syntheticLab.getTemplates).toHaveBeenCalled();
      expect(api.syntheticLab.generateSamples).toHaveBeenCalled();
    });

    await waitFor(() => {
      expect(screen.getByText('SYNTH-SELF-101')).toBeInTheDocument();
      expect(screen.getByText('İkili arama algoritması nedir?')).toBeInTheDocument();
      expect(screen.getByText(/Skor: 85.0/i)).toBeInTheDocument();
    });
  });

  it('switches to Filter Pipeline tab and displays funnel stages', async () => {
    render(<SyntheticLabPage />);

    const pipelineTabBtn = screen.getByRole('button', { name: /Kalite Filtreleme & Huni Analizi/i });
    fireEvent.click(pipelineTabBtn);

    await waitFor(() => {
      expect(screen.getByText(/Çok Aşamalı Kalite Filtreleme Parametreleri/i)).toBeInTheDocument();
      expect(screen.getByText('Çok Aşamalı Eleme Hunisi (Pipeline Funnel)')).toBeInTheDocument();
      expect(screen.getByText(/Nihai Verim/i)).toBeInTheDocument();
      expect(screen.getByText('%80')).toBeInTheDocument();
      expect(screen.getByText('1. Uzunluk Filtresi')).toBeInTheDocument();
    });
  });

  it('switches to Single Inspector tab and computes diagnostic metrics', async () => {
    render(<SyntheticLabPage />);

    const inspectorTabBtn = screen.getByRole('button', { name: /Tekil Örnek Teşhisi & Skorlayıcı/i });
    fireEvent.click(inspectorTabBtn);

    await waitFor(() => {
      expect(screen.getByText('Özel Örnek Girdisi')).toBeInTheDocument();
      expect(screen.getByText('Detaylı Kalite Teşhisi')).toBeInTheDocument();
      expect(screen.getByText('88')).toBeInTheDocument();
      expect(screen.getByText('Perplexity (Akıcılık):')).toBeInTheDocument();
      expect(screen.getByText('24.2')).toBeInTheDocument();
    });
  });

  it('switches to Filter Pipeline tab and triggers Dataset Compiler ingestion bridge', async () => {
    render(<SyntheticLabPage />);

    const pipelineTabBtn = screen.getByRole('button', { name: /Kalite Filtreleme & Huni Analizi/i });
    fireEvent.click(pipelineTabBtn);

    await waitFor(() => {
      expect(screen.getByText("Dataset Compiler'a Aktar")).toBeInTheDocument();
    });

    const ingestBtn = screen.getByRole('button', { name: /Dataset Compiler'a Aktar/i });
    fireEvent.click(ingestBtn);

    await waitFor(() => {
      expect(api.syntheticLab.ingestToDataset).toHaveBeenCalled();
      expect(
        screen.getByText(/4 sentetik doküman başarıyla derleyici havuzuna aktarıldı/i)
      ).toBeInTheDocument();
      expect(screen.getByText(/File ID: file_synth_101/i)).toBeInTheDocument();
      expect(screen.getByText(/Dataset Compiler'da Derle/i)).toBeInTheDocument();
    });
  });
});
