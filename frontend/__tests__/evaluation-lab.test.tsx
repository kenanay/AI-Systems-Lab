import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import EvaluationLabPage from '@/app/evaluation/page';
import { api } from '@/lib/api';

// Mock api
jest.mock('@/lib/api', () => {
  const original = jest.requireActual('@/lib/api');
  return {
    ...original,
    api: {
      ...original.api,
      evaluation: {
        inspectText: jest.fn(() =>
          Promise.resolve({
            candidate_tokens: ['yapay', 'zeka', 'modelleri'],
            reference_tokens: ['yapay', 'zeka', 'sistemleri'],
            matched_unigrams: ['yapay', 'zeka'],
            matched_bigrams: ['yapay zeka'],
            matched_trigrams: [],
            bleu: { 'bleu-1': 66.7, 'bleu-2': 50.0, 'bleu-3': 0.0, 'bleu-4': 0.0 },
            brevity_penalty: 1.0,
            candidate_len: 3,
            reference_len: 3,
            rouge: { 'rouge-1': 66.7, 'rouge-2': 50.0, 'rouge-l': 66.7 },
          })
        ),
        runBenchmark: jest.fn(() =>
          Promise.resolve({
            benchmark_id: 'BENCH-MOCK-123',
            model_name: 'turkish-gpt-small',
            benchmark_name: 'perplexity',
            score: 14.82,
            metrics: { perplexity: 14.82, loss: 2.69 },
            timestamp: '2026-09-22T12:00:00',
            samples_evaluated: 50,
          })
        ),
        getResults: jest.fn(() =>
          Promise.resolve([
            {
              benchmark_id: 'BENCH-HIST-001',
              model_name: 'nano-gpt-v1',
              benchmark_name: 'bleu',
              score: 72.4,
              metrics: { 'bleu-1': 80.0, 'bleu-2': 72.4 },
              timestamp: '2026-09-22T10:00:00',
              samples_evaluated: 100,
            },
          ])
        ),
        getResult: jest.fn(() =>
          Promise.resolve({
            benchmark_id: 'BENCH-HIST-001',
            model_name: 'nano-gpt-v1',
            benchmark_name: 'bleu',
            score: 72.4,
            metrics: { 'bleu-1': 80.0 },
            timestamp: '2026-09-22T10:00:00',
            samples_evaluated: 100,
          })
        ),
        deleteResult: jest.fn(() =>
          Promise.resolve({ message: 'Benchmark result deleted successfully' })
        ),
        compareModels: jest.fn(() =>
          Promise.resolve({
            benchmark_name: 'perplexity',
            comparisons: [
              { model_name: 'nano-gpt-v1', score: 18.2, metrics: { perplexity: 18.2 } },
              { model_name: 'nano-gpt-v2', score: 14.5, metrics: { perplexity: 14.5 } },
            ],
            winner: 'nano-gpt-v2',
          })
        ),
        getBenchmarks: jest.fn(() =>
          Promise.resolve({
            benchmarks: {
              perplexity: { name: 'Perplexity', description: 'Test loss perplexity', metric: 'perplexity', lower_is_better: true },
              bleu: { name: 'BLEU Score', description: 'Generation overlap', metric: 'bleu', lower_is_better: false },
              gsm8k_cot: { name: 'GSM8K Reasoning', description: 'Math CoT', metric: 'accuracy', lower_is_better: false },
              turkish_knowledge: { name: 'Turkish Facts', description: 'Factual accuracy', metric: 'accuracy', lower_is_better: false },
            },
            count: 4,
          })
        ),
        getSampleQuestions: jest.fn(() =>
          Promise.resolve([
            {
              id: 'gsm8k_01',
              input: 'Örnek soru metni',
              target: 'Adım 1: işlem #### 42',
              domain: 'aritmetik',
              category: 'math_word_problem',
              difficulty: 'easy',
              numeric_answer: 42,
              steps: ['Adım 1: 20 + 22 = 42'],
              keywords: ['toplama'],
            },
          ])
        ),
        getRadarComparison: jest.fn(() =>
          Promise.resolve({
            models: [
              {
                model_name: 'nano-gpt-v1',
                overall_score: 78.5,
                dimensions: [
                  { name: 'Akıl Yürütme (GSM8K CoT)', score: 75.0, raw_score: 75.0 },
                  { name: 'Türkçe Bilgi & Doğruluk', score: 80.0, raw_score: 80.0 },
                  { name: 'Metin Akıcılığı (BLEU)', score: 78.0, raw_score: 78.0 },
                  { name: 'Özetleme & Kapsam (ROUGE)', score: 76.0, raw_score: 76.0 },
                  { name: 'Model Tutarlılığı (PPL)', score: 85.0, raw_score: 85.0 },
                ],
              },
              {
                model_name: 'nano-gpt-v2',
                overall_score: 84.2,
                dimensions: [
                  { name: 'Akıl Yürütme (GSM8K CoT)', score: 85.0, raw_score: 85.0 },
                  { name: 'Türkçe Bilgi & Doğruluk', score: 88.0, raw_score: 88.0 },
                  { name: 'Metin Akıcılığı (BLEU)', score: 82.0, raw_score: 82.0 },
                  { name: 'Özetleme & Kapsam (ROUGE)', score: 80.0, raw_score: 80.0 },
                  { name: 'Model Tutarlılığı (PPL)', score: 86.0, raw_score: 86.0 },
                ],
              },
            ],
            dimensions: [
              'Akıl Yürütme (GSM8K CoT)',
              'Türkçe Bilgi & Doğruluk',
              'Metin Akıcılığı (BLEU)',
              'Özetleme & Kapsam (ROUGE)',
              'Model Tutarlılığı (PPL)',
            ],
            winner_by_dimension: {
              'Akıl Yürütme (GSM8K CoT)': 'nano-gpt-v2',
              'Türkçe Bilgi & Doğruluk': 'nano-gpt-v2',
              'Metin Akıcılığı (BLEU)': 'nano-gpt-v2',
              'Özetleme & Kapsam (ROUGE)': 'nano-gpt-v2',
              'Model Tutarlılığı (PPL)': 'nano-gpt-v2',
            },
            overall_winner: 'nano-gpt-v2',
          })
        ),
        getModelMetrics: jest.fn(() => Promise.resolve({})),
      },
      models: {
        list: jest.fn(() =>
          Promise.resolve([
            { id: 1, name: 'nano-gpt-v1', model_name: 'nano-gpt-v1' },
            { id: 2, name: 'nano-gpt-v2', model_name: 'nano-gpt-v2' },
          ])
        ),
      },
    },
  };
});

describe('EvaluationLabPage Component', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('renders the header title and all four tab buttons', async () => {
    render(<EvaluationLabPage />);

    expect(screen.getByText('Model Evaluation & Benchmark Lab')).toBeInTheDocument();
    expect(screen.getByText(/Bölüm 31 & 32/i)).toBeInTheDocument();

    expect(screen.getByRole('button', { name: /BLEU & ROUGE N-Gram Inspector/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Benchmark Runner/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Model Comparison Arena/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Leaderboard & Geçmiş/i })).toBeInTheDocument();
  });

  it('runs initial text inspection on load and displays metrics', async () => {
    render(<EvaluationLabPage />);

    await waitFor(() => {
      expect(api.evaluation.inspectText).toHaveBeenCalled();
    });

    const bpElements = await screen.findAllByText(/Brevity Penalty \(BP\)/i);
    expect(bpElements.length).toBeGreaterThan(0);
    expect(screen.getByText(/Kümülatif BLEU/i)).toBeInTheDocument();
    expect(screen.getByText(/ROUGE-1 & ROUGE-2/i)).toBeInTheDocument();
    expect(screen.getByText(/ROUGE-L \(LCS F1\)/i)).toBeInTheDocument();
    expect(screen.getByText(/Sözcük \(Token\) Eşleşme Vurgulayıcı/i)).toBeInTheDocument();
  });

  it('switches to Benchmark Runner tab and executes benchmark test', async () => {
    render(<EvaluationLabPage />);

    const runnerTab = screen.getByRole('button', { name: /Benchmark Runner/i });
    fireEvent.click(runnerTab);

    expect(await screen.findByText(/Otomatik Benchmark Testi Çalıştır/i)).toBeInTheDocument();

    const runBtn = screen.getByRole('button', { name: /Benchmark Başlat/i });
    fireEvent.click(runBtn);

    await waitFor(() => {
      expect(api.evaluation.runBenchmark).toHaveBeenCalled();
    });

    expect(await screen.findByText(/Benchmark Başarıyla Tamamlandı/i)).toBeInTheDocument();
    expect(screen.getByText('14.82')).toBeInTheDocument();
    expect(screen.getByText(/BENCH-MOCK-123/i)).toBeInTheDocument();
  });

  it('loads sample questions and displays question pool drawer in Runner tab', async () => {
    render(<EvaluationLabPage />);

    const runnerTab = screen.getByRole('button', { name: /Benchmark Runner/i });
    fireEvent.click(runnerTab);

    // Switch benchmark to gsm8k_cot
    const selectBench = screen.getByLabelText(/Benchmark Türü/i);
    fireEvent.change(selectBench, { target: { value: 'gsm8k_cot' } });

    const inspectQuestionsBtn = await screen.findByRole('button', { name: /Soruları Gör/i });
    fireEvent.click(inspectQuestionsBtn);

    await waitFor(() => {
      expect(api.evaluation.getSampleQuestions).toHaveBeenCalled();
    });

    expect(await screen.findByText(/Örnek soru metni/i)).toBeInTheDocument();
    expect(screen.getByText(/Beklenen Çözüm & Cevap/i)).toBeInTheDocument();
  });

  it('switches to Model Comparison Arena and runs comparison', async () => {
    render(<EvaluationLabPage />);

    const arenaTab = screen.getByRole('button', { name: /Model Comparison Arena/i });
    fireEvent.click(arenaTab);

    expect(await screen.findByText(/[56]-Eksenli Model Radar Analizi/i)).toBeInTheDocument();

    // Switch to Head-to-Head mode
    const headToHeadSubTab = screen.getByRole('button', { name: /İkili Karşılaştırma/i });
    fireEvent.click(headToHeadSubTab);

    expect(await screen.findByText(/Model Karşılaştırma Arenası/i)).toBeInTheDocument();

    // Select Model A and Model B (default nano-gpt-v1 and nano-gpt-v2)
    const compareBtn = screen.getByRole('button', { name: /Arena Karşılaştırması Başlat/i });
    fireEvent.click(compareBtn);

    await waitFor(() => {
      expect(api.evaluation.compareModels).toHaveBeenCalled();
    });

    expect(await screen.findByText(/Arena Kazananı \(Winner\)/i)).toBeInTheDocument();
    expect(screen.getAllByText('nano-gpt-v2').length).toBeGreaterThan(0);
    expect(screen.getByText('KAZANAN')).toBeInTheDocument();
  });

  it('displays 6-Axis Model Radar chart and matrix in Arena mode', async () => {
    render(<EvaluationLabPage />);

    const arenaTab = screen.getByRole('button', { name: /Model Comparison Arena/i });
    fireEvent.click(arenaTab);

    expect(await screen.findByText(/[56]-Eksenli Model Radar Analizi/i)).toBeInTheDocument();
    expect(screen.getByText(/Radar Analizi İçin Modelleri Seçin/i)).toBeInTheDocument();

    await waitFor(() => {
      expect(api.evaluation.getRadarComparison).toHaveBeenCalled();
    });

    expect(await screen.findByText(/[56]-Eksenli Radar Şampiyonu/i)).toBeInTheDocument();
    expect(screen.getByText(/Model Spider \/ Radar Grafiği/i)).toBeInTheDocument();
    expect(screen.getByText(/Çok Boyutlu Başarım Matrisi/i)).toBeInTheDocument();
  });

  it('switches to Leaderboard & History and renders table with delete action', async () => {
    render(<EvaluationLabPage />);

    const historyTab = screen.getByRole('button', { name: /Leaderboard & Geçmiş/i });
    fireEvent.click(historyTab);

    await waitFor(() => {
      expect(api.evaluation.getResults).toHaveBeenCalled();
    });

    expect(await screen.findByText('BENCH-HIST-001')).toBeInTheDocument();
    expect(screen.getByText('72.40')).toBeInTheDocument();

    // Delete flow: click trash icon, click "Onayla"
    const deleteBtn = screen.getByTitle('Sil');
    fireEvent.click(deleteBtn);

    const confirmBtn = await screen.findByRole('button', { name: 'Onayla' });
    fireEvent.click(confirmBtn);

    await waitFor(() => {
      expect(api.evaluation.deleteResult).toHaveBeenCalledWith('BENCH-HIST-001');
    });
  });
});
