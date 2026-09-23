import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import JourneyPage from '@/app/journey/page';
import { api } from '@/lib/api';

// Mock api
jest.mock('@/lib/api', () => {
  const original = jest.requireActual('@/lib/api');
  return {
    ...original,
    api: {
      ...original.api,
      journey: {
        getProgress: jest.fn(() =>
          Promise.resolve({ answers: {} })
        ),
        getCurriculum: jest.fn(() =>
          Promise.resolve([
            {
              id: 'stage_0',
              order: 0,
              title: 'Aşama 0 — Temel Matematik & Hesaplama İlkeleri',
              category: 'Matematik & Altyapı',
              summary: 'Matris çarpımı ve aktivasyonlar.',
              concepts: ['Matris Çarpımı', 'Softmax'],
              objectives: ['Boyut kurallarını anlamak'],
              prerequisites: [],
              lab_url: '/tensor-lab',
              lab_title: 'Tensor Lab',
              questions: [
                {
                  id: 'q_stage_0_1',
                  stage_id: 'stage_0',
                  question: '(32, 128) ve (128, 64) tensör çarpım boyutu nedir?',
                  options: ['(32, 128)', '(32, 64)', '(128, 128)'],
                  correct_index: 1,
                  explanation: 'İç boyutlar eşleşir, sonuç (32, 64) olur.',
                  math_intuition: '(M, K) x (K, N) -> (M, N)',
                },
              ],
            },
            {
              id: 'stage_1',
              order: 1,
              title: 'Aşama 1 — Ham Veri & Ayrıştırma',
              category: 'Veri Mühendisliği',
              summary: 'PDF, TXT, CSV ayrıştırma.',
              concepts: ['SHA-256', 'Encoding'],
              objectives: ['Veri çıkarma'],
              prerequisites: ['stage_0'],
              lab_url: '/upload',
              lab_title: 'Upload & Explorer',
              questions: [
                {
                  id: 'q_stage_1_1',
                  stage_id: 'stage_1',
                  question: 'SHA-256 neden hesaplanır?',
                  options: ['Sıkıştırma', 'Bütünlük doğrulama'],
                  correct_index: 1,
                  explanation: 'Tek bir bit bile değişse tespit eder.',
                },
              ],
            },
          ])
        ),
        getStage: jest.fn((id: string) =>
          Promise.resolve({
            id,
            order: 0,
            title: 'Aşama 0',
            category: 'Matematik',
            summary: 'Özet',
            concepts: [],
            objectives: [],
            prerequisites: [],
            questions: [],
          })
        ),
        getGraph: jest.fn(() =>
          Promise.resolve({
            nodes: [
              { id: 'math', label: 'Temel Matematik', stage_id: 'stage_0', category: 'Matematik', level: 0, lab_url: '/tensor-lab' },
              { id: 'raw_data', label: 'Ham Veri', stage_id: 'stage_1', category: 'Veri', level: 1, lab_url: '/upload' },
            ],
            edges: [
              { id: 'e1', source: 'math', target: 'raw_data', label: 'Veri Temsili' },
            ],
          })
        ),
        getGlossary: jest.fn(() =>
          Promise.resolve([
            {
              term: 'BPE (Byte-Pair Encoding)',
              category: 'NLP & Tokenizer',
              short_def: 'Alt-kelime tokenizasyon algoritması.',
              detailed_explanation: 'Karakter çiftlerini birleştirir.',
              formula: 'count(pair)',
              related_lab_url: '/tokenizer',
            },
            {
              term: 'Self-Attention',
              category: 'Mimari',
              short_def: 'Bağlamsal dikkat mekanizması.',
              detailed_explanation: 'Q, K, V projeksiyonları.',
              formula: 'softmax(QK^T / sqrt(d)) V',
              related_lab_url: '/attention-lab',
            },
          ])
        ),
        checkQuestion: jest.fn((payload) =>
          Promise.resolve({
            question_id: payload.question_id,
            is_correct: payload.selected_option === 1,
            selected_option: payload.selected_option,
            correct_index: 1,
            explanation: 'İç boyutlar eşleşir, sonuç (32, 64) olur.',
            math_intuition: '(M, K) x (K, N) -> (M, N)',
          })
        ),
      },
    },
  };
});

describe('JourneyPage Component', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('renders the header title and all three tab buttons', async () => {
    render(<JourneyPage />);

    expect(screen.getByText('Guided Learning Journey & Knowledge Map')).toBeInTheDocument();
    expect(screen.getByText(/Bölüm 56/i)).toBeInTheDocument();

    expect(screen.getByRole('button', { name: /Rehberli Yolculuk/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Kavram & Ön Bilgi Haritası/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /AI & LLM Terimler Sözlüğü/i })).toBeInTheDocument();
  });

  it('loads curriculum stages and handles milestone question checking', async () => {
    render(<JourneyPage />);

    await waitFor(() => {
      expect(api.journey.getCurriculum).toHaveBeenCalled();
    });

    expect(await screen.findByText(/Aşama 0 — Temel Matematik & Hesaplama İlkeleri/i)).toBeInTheDocument();
    expect(screen.getByText(/Aşama 1 — Ham Veri & Ayrıştırma/i)).toBeInTheDocument();

    // Answer question 1
    const optionRadio = screen.getByLabelText(/\(32, 64\)/i);
    fireEvent.click(optionRadio);

    const checkBtn = screen.getAllByRole('button', { name: /Cevabı Doğrula/i })[0];
    fireEvent.click(checkBtn);

    await waitFor(() => {
      expect(api.journey.checkQuestion).toHaveBeenCalledWith({
        question_id: 'q_stage_0_1',
        selected_option: 1,
      });
    });

    expect(await screen.findByText(/Harika, Doğru Cevap!/i)).toBeInTheDocument();
    expect(screen.getByText(/İç boyutlar eşleşir/i)).toBeInTheDocument();
  });

  it('switches to Knowledge DAG tab and shows node prerequisites', async () => {
    render(<JourneyPage />);

    const graphTab = screen.getByRole('button', { name: /Kavram & Ön Bilgi Haritası/i });
    fireEvent.click(graphTab);

    await waitFor(() => {
      expect(api.journey.getGraph).toHaveBeenCalled();
    });

    expect(await screen.findByText(/Kavram & Ön Bilgi Bağımlılık Haritası/i)).toBeInTheDocument();
    expect(screen.getAllByText('Temel Matematik').length).toBeGreaterThan(0);
    expect(screen.getAllByText('Ham Veri').length).toBeGreaterThan(0);

    // Click on "Ham Veri" node to see incoming prerequisite "Temel Matematik"
    const rawDataNode = screen.getAllByText('Ham Veri')[0];
    fireEvent.click(rawDataNode);

    expect(screen.getByText(/Gerekli Ön Koşullar:/i)).toBeInTheDocument();
  });

  it('switches to Glossary tab and allows searching terms', async () => {
    render(<JourneyPage />);

    const glossaryTab = screen.getByRole('button', { name: /AI & LLM Terimler Sözlüğü/i });
    fireEvent.click(glossaryTab);

    await waitFor(() => {
      expect(api.journey.getGlossary).toHaveBeenCalled();
    });

    expect(await screen.findByText('BPE (Byte-Pair Encoding)')).toBeInTheDocument();
    expect(screen.getByText('Self-Attention')).toBeInTheDocument();

    // Type in search input
    const searchInput = screen.getByPlaceholderText(/Terimlerde veya açıklamalarda ara/i);
    fireEvent.change(searchInput, { target: { value: 'BPE' } });

    await waitFor(() => {
      expect(api.journey.getGlossary).toHaveBeenCalled();
    });
  });
});
