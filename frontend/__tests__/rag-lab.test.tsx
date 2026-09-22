import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import RAGLabPage from '@/app/rag-lab/page';
import { ragApi } from '@/lib/api';

// Mock ragApi
jest.mock('@/lib/api', () => {
  const original = jest.requireActual('@/lib/api');
  return {
    ...original,
    ragApi: {
      collections: jest.fn(() =>
        Promise.resolve([
          {
            collection_name: 'default',
            vector_count: 24,
            d_model: 64,
            backend: 'pytorch_native',
          },
        ])
      ),
      chunk: jest.fn(() =>
        Promise.resolve({
          chunks: [
            {
              chunk_id: 'chk_1',
              text: 'Parça bir içeriği burada.',
              chunk_index: 0,
              start_char: 0,
              end_char: 25,
              token_count: 5,
              metadata: {},
            },
            {
              chunk_id: 'chk_2',
              text: 'Parça iki içeriği burada.',
              chunk_index: 1,
              start_char: 26,
              end_char: 50,
              token_count: 5,
              metadata: {},
            },
          ],
          total_chunks: 2,
          strategy: 'recursive',
          total_characters: 50,
        })
      ),
      search: jest.fn(() =>
        Promise.resolve({
          query: 'test query',
          mode: 'hybrid',
          collection_name: 'default',
          results: [
            {
              chunk_id: 'chk_1',
              text: 'Eşleşen parça metni',
              score: 0.92,
              rank: 1,
              metadata: { title: 'Test Belgesi', dense_rank: 1, sparse_rank: 1 },
            },
          ],
          count: 1,
          latency_ms: 1.45,
        })
      ),
      query: jest.fn(() =>
        Promise.resolve({
          query: 'test soru',
          answer: 'Test cevabı [Kaynak 1].',
          retrieved_chunks: [
            {
              chunk_id: 'chk_1',
              text: 'Test bağlamı',
              score: 0.95,
              rank: 1,
              metadata: { title: 'Test Belgesi' },
            },
          ],
          citations: [
            {
              citation_id: 1,
              chunk_id: 'chk_1',
              source_title: 'Test Belgesi',
              snippet: 'Test bağlamı',
              score: 0.95,
            },
          ],
          model_name: 'local-rag-generator',
          prompt_used: 'Grounded prompt içeriği',
          stats: { latency_ms: 25.4 },
        })
      ),
    },
  };
});

describe('RAG Lab Page', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('renders header, title, and initial Q&A tab', async () => {
    render(<RAGLabPage />);

    expect(screen.getByText('RAG & Retrieval Lab')).toBeInTheDocument();
    expect(screen.getByText('Kaynak Doğrulamalı Soru-Cevap')).toBeInTheDocument();
    expect(screen.getByPlaceholderText(/Örn: Transformer mimarisinde/i)).toBeInTheDocument();
  });

  it('switches to Chunking Lab tab and displays strategy selectors', async () => {
    render(<RAGLabPage />);

    const chunkingTabBtn = screen.getByText('Parçalama (Chunking) Lab');
    fireEvent.click(chunkingTabBtn);

    expect(screen.getByText('Etkileşimli Metin Parçalama (Chunking)')).toBeInTheDocument();
    expect(screen.getByText('recursive')).toBeInTheDocument();
    expect(screen.getByText('sentence')).toBeInTheDocument();
    expect(screen.getByText('fixed')).toBeInTheDocument();
  });

  it('switches to Retrieval Lab tab and displays mode buttons', async () => {
    render(<RAGLabPage />);

    const retrievalTabBtn = screen.getByText('Hibrit Arama (Retrieval) Lab');
    fireEvent.click(retrievalTabBtn);

    expect(screen.getByText('Hibrit Arama (Dense vs Sparse vs RRF)')).toBeInTheDocument();
    expect(screen.getByText('⚡ Hibrit (RRF)')).toBeInTheDocument();
    expect(screen.getByText('🧠 Dense (Vektör)')).toBeInTheDocument();
    expect(screen.getByText('🔍 Sparse (BM25)')).toBeInTheDocument();
  });

  it('triggers RAG query and renders grounded answer with citation badges', async () => {
    render(<RAGLabPage />);

    const queryBtn = screen.getByText('Sorgula & Üret');
    fireEvent.click(queryBtn);

    await waitFor(() => {
      expect(ragApi.query).toHaveBeenCalled();
    });

    await waitFor(() => {
      expect(screen.getByText('Grounded Model Yanıtı')).toBeInTheDocument();
      expect(screen.getByText('Test cevabı [Kaynak 1].')).toBeInTheDocument();
      expect(screen.getAllByText(/\[Kaynak 1\]/i).length).toBeGreaterThanOrEqual(1);
    });
  });
});
