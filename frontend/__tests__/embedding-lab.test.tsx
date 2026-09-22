import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import EmbeddingLabPage from '@/app/embedding-lab/page';

// Mock api
jest.mock('@/lib/api', () => {
  const original = jest.requireActual('@/lib/api');
  return {
    ...original,
    api: {
      ...original.api,
      embeddings: {
        project: jest.fn(() =>
          Promise.resolve({
            points: [
              { text: 'sıcak', x: 25.0, y: 15.0, z: 5.0, norm: 1.0 },
              { text: 'soğuk', x: -25.0, y: -15.0, z: -5.0, norm: 1.0 },
            ],
            dimensions: 2,
            explained_variance_ratio: [0.65, 0.25],
            d_model: 64,
            model_name: 'test-gpt',
          })
        ),
        similarity: jest.fn(() =>
          Promise.resolve({
            word_a: 'sıcak',
            word_b: 'soğuk',
            cosine_similarity: -0.85,
            angle_degrees: 148.2,
            euclidean_distance: 1.92,
            dot_product: -0.85,
            model_name: 'test-gpt',
          })
        ),
        analogy: jest.fn(() =>
          Promise.resolve({
            word_a: 'kral',
            word_b: 'adam',
            word_c: 'kadın',
            formula: 'kral - adam + kadın = ?',
            top_matches: [{ word: 'kraliçe', similarity: 0.94 }],
            model_name: 'test-gpt',
          })
        ),
      },
    },
  };
});

function renderWithClient(ui: React.ReactElement) {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  });
  return render(
    <QueryClientProvider client={queryClient}>{ui}</QueryClientProvider>
  );
}

describe('Embedding Lab Page', () => {
  it('renders page header and model info badge', () => {
    renderWithClient(<EmbeddingLabPage />);

    expect(screen.getByText('Embedding Lab')).toBeInTheDocument();
    expect(screen.getByText('PCA & Vektör Geometrisi')).toBeInTheDocument();
  });

  it('renders all preset word cluster buttons', () => {
    renderWithClient(<EmbeddingLabPage />);

    expect(screen.getByText('Zıtlıklar & Doğa')).toBeInTheDocument();
    expect(screen.getByText('Yapay Zeka & Teknoloji')).toBeInTheDocument();
    expect(screen.getByText('Duygular')).toBeInTheDocument();
    expect(screen.getByText('Analoji & Rol')).toBeInTheDocument();
  });

  it('allows switching dimension between 2D and 3D mode', () => {
    renderWithClient(<EmbeddingLabPage />);

    const btn3D = screen.getByText(/3D İzometrik/i);
    fireEvent.click(btn3D);

    expect(screen.getByText(/Döndür X:/i)).toBeInTheDocument();
  });
});
