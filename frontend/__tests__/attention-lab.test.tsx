import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import AttentionLabPage from '@/app/attention-lab/page';

// Mock api
jest.mock('@/lib/api', () => {
  const original = jest.requireActual('@/lib/api');
  return {
    ...original,
    api: {
      ...original.api,
      inference: {
        attention: jest.fn(() =>
          Promise.resolve({
            matrix: [
              [1.0, 0.0, 0.0],
              [0.4, 0.6, 0.0],
              [0.3, 0.3, 0.4],
            ],
            tokens: ['Transformer', 'mimarisinde', 'attention'],
            num_layers: 4,
            num_heads: 4,
            layer_idx: 0,
            head_idx: 0,
            is_causal: true,
            model_name: 'test-gpt',
            all_heads_matrix: {},
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

describe('Attention Lab Page', () => {
  it('renders page header and Transformer architecture badge', () => {
    renderWithClient(<AttentionLabPage />);

    expect(screen.getByText('Attention Lab')).toBeInTheDocument();
    expect(screen.getByText('Transformer İç Mimarisi')).toBeInTheDocument();
    expect(screen.getByPlaceholderText(/İncelemek istediğin metni gir/i)).toBeInTheDocument();
  });

  it('renders all sample preset buttons', () => {
    renderWithClient(<AttentionLabPage />);

    expect(screen.getByText('Yapay Zeka:')).toBeInTheDocument();
    expect(screen.getByText('Transformer & Attention:')).toBeInTheDocument();
    expect(screen.getByText('Bağlamsal Zamir:')).toBeInTheDocument();
    expect(screen.getByText('Derin Öğrenme:')).toBeInTheDocument();
  });

  it('updates text input when a sample preset button is clicked', () => {
    renderWithClient(<AttentionLabPage />);

    const presetBtn = screen.getByText('Yapay Zeka:');
    fireEvent.click(presetBtn);

    const input = screen.getByPlaceholderText(/İncelemek istediğin metni gir/i) as HTMLInputElement;
    expect(input.value).toBe('Yapay zeka sistemleri öğrenir ve üretir.');
  });
});
