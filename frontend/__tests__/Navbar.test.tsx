import React from 'react';
import { render, screen } from '@testing-library/react';
import { Navbar } from '@/components/Navbar';

// Mock next/navigation
jest.mock('next/navigation', () => ({
  usePathname: () => '/',
}));

describe('Navbar Component', () => {
  beforeEach(() => {
    // Mock global fetch for engine status check
    global.fetch = jest.fn(() =>
      Promise.resolve({
        ok: true,
        json: () => Promise.resolve({ status: 'healthy' }),
      })
    ) as jest.Mock;
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('renders the brand title and research infrastructure subtitle', () => {
    render(<Navbar />);
    expect(screen.getByText('Local AI Lab')).toBeInTheDocument();
    expect(screen.getByText('RESEARCH INFRASTRUCTURE')).toBeInTheDocument();
  });

  it('renders all key laboratory navigation items', () => {
    render(<Navbar />);
    
    // Core Labs
    expect(screen.getByText('Öğrenme Yolu')).toBeInTheDocument();
    expect(screen.getByText('Attention Lab')).toBeInTheDocument();
    expect(screen.getByText('Embedding Lab')).toBeInTheDocument();
    expect(screen.getByText('RAG Lab')).toBeInTheDocument();
    expect(screen.getByText('Math Lab')).toBeInTheDocument();
    expect(screen.getByText('Tensor Lab')).toBeInTheDocument();
    expect(screen.getByText('Transformer Lab')).toBeInTheDocument();
    expect(screen.getByText('Evaluation Lab')).toBeInTheDocument();
    expect(screen.getByText('Systems Lab')).toBeInTheDocument();
    expect(screen.getByText('Synthetic Lab')).toBeInTheDocument();
    expect(screen.getByText('Tokenizer')).toBeInTheDocument();
    expect(screen.getByText('Compiler')).toBeInTheDocument();
    expect(screen.getByText('Modeller')).toBeInTheDocument();
  });

  it('contains correct href links for all labs', () => {
    render(<Navbar />);

    const journeyLink = screen.getByText('Öğrenme Yolu').closest('a');
    expect(journeyLink).toHaveAttribute('href', '/journey');

    const evalLink = screen.getByText('Evaluation Lab').closest('a');
    expect(evalLink).toHaveAttribute('href', '/evaluation');

    const systemsLink = screen.getByText('Systems Lab').closest('a');
    expect(systemsLink).toHaveAttribute('href', '/systems-lab');

    const syntheticLink = screen.getByText('Synthetic Lab').closest('a');
    expect(syntheticLink).toHaveAttribute('href', '/synthetic-lab');

    const ragLink = screen.getByText('RAG Lab').closest('a');
    expect(ragLink).toHaveAttribute('href', '/rag-lab');

    const mathLink = screen.getByText('Math Lab').closest('a');
    expect(mathLink).toHaveAttribute('href', '/math-lab');

    const tensorLink = screen.getByText('Tensor Lab').closest('a');
    expect(tensorLink).toHaveAttribute('href', '/tensor-lab');

    const transformerLink = screen.getByText('Transformer Lab').closest('a');
    expect(transformerLink).toHaveAttribute('href', '/transformer-lab');

    const attentionLink = screen.getByText('Attention Lab').closest('a');
    expect(attentionLink).toHaveAttribute('href', '/attention-lab');

    const embeddingLink = screen.getByText('Embedding Lab').closest('a');
    expect(embeddingLink).toHaveAttribute('href', '/embedding-lab');
  });
});

