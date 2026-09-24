import React from 'react';
import { render, screen, act } from '@testing-library/react';
import { ExperimentProvider, useExperiment, ExperimentStatusBar } from '@/contexts/ExperimentContext';
import * as authContext from '@/lib/auth-context';

// Mock useAuth
jest.mock('@/lib/auth-context', () => ({
  useAuth: jest.fn(),
}));

function TestConsumer() {
  const {
    experiment,
    compatibility,
    setDataset,
    setTokenizer,
    setModel,
    clearExperiment
  } = useExperiment();

  return (
    <div>
      <div data-testid="experiment-dataset">{experiment.datasetName || 'no-dataset'}</div>
      <div data-testid="experiment-tokenizer">{experiment.tokenizerName || 'no-tokenizer'}</div>
      <div data-testid="experiment-model">{experiment.modelName || 'no-model'}</div>
      <div data-testid="compat-valid">{compatibility.isValid ? 'valid' : 'invalid'}</div>
      <div data-testid="compat-warnings">{compatibility.warnings.join('; ')}</div>
      <div data-testid="compat-notes">{compatibility.notes?.join('; ') || ''}</div>
      <button
        onClick={() => setDataset('ds-1', 'Turkish Wiki', 'v1.0')}
        data-testid="btn-set-dataset"
      >
        Set Dataset
      </button>
      <button
        onClick={() => setTokenizer('tok-1', 'BPE Tokenizer', 1000, { datasetVersion: 'v1.0' })}
        data-testid="btn-set-tokenizer"
      >
        Set Tokenizer
      </button>
      <button
        onClick={() => setModel('mod-1', 'Mini-GPT', undefined, { vocabSize: 500, tokenizerId: 'tok-diff' })}
        data-testid="btn-set-incompat-model"
      >
        Set Incompatible Model
      </button>
      <button
        onClick={() => setModel('mod-2', 'Mini-GPT-Big', undefined, { vocabSize: 2000, tokenizerId: 'tok-1' })}
        data-testid="btn-set-compat-model"
      >
        Set Compatible Model
      </button>
      <button onClick={() => clearExperiment()} data-testid="btn-clear">
        Clear
      </button>
    </div>
  );
}

describe('ExperimentContext Lifecycle & Compatibility', () => {
  beforeEach(() => {
    localStorage.clear();
    jest.clearAllMocks();
  });

  it('correctly sets artifacts and isolates state under user-specific storageKey', () => {
    (authContext.useAuth as jest.Mock).mockReturnValue({
      user: { user_id: 'user_researcher_1', username: 'res1', role: 'researcher' },
      isAuthenticated: true,
    });

    render(
      <ExperimentProvider>
        <TestConsumer />
        <ExperimentStatusBar />
      </ExperimentProvider>
    );

    expect(screen.getByTestId('experiment-dataset').textContent).toBe('no-dataset');

    // Set dataset and tokenizer
    act(() => {
      screen.getByTestId('btn-set-dataset').click();
      screen.getByTestId('btn-set-tokenizer').click();
    });

    expect(screen.getByTestId('experiment-dataset').textContent).toBe('Turkish Wiki');
    expect(screen.getByTestId('experiment-tokenizer').textContent).toBe('BPE Tokenizer');

    // Check localStorage saved under user-specific key
    const userStorage = localStorage.getItem('ai-lab-experiment-context:user_researcher_1');
    expect(userStorage).not.toBeNull();
    const parsed = JSON.parse(userStorage!);
    expect(parsed.datasetName).toBe('Turkish Wiki');
    expect(parsed.ownerUserId).toBe('user_researcher_1');

    // Guest or other user keys must remain empty
    expect(localStorage.getItem('ai-lab-experiment-context:guest')).toBeNull();
    expect(localStorage.getItem('ai-lab-experiment-context:user_other')).toBeNull();
  });

  it('detects artifact compatibility issues accurately (index overflow & tokenizer mismatch)', () => {
    (authContext.useAuth as jest.Mock).mockReturnValue({
      user: { user_id: 'user_researcher_1', username: 'res1', role: 'researcher' },
      isAuthenticated: true,
    });

    render(
      <ExperimentProvider>
        <TestConsumer />
      </ExperimentProvider>
    );

    // Set Tokenizer with vocab size 1000
    act(() => {
      screen.getByTestId('btn-set-tokenizer').click();
    });

    // Set Model with vocab size 500 (smaller than tokenizer -> index overflow) and different tokenizerId
    act(() => {
      screen.getByTestId('btn-set-incompat-model').click();
    });

    expect(screen.getByTestId('compat-valid').textContent).toBe('invalid');
    const warnings = screen.getByTestId('compat-warnings').textContent || '';
    expect(warnings).toContain('Tokenizer Uyuşmazlığı');
    expect(warnings).toContain('Kritik İndeks Taşması Riski');

    // Now set Model with vocab size 2000 and matching tokenizerId
    act(() => {
      screen.getByTestId('btn-set-compat-model').click();
    });

    expect(screen.getByTestId('compat-valid').textContent).toBe('valid');
    const notes = screen.getByTestId('compat-notes').textContent || '';
    expect(notes).toContain('Fazladan embedding rezervi');
  });

  it('prevents state leaking when switching to another user', () => {
    // User A sets an experiment
    localStorage.setItem(
      'ai-lab-experiment-context:user_A',
      JSON.stringify({ datasetName: 'User A Secret Dataset', ownerUserId: 'user_A' })
    );

    // Render as User B
    (authContext.useAuth as jest.Mock).mockReturnValue({
      user: { user_id: 'user_B', username: 'userB', role: 'researcher' },
      isAuthenticated: true,
    });

    render(
      <ExperimentProvider>
        <TestConsumer />
      </ExperimentProvider>
    );

    // User B must start fresh and NOT see User A's dataset
    expect(screen.getByTestId('experiment-dataset').textContent).toBe('no-dataset');
    // User A's localStorage must not be modified or overwritten by User B
    const userAStorage = JSON.parse(localStorage.getItem('ai-lab-experiment-context:user_A')!);
    expect(userAStorage.datasetName).toBe('User A Secret Dataset');
  });
});
