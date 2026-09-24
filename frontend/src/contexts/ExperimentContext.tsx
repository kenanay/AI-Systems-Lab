/**
 * Experiment Context
 * 
 * Lab'lar arası bağlam aktarımı için merkezi state yönetimi.
 * Kullanıcının seçtiği dataset, model, tokenizer gibi artefaktları
 * farklı lab'lar arasında taşır.
 * 
 * Örnek Akış:
 * 1. Dataset Lab'da dataset seç
 * 2. Tokenizer Lab'a git - dataset otomatik yüklü
 * 3. Training Lab'a git - dataset + tokenizer hazır
 * 4. Evaluation Lab'a git - trained model ile değerlendirme
 */

'use client';

import React, { createContext, useContext, useState, useCallback, useEffect } from 'react';
import { useAuth } from '@/lib/auth-context';

export interface ArtifactCompatibility {
  isValid: boolean;
  warnings: string[];
}

// Experiment state type
export interface ExperimentState {
  // Identifiers
  experimentId?: string;
  experimentName?: string;
  
  // Artifacts
  datasetId?: string;
  datasetName?: string;
  datasetVersion?: string;
  datasetSourceFiles?: string[];
  
  tokenizerId?: string;
  tokenizerName?: string;
  tokenizerVocabSize?: number;
  tokenizerDatasetVersion?: string;
  
  modelId?: string;
  modelName?: string;
  modelCheckpoint?: string;
  modelVocabSize?: number;
  modelTokenizerId?: string;
  
  // Configuration
  taskType?: 'pretrain' | 'sft' | 'lora' | 'evaluation' | 'rag';
  
  // Metadata
  createdAt?: string;
  lastModified?: string;
  
  // Navigation history
  visitedLabs?: string[];
}

interface ExperimentContextType {
  // Current state
  experiment: ExperimentState;
  compatibility: ArtifactCompatibility;
  
  // Actions
  setDataset: (id: string, name: string, version?: string, options?: { resetDownstream?: boolean }) => void;
  setTokenizer: (id: string, name: string, vocabSize?: number, options?: { resetDownstream?: boolean; datasetVersion?: string }) => void;
  setModel: (id: string, name: string, checkpoint?: string, options?: { vocabSize?: number; tokenizerId?: string }) => void;
  setTaskType: (taskType: ExperimentState['taskType']) => void;
  validateCompatibility: (state?: ExperimentState) => ArtifactCompatibility;
  
  // Experiment management
  startNewExperiment: (name?: string) => void;
  clearExperiment: () => void;
  loadExperiment: (state: ExperimentState) => void;
  
  // Navigation tracking
  markLabVisited: (labName: string) => void;
  
  // Helpers
  hasDataset: () => boolean;
  hasTokenizer: () => boolean;
  hasModel: () => boolean;
  isReady: (requiredItems: ('dataset' | 'tokenizer' | 'model')[]) => boolean;
}

const ExperimentContext = createContext<ExperimentContextType | undefined>(undefined);

export function ExperimentProvider({ children }: { children: React.ReactNode }) {
  const { user } = useAuth();
  const [experiment, setExperiment] = useState<ExperimentState>({});

  // Kullanıcıya özel localStorage anahtarı (Kullanıcı değiştiğinde veya çıkış yapıldığında izolasyon sağlar)
  const storageKey = user?.user_id
    ? `ai-lab-experiment-context:${user.user_id}`
    : 'ai-lab-experiment-context:guest';

  // Artefakt uyumluluk denetleyicisi (Dataset - Tokenizer - Model uyumu)
  const validateCompatibility = useCallback((targetState?: ExperimentState): ArtifactCompatibility => {
    const state = targetState || experiment;
    const warnings: string[] = [];

    // Dataset vs Tokenizer kontrolü
    if (state.datasetVersion && state.tokenizerDatasetVersion && state.datasetVersion !== state.tokenizerDatasetVersion) {
      warnings.push(
        `Seçili Tokenizer (${state.tokenizerName}) farklı bir dataset versiyonu (${state.tokenizerDatasetVersion}) ile eğitilmiş; aktif dataset: ${state.datasetVersion}.`
      );
    }

    // Tokenizer vs Model kontrolü
    if (state.modelTokenizerId && state.tokenizerId && state.modelTokenizerId !== state.tokenizerId) {
      warnings.push(
        `Seçili Model (${state.modelName}) mevcut Tokenizer (${state.tokenizerName}) yerine farklı bir tokenizer ile eğitilmiş.`
      );
    }

    if (state.modelVocabSize && state.tokenizerVocabSize && state.modelVocabSize !== state.tokenizerVocabSize) {
      warnings.push(
        `Sözlük boyutu uyuşmazlığı: Model sözlüğü ${state.modelVocabSize}, Tokenizer sözlüğü ${state.tokenizerVocabSize}. Tahminlerde indeks taşması riski var.`
      );
    }

    return {
      isValid: warnings.length === 0,
      warnings,
    };
  }, [experiment]);

  const compatibility = validateCompatibility(experiment);

  // Load from localStorage on mount or when user changes
  useEffect(() => {
    if (typeof window !== 'undefined') {
      const stored = localStorage.getItem(storageKey);
      if (stored) {
        try {
          const parsed = JSON.parse(stored);
          setExperiment(parsed);
          return;
        } catch (e) {
          console.error('Failed to load experiment context:', e);
        }
      }
      // Kullanıcı değiştiyse veya kayıt yoksa temiz state'e geç
      setExperiment({});
    }
  }, [storageKey]);

  // Save to localStorage on change
  useEffect(() => {
    if (typeof window !== 'undefined') {
      if (Object.keys(experiment).length > 0) {
        localStorage.setItem(storageKey, JSON.stringify(experiment));
      } else {
        localStorage.removeItem(storageKey);
      }
    }
  }, [experiment, storageKey]);

  const setDataset = useCallback(
    (id: string, name: string, version?: string, options?: { resetDownstream?: boolean }) => {
      setExperiment(prev => {
        const isNewDataset = prev.datasetId && prev.datasetId !== id;
        const reset = options?.resetDownstream ?? false;
        return {
          ...prev,
          datasetId: id,
          datasetName: name,
          datasetVersion: version,
          tokenizerId: reset && isNewDataset ? undefined : prev.tokenizerId,
          tokenizerName: reset && isNewDataset ? undefined : prev.tokenizerName,
          tokenizerVocabSize: reset && isNewDataset ? undefined : prev.tokenizerVocabSize,
          tokenizerDatasetVersion: reset && isNewDataset ? undefined : prev.tokenizerDatasetVersion,
          modelId: reset && isNewDataset ? undefined : prev.modelId,
          modelName: reset && isNewDataset ? undefined : prev.modelName,
          modelCheckpoint: reset && isNewDataset ? undefined : prev.modelCheckpoint,
          lastModified: new Date().toISOString(),
        };
      });
    },
    []
  );

  const setTokenizer = useCallback(
    (id: string, name: string, vocabSize?: number, options?: { resetDownstream?: boolean; datasetVersion?: string }) => {
      setExperiment(prev => {
        const isNewTokenizer = prev.tokenizerId && prev.tokenizerId !== id;
        const reset = options?.resetDownstream ?? false;
        return {
          ...prev,
          tokenizerId: id,
          tokenizerName: name,
          tokenizerVocabSize: vocabSize,
          tokenizerDatasetVersion: options?.datasetVersion || prev.datasetVersion,
          modelId: reset && isNewTokenizer ? undefined : prev.modelId,
          modelName: reset && isNewTokenizer ? undefined : prev.modelName,
          modelCheckpoint: reset && isNewTokenizer ? undefined : prev.modelCheckpoint,
          lastModified: new Date().toISOString(),
        };
      });
    },
    []
  );

  const setModel = useCallback(
    (id: string, name: string, checkpoint?: string, options?: { vocabSize?: number; tokenizerId?: string }) => {
      setExperiment(prev => ({
        ...prev,
        modelId: id,
        modelName: name,
        modelCheckpoint: checkpoint,
        modelVocabSize: options?.vocabSize,
        modelTokenizerId: options?.tokenizerId || prev.tokenizerId,
        lastModified: new Date().toISOString(),
      }));
    },
    []
  );

  const setTaskType = useCallback((taskType: ExperimentState['taskType']) => {
    setExperiment(prev => ({
      ...prev,
      taskType,
      lastModified: new Date().toISOString(),
    }));
  }, []);

  const startNewExperiment = useCallback((name?: string) => {
    const newExperiment: ExperimentState = {
      experimentId: `exp_${Date.now()}`,
      experimentName: name || `Experiment ${new Date().toLocaleDateString('tr-TR')}`,
      createdAt: new Date().toISOString(),
      lastModified: new Date().toISOString(),
      visitedLabs: [],
    };
    setExperiment(newExperiment);
  }, []);

  const clearExperiment = useCallback(() => {
    setExperiment({});
    if (typeof window !== 'undefined') {
      localStorage.removeItem(storageKey);
    }
  }, [storageKey]);

  const loadExperiment = useCallback((state: ExperimentState) => {
    setExperiment(state);
  }, []);

  const markLabVisited = useCallback((labName: string) => {
    setExperiment(prev => ({
      ...prev,
      visitedLabs: [...(prev.visitedLabs || []), labName].filter(
        (lab, index, self) => self.indexOf(lab) === index // unique
      ),
    }));
  }, []);

  const hasDataset = useCallback(() => {
    return !!experiment.datasetId;
  }, [experiment.datasetId]);

  const hasTokenizer = useCallback(() => {
    return !!experiment.tokenizerId;
  }, [experiment.tokenizerId]);

  const hasModel = useCallback(() => {
    return !!experiment.modelId;
  }, [experiment.modelId]);

  const isReady = useCallback((requiredItems: ('dataset' | 'tokenizer' | 'model')[]) => {
    return requiredItems.every(item => {
      if (item === 'dataset') return hasDataset();
      if (item === 'tokenizer') return hasTokenizer();
      if (item === 'model') return hasModel();
      return false;
    });
  }, [hasDataset, hasTokenizer, hasModel]);

  const value: ExperimentContextType = {
    experiment,
    compatibility,
    setDataset,
    setTokenizer,
    setModel,
    setTaskType,
    validateCompatibility,
    startNewExperiment,
    clearExperiment,
    loadExperiment,
    markLabVisited,
    hasDataset,
    hasTokenizer,
    hasModel,
    isReady,
  };

  return (
    <ExperimentContext.Provider value={value}>
      {children}
    </ExperimentContext.Provider>
  );
}

// Hook
export function useExperiment() {
  const context = useContext(ExperimentContext);
  if (context === undefined) {
    throw new Error('useExperiment must be used within an ExperimentProvider');
  }
  return context;
}

// Helper component: Experiment status bar
export function ExperimentStatusBar() {
  const { experiment, hasDataset, hasTokenizer, hasModel, compatibility } = useExperiment();

  if (!experiment.experimentId) {
    return null;
  }

  return (
    <div className="bg-blue-50 dark:bg-blue-900/20 border-b border-blue-200 dark:border-blue-800 px-4 py-2">
      <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2">
        <div className="flex items-center gap-4 flex-wrap">
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse" />
            <span className="text-sm font-medium text-blue-900 dark:text-blue-100">
              {experiment.experimentName}
            </span>
          </div>
          
          <div className="flex items-center gap-2 text-xs text-blue-700 dark:text-blue-300 flex-wrap">
            {hasDataset() && (
              <div className="flex items-center gap-1 px-2 py-1 bg-blue-100 dark:bg-blue-900/40 rounded">
                <span>📊</span>
                <span>{experiment.datasetName} {experiment.datasetVersion ? `(${experiment.datasetVersion})` : ''}</span>
              </div>
            )}
            {hasTokenizer() && (
              <div className="flex items-center gap-1 px-2 py-1 bg-blue-100 dark:bg-blue-900/40 rounded">
                <span>🔤</span>
                <span>{experiment.tokenizerName} {experiment.tokenizerVocabSize ? `(${experiment.tokenizerVocabSize})` : ''}</span>
              </div>
            )}
            {hasModel() && (
              <div className="flex items-center gap-1 px-2 py-1 bg-blue-100 dark:bg-blue-900/40 rounded">
                <span>🤖</span>
                <span>{experiment.modelName}</span>
              </div>
            )}
          </div>
        </div>

        {compatibility.warnings.length > 0 && (
          <div className="flex items-center gap-2 text-xs text-amber-700 dark:text-amber-300 bg-amber-100 dark:bg-amber-900/40 px-2 py-1 rounded">
            <span>⚠️</span>
            <span>{compatibility.warnings[0]}</span>
          </div>
        )}
      </div>
    </div>
  );
}

// Helper component: Missing requirements alert
export function ExperimentRequirements({ 
  required,
  message 
}: { 
  required: ('dataset' | 'tokenizer' | 'model')[],
  message?: string
}) {
  const { isReady, hasDataset, hasTokenizer, hasModel } = useExperiment();

  if (isReady(required)) {
    return null;
  }

  const missing = required.filter(item => {
    if (item === 'dataset') return !hasDataset();
    if (item === 'tokenizer') return !hasTokenizer();
    if (item === 'model') return !hasModel();
    return false;
  });

  const labels = {
    dataset: 'Dataset',
    tokenizer: 'Tokenizer',
    model: 'Model',
  };

  return (
    <div className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg p-4">
      <div className="flex items-start gap-3">
        <div className="text-yellow-600 dark:text-yellow-400">⚠️</div>
        <div>
          <h4 className="font-semibold text-yellow-900 dark:text-yellow-100 mb-1">
            Eksik Bileşenler
          </h4>
          <p className="text-sm text-yellow-800 dark:text-yellow-200 mb-2">
            {message || 'Bu lab için gerekli bileşenler seçilmedi:'}
          </p>
          <ul className="text-sm text-yellow-700 dark:text-yellow-300 list-disc list-inside">
            {missing.map(item => (
              <li key={item}>{labels[item]}</li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}
