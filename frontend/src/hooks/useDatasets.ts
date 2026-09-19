/**
 * Datasets API Hooks
 * 
 * React Query ile datasets API'sine erişim.
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { api } from '@/lib/api';
import type { IngestionStats, DocumentPreview, DocumentRecord } from '@/types';

// Query keys
export const datasetsKeys = {
  all: ['datasets'] as const,
  stats: () => [...datasetsKeys.all, 'stats'] as const,
  documents: () => [...datasetsKeys.all, 'documents'] as const,
  documentsList: (params?: any) => [...datasetsKeys.documents(), 'list', params] as const,
  document: (id: string) => [...datasetsKeys.documents(), 'detail', id] as const,
};

/**
 * Dataset istatistikleri
 */
export function useDatasetStats() {
  return useQuery({
    queryKey: datasetsKeys.stats(),
    queryFn: () => api.datasets.getStats(),
  });
}

/**
 * Dokümanları listele
 */
export function useDocuments(params?: {
  skip?: number;
  limit?: number;
  language?: string;
  min_quality?: number;
}) {
  return useQuery({
    queryKey: datasetsKeys.documentsList(params),
    queryFn: () => api.datasets.listDocuments(params),
  });
}

/**
 * Doküman detayı
 */
export function useDocument(documentId: string) {
  return useQuery({
    queryKey: datasetsKeys.document(documentId),
    queryFn: () => api.datasets.getDocument(documentId),
    enabled: !!documentId,
  });
}

/**
 * File ID'ye göre doküman getir
 */
export function useDocumentByFile(fileId: string, enabled: boolean = true) {
  return useQuery({
    queryKey: [...datasetsKeys.documents(), 'by-file', fileId],
    queryFn: () => api.datasets.getDocumentByFile(fileId),
    enabled: enabled && !!fileId,
  });
}

/**
 * Parquet export mutation
 */
export function useExportParquet() {
  return useMutation({
    mutationFn: () => api.datasets.exportParquet(),
  });
}

/**
 * Pretraining export mutation
 */
export function useExportPretraining() {
  return useMutation({
    mutationFn: (minQuality: number = 0.5) => api.datasets.exportPretraining(minQuality),
  });
}

/**
 * Dataset version oluştur mutation
 */
export function useCreateDatasetVersion() {
  return useMutation({
    mutationFn: ({ version, description }: { version: string; description?: string }) =>
      api.datasets.createVersion(version, description),
  });
}
