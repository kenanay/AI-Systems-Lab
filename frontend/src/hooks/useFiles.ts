/**
 * Files API Hooks
 * 
 * React Query ile files API'sine erişim.
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { api } from '@/lib/api';
import type { FileRecord, FileUploadResponse } from '@/types';

// Query keys
export const filesKeys = {
  all: ['files'] as const,
  lists: () => [...filesKeys.all, 'list'] as const,
  list: (skip: number, limit: number) => [...filesKeys.lists(), skip, limit] as const,
  detail: (id: string) => [...filesKeys.all, 'detail', id] as const,
};

/**
 * Dosyaları listele
 */
export function useFiles(skip = 0, limit = 100) {
  return useQuery({
    queryKey: filesKeys.list(skip, limit),
    queryFn: () => api.files.list(skip, limit),
  });
}

/**
 * Dosya detayı
 */
export function useFile(fileId: string) {
  return useQuery({
    queryKey: filesKeys.detail(fileId),
    queryFn: () => api.files.get(fileId),
    enabled: !!fileId,
  });
}

/**
 * Dosya yükle mutation
 */
export function useUploadFile() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (file: File) => api.files.upload(file),
    onSuccess: () => {
      // Files listesini yenile
      queryClient.invalidateQueries({ queryKey: filesKeys.lists() });
    },
  });
}

/**
 * Dosya sil mutation
 */
export function useDeleteFile() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (fileId: string) => api.files.delete(fileId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: filesKeys.lists() });
    },
  });
}

/**
 * Dosya metadata güncelle mutation
 */
export function useUpdateFileMetadata() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ 
      fileId, 
      updates 
    }: { 
      fileId: string; 
      updates: {
        training_allowed?: boolean;
        security_level?: string;
        license?: string;
        copyright_status?: string;
        source?: string;
        dataset_version?: string;
      }
    }) => api.files.updateMetadata(fileId, updates),
    onSuccess: (_, variables) => {
      // File detayını ve listesini yenile
      queryClient.invalidateQueries({ queryKey: filesKeys.detail(variables.fileId) });
      queryClient.invalidateQueries({ queryKey: filesKeys.lists() });
    },
  });
}

/**
 * Dosya process et mutation
 */
export function useProcessFile() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (fileId: string) => api.files.process(fileId),
    onSuccess: (_, fileId) => {
      // File detayını ve documents listesini yenile
      queryClient.invalidateQueries({ queryKey: filesKeys.detail(fileId) });
      queryClient.invalidateQueries({ queryKey: ['documents'] });
    },
  });
}

/**
 * Batch process mutation
 */
export function useBatchProcessFiles() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (fileIds: string[]) => api.files.batchProcess(fileIds),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: filesKeys.lists() });
      queryClient.invalidateQueries({ queryKey: ['documents'] });
    },
  });
}
