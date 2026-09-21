/**
 * Tokenizer Hooks
 * 
 * TanStack Query hooks for tokenizer operations.
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  createTokenizerTrainingJob,
  listTokenizerJobs,
  getTokenizerJob,
  cancelTokenizerJob,
  listTokenizers,
  getTokenizer,
  encodeText,
  decodeTokens,
  updateTokenizer,
  deleteTokenizer,
  type TokenizerTrainingRequest,
  type TokenizerJob,
  type TokenizerRecord,
} from '@/lib/api';

// ============================================================================
// Training Jobs
// ============================================================================

/**
 * List tokenizer training jobs
 */
export function useTokenizerJobs(status?: string) {
  return useQuery({
    queryKey: ['tokenizer-jobs', status],
    queryFn: () => listTokenizerJobs(status),
    refetchInterval: 5000, // Auto-refresh every 5 seconds
  });
}

/**
 * Get single tokenizer job
 */
export function useTokenizerJob(jobId: string | null) {
  return useQuery({
    queryKey: ['tokenizer-job', jobId],
    queryFn: () => getTokenizerJob(jobId!),
    enabled: !!jobId,
    refetchInterval: (query) => {
      // Auto-refresh if job is running
      // In TanStack Query v5, access data via query.state.data
      const data = query.state.data;
      if (data?.status === 'RUNNING' || data?.status === 'PENDING') {
        return 2000; // 2 seconds
      }
      return false; // Stop polling when completed/failed
    },
  });
}

/**
 * Create training job mutation
 */
export function useCreateTokenizerJob() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (request: TokenizerTrainingRequest) => createTokenizerTrainingJob(request),
    onSuccess: () => {
      // Invalidate jobs list
      queryClient.invalidateQueries({ queryKey: ['tokenizer-jobs'] });
    },
  });
}

/**
 * Cancel training job mutation
 */
export function useCancelTokenizerJob() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (jobId: string) => cancelTokenizerJob(jobId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['tokenizer-jobs'] });
    },
  });
}

// ============================================================================
// Tokenizers
// ============================================================================

/**
 * List tokenizers
 */
export function useTokenizers(isActive?: boolean) {
  return useQuery({
    queryKey: ['tokenizers', isActive],
    queryFn: () => listTokenizers(isActive),
  });
}

/**
 * Get single tokenizer
 */
export function useTokenizer(tokenizerId: string | null) {
  return useQuery({
    queryKey: ['tokenizer', tokenizerId],
    queryFn: () => getTokenizer(tokenizerId!),
    enabled: !!tokenizerId,
  });
}

/**
 * Update tokenizer mutation
 */
export function useUpdateTokenizer() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({
      tokenizerId,
      data,
    }: {
      tokenizerId: string;
      data: { description?: string; tags?: string[]; is_active?: boolean };
    }) => updateTokenizer(tokenizerId, data),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: ['tokenizer', variables.tokenizerId] });
      queryClient.invalidateQueries({ queryKey: ['tokenizers'] });
    },
  });
}

/**
 * Delete tokenizer mutation
 */
export function useDeleteTokenizer() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ tokenizerId, hardDelete }: { tokenizerId: string; hardDelete?: boolean }) =>
      deleteTokenizer(tokenizerId, hardDelete),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['tokenizers'] });
    },
  });
}

// ============================================================================
// Encode/Decode
// ============================================================================

/**
 * Encode text mutation
 */
export function useEncodeText() {
  return useMutation({
    mutationFn: ({ tokenizerId, text }: { tokenizerId: string; text: string }) =>
      encodeText(tokenizerId, text),
  });
}

/**
 * Decode tokens mutation
 */
export function useDecodeTokens() {
  return useMutation({
    mutationFn: ({ tokenizerId, tokenIds }: { tokenizerId: string; tokenIds: number[] }) =>
      decodeTokens(tokenizerId, tokenIds),
  });
}
