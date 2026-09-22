/**
 * API Client
 * 
 * Backend API ile iletişim için axios wrapper.
 */

import axios, { AxiosInstance, AxiosError } from 'axios';
import type {
  FileRecord,
  FileUploadResponse,
  DocumentRecord,
  DocumentPreview,
  IngestionStats,
  BatchProcessResponse,
  ExportResponse,
} from '@/types';

// API base URL
export const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// Axios instance
const apiClient: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor (logging)
apiClient.interceptors.request.use(
  (config) => {
    console.log(`[API] ${config.method?.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor (error handling)
apiClient.interceptors.response.use(
  (response) => {
    return response;
  },
  (error: AxiosError) => {
    if (error.response) {
      // Backend error response
      console.error('[API Error]', error.response.status, error.response.data);
    } else if (error.request) {
      // Network error
      console.error('[Network Error]', error.message);
    } else {
      console.error('[Request Error]', error.message);
    }
    return Promise.reject(error);
  }
);

// ============================================================================
// Files API
// ============================================================================

export const filesApi = {
  /**
   * Dosya yükle
   */
  async upload(file: File): Promise<FileUploadResponse> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await apiClient.post<FileUploadResponse>(
      '/api/v1/files/upload',
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      }
    );

    return response.data;
  },

  /**
   * Dosyaları listele
   */
  async list(skip = 0, limit = 100): Promise<FileRecord[]> {
    const response = await apiClient.get<FileRecord[]>('/api/v1/files/', {
      params: { skip, limit },
    });

    return response.data;
  },

  /**
   * Dosya detayı
   */
  async get(fileId: string): Promise<FileRecord> {
    const response = await apiClient.get<FileRecord>(`/api/v1/files/${fileId}`);
    return response.data;
  },

  /**
   * Dosya sil
   */
  async delete(fileId: string): Promise<void> {
    await apiClient.delete(`/api/v1/files/${fileId}`);
  },

  /**
   * Dosya metadata güncelle
   */
  async updateMetadata(
    fileId: string,
    updates: {
      training_allowed?: boolean;
      security_level?: string;
      license?: string;
      copyright_status?: string;
      source?: string;
      dataset_version?: string;
    }
  ): Promise<FileRecord> {
    const response = await apiClient.patch<FileRecord>(
      `/api/v1/files/${fileId}`,
      updates
    );
    return response.data;
  },

  /**
   * Dosyayı process et (parse + DocumentRecord oluştur)
   */
  async process(fileId: string): Promise<DocumentRecord> {
    const response = await apiClient.post<DocumentRecord>(
      `/api/v1/files/${fileId}/process`
    );
    return response.data;
  },

  /**
   * Batch processing
   */
  async batchProcess(fileIds: string[]): Promise<BatchProcessResponse> {
    const response = await apiClient.post<BatchProcessResponse>(
      '/api/v1/files/batch-process',
      { file_ids: fileIds }
    );
    return response.data;
  },
};

// ============================================================================
// Datasets API
// ============================================================================

export const datasetsApi = {
  /**
   * Dataset istatistikleri
   */
  async getStats(): Promise<IngestionStats> {
    const response = await apiClient.get<IngestionStats>('/api/v1/datasets/stats');
    return response.data;
  },

  /**
   * Dokümanları listele (preview)
   */
  async listDocuments(params?: {
    skip?: number;
    limit?: number;
    language?: string;
    min_quality?: number;
  }): Promise<DocumentPreview[]> {
    const response = await apiClient.get<DocumentPreview[]>(
      '/api/v1/datasets/documents',
      { params }
    );
    return response.data;
  },

  /**
   * Doküman detayı (full text)
   */
  async getDocument(documentId: string): Promise<DocumentRecord> {
    const response = await apiClient.get<DocumentRecord>(
      `/api/v1/datasets/documents/${documentId}`
    );
    return response.data;
  },

  /**
   * File ID'ye göre doküman getir
   */
  async getDocumentByFile(fileId: string): Promise<DocumentRecord> {
    const response = await apiClient.get<DocumentRecord>(
      `/api/v1/datasets/documents/by-file/${fileId}`
    );
    return response.data;
  },

  /**
   * Parquet export
   */
  async exportParquet(): Promise<ExportResponse> {
    const response = await apiClient.post<ExportResponse>(
      '/api/v1/datasets/export/parquet'
    );
    return response.data;
  },

  /**
   * Pretraining export (JSONL)
   */
  async exportPretraining(minQuality = 0.5): Promise<ExportResponse> {
    const response = await apiClient.post<ExportResponse>(
      '/api/v1/datasets/export/pretraining',
      null,
      {
        params: { min_quality_score: minQuality },
      }
    );
    return response.data;
  },

  /**
   * Yeni dataset versiyonu oluştur
   */
  async createVersion(version: string, description = ''): Promise<ExportResponse> {
    const response = await apiClient.post<ExportResponse>(
      `/api/v1/datasets/versions/${version}`,
      null,
      {
        params: { description },
      }
    );
    return response.data;
  },
};

// ============================================================================
// System API
// ============================================================================

export const systemApi = {
  /**
   * Health check
   */
  async health(): Promise<{ status: string; service: string }> {
    const response = await apiClient.get('/health');
    return response.data;
  },

  /**
   * Sistem bilgisi
   */
  async info(): Promise<any> {
    const response = await apiClient.get('/api/v1/info');
    return response.data;
  },
};


// ============================================================================
// Dataset Compiler API
// ============================================================================

export interface CompilationJobRequest {
  job_name: string;
  dataset_name: string;
  dataset_version?: string;
  document_ids?: string[];
  file_ids?: string[];
  tokenizer_id: string;
  compilation_params?: {
    min_quality_score?: number;
    max_quality_score?: number;
    min_length?: number;
    max_length?: number;
    allow_pii?: boolean;
    mask_pii?: boolean;
    require_training_allowed?: boolean;
    remove_duplicates?: boolean;
  };
}

export interface CompilationJob {
  job_id: string;
  job_name: string;
  status: 'PENDING' | 'RUNNING' | 'COMPLETED' | 'FAILED' | 'CANCELLED';
  progress: number;
  created_at: string;
  started_at?: string;
  completed_at?: string;
  result_metadata: Record<string, any>;
  error?: string;
}

export interface DatasetVersion {
  dataset_id: string;
  name: string;
  version: string;
  num_documents: number;
  total_tokens?: number;
  file_size_bytes?: number;
  is_active: boolean;
  compiled_at?: string;
  tags?: string[];
}

export interface DatasetVersionDetail extends DatasetVersion {
  description?: string;
  schema_version: string;
  compiler_version: string;
  storage_path: string;
  metadata_path?: string;
  source_document_ids: string[];
  source_file_ids: string[];
  tokenizer_id: string;
  compilation_job_id?: string;
  total_chars?: number;
  compilation_params: Record<string, any>;
  filter_stats: Record<string, any>;
  is_snapshot: boolean;
  created_at: string;
  custom_metadata?: Record<string, any>;
}

export const datasetCompilerApi = {
  /**
   * Compilation job oluştur
   */
  async createJob(request: CompilationJobRequest): Promise<CompilationJob> {
    const response = await apiClient.post<CompilationJob>(
      '/api/v1/datasets/compile',
      request
    );
    return response.data;
  },

  /**
   * Compilation job'ları listele
   */
  async listJobs(status?: string, limit: number = 50): Promise<CompilationJob[]> {
    const params = new URLSearchParams();
    if (status) params.append('status', status);
    params.append('limit', limit.toString());

    const response = await apiClient.get<CompilationJob[]>(
      `/api/v1/datasets/compile/jobs?${params.toString()}`
    );
    return response.data;
  },

  /**
   * Compilation job durumunu sorgula
   */
  async getJob(jobId: string): Promise<CompilationJob> {
    const response = await apiClient.get<CompilationJob>(
      `/api/v1/datasets/compile/jobs/${jobId}`
    );
    return response.data;
  },

  /**
   * Compilation job'ı iptal et
   */
  async cancelJob(jobId: string): Promise<void> {
    await apiClient.delete(`/api/v1/datasets/compile/jobs/${jobId}`);
  },

  /**
   * Dataset version'larını listele
   */
  async listVersions(
    name?: string,
    isActive?: boolean,
    limit: number = 50
  ): Promise<DatasetVersion[]> {
    const params = new URLSearchParams();
    if (name) params.append('name', name);
    if (isActive !== undefined) params.append('is_active', isActive.toString());
    params.append('limit', limit.toString());

    const response = await apiClient.get<DatasetVersion[]>(
      `/api/v1/datasets/versions?${params.toString()}`
    );
    return response.data;
  },

  /**
   * Dataset version detayını getir
   */
  async getVersion(datasetId: string): Promise<DatasetVersionDetail> {
    const response = await apiClient.get<DatasetVersionDetail>(
      `/api/v1/datasets/versions/${datasetId}`
    );
    return response.data;
  },

  /**
   * Dataset download URL'i oluştur
   */
  getDownloadUrl(datasetId: string): string {
    return `${API_BASE_URL}/api/v1/datasets/versions/${datasetId}/download`;
  },

  /**
   * Metadata download URL'i oluştur
   */
  getMetadataDownloadUrl(datasetId: string): string {
    return `${API_BASE_URL}/api/v1/datasets/versions/${datasetId}/metadata`;
  },
};

// ============================================================================
// Tokenizer API
// ============================================================================

export interface TokenizerTrainingRequest {
  job_name: string;
  dataset_ids?: string[];
  file_ids?: string[];
  vocab_size: number;
  min_frequency: number;
  special_tokens?: string[];
}

export interface TokenizerJob {
  job_id: string;
  job_name: string;
  status: 'PENDING' | 'RUNNING' | 'COMPLETED' | 'FAILED' | 'CANCELLED';
  progress: number;
  created_at: string;
  started_at?: string;
  completed_at?: string;
  metadata: Record<string, any>;
  error?: string;
}

export interface TokenizerRecord {
  tokenizer_id: string;
  name: string;
  tokenizer_type: string;
  version: string;
  vocab_size: number;
  num_merges?: number;
  special_tokens: string[];
  num_training_documents: number;
  training_duration_seconds?: number;
  is_active: boolean;
  created_at: string;
  description?: string;
  tags?: string[];
}

export interface EncodeResponse {
  text: string;
  token_ids: number[];
  num_tokens: number;
}

export interface DecodeResponse {
  token_ids: number[];
  text: string;
}

/**
 * Create tokenizer training job
 */
export const createTokenizerTrainingJob = async (
  request: TokenizerTrainingRequest
): Promise<TokenizerJob> => {
  const response = await apiClient.post<TokenizerJob>('/api/v1/tokenizer/train', request);
  return response.data;
};

/**
 * List tokenizer training jobs
 */
export const listTokenizerJobs = async (
  status?: string,
  limit: number = 50
): Promise<TokenizerJob[]> => {
  const params = new URLSearchParams();
  if (status) params.append('status', status);
  params.append('limit', limit.toString());
  
  const response = await apiClient.get<TokenizerJob[]>(
    `/api/v1/tokenizer/jobs?${params.toString()}`
  );
  return response.data;
};

/**
 * Get tokenizer training job status
 */
export const getTokenizerJob = async (jobId: string): Promise<TokenizerJob> => {
  const response = await apiClient.get<TokenizerJob>(`/api/v1/tokenizer/jobs/${jobId}`);
  return response.data;
};

/**
 * Cancel tokenizer training job
 */
export const cancelTokenizerJob = async (jobId: string): Promise<void> => {
  await apiClient.delete(`/api/v1/tokenizer/jobs/${jobId}`);
};

/**
 * List tokenizers
 */
export const listTokenizers = async (
  isActive?: boolean,
  limit: number = 50
): Promise<TokenizerRecord[]> => {
  const params = new URLSearchParams();
  if (isActive !== undefined) params.append('is_active', isActive.toString());
  params.append('limit', limit.toString());
  
  const response = await apiClient.get<TokenizerRecord[]>(
    `/api/v1/tokenizer/list?${params.toString()}`
  );
  return response.data;
};

/**
 * Get tokenizer metadata
 */
export const getTokenizer = async (tokenizerId: string): Promise<TokenizerRecord> => {
  const response = await apiClient.get<TokenizerRecord>(`/api/v1/tokenizer/${tokenizerId}`);
  return response.data;
};

/**
 * Encode text to token IDs
 */
export const encodeText = async (
  tokenizerId: string,
  text: string
): Promise<EncodeResponse> => {
  const response = await apiClient.post<EncodeResponse>(
    `/api/v1/tokenizer/${tokenizerId}/encode`,
    { text }
  );
  return response.data;
};

/**
 * Decode token IDs to text
 */
export const decodeTokens = async (
  tokenizerId: string,
  tokenIds: number[]
): Promise<DecodeResponse> => {
  const response = await apiClient.post<DecodeResponse>(
    `/api/v1/tokenizer/${tokenizerId}/decode`,
    { token_ids: tokenIds }
  );
  return response.data;
};

/**
 * Update tokenizer metadata
 */
export const updateTokenizer = async (
  tokenizerId: string,
  data: { description?: string; tags?: string[]; is_active?: boolean }
): Promise<TokenizerRecord> => {
  const response = await apiClient.patch<TokenizerRecord>(
    `/api/v1/tokenizer/${tokenizerId}`,
    data
  );
  return response.data;
};

/**
 * Delete tokenizer
 */
export const deleteTokenizer = async (
  tokenizerId: string,
  hardDelete: boolean = false
): Promise<void> => {
  const params = hardDelete ? '?hard_delete=true' : '';
  await apiClient.delete(`/api/v1/tokenizer/${tokenizerId}${params}`);
};


// ============================================================================
// Training API
// ============================================================================

export interface StartTrainingRequest {
  job_name: string;
  model_name: string;
  job_type: 'PRETRAIN' | 'SFT' | 'SFT_LORA';
  dataset_id?: string;
  tokenizer_id?: string;
  epochs: number;
  batch_size: number;
  learning_rate: number;
  d_model: number;
  n_layers: number;
  n_heads: number;
  max_seq_len: number;
  lora_r?: number;
  lora_alpha?: number;
}

export interface TrainingMetricPoint {
  step: number;
  epoch: number;
  loss: number;
  perplexity?: number;
  lr?: number;
}

export interface TrainingJobResponse {
  job_id: string;
  job_name: string;
  job_type: string;
  status: 'PENDING' | 'RUNNING' | 'COMPLETED' | 'FAILED' | 'CANCELLED';
  model_name: string;
  dataset_id?: string;
  tokenizer_id?: string;
  progress: number;
  current_epoch: number;
  total_epochs: number;
  current_step: number;
  total_steps: number;
  created_at: string;
  started_at?: string;
  completed_at?: string;
  error?: string;
  metrics: TrainingMetricPoint[];
}

export const trainingApi = {
  async start(data: StartTrainingRequest): Promise<TrainingJobResponse> {
    const response = await apiClient.post<TrainingJobResponse>('/api/v1/training/start', data);
    return response.data;
  },
  async listJobs(limit = 20): Promise<TrainingJobResponse[]> {
    const response = await apiClient.get<TrainingJobResponse[]>(`/api/v1/training/jobs?limit=${limit}`);
    return response.data;
  },
  async getJob(jobId: string): Promise<TrainingJobResponse> {
    const response = await apiClient.get<TrainingJobResponse>(`/api/v1/training/jobs/${jobId}`);
    return response.data;
  },
  async cancelJob(jobId: string): Promise<{ status: string; message: string }> {
    const response = await apiClient.post(`/api/v1/training/jobs/${jobId}/cancel`);
    return response.data;
  },
};


// ============================================================================
// Models API
// ============================================================================

export interface ModelItem {
  model_name: string;
  version: string;
  description: string;
  architecture: string;
  parameters: number;
  metrics: Record<string, any>;
  training_config: Record<string, any>;
  tags: string[];
  created_at: string;
  environment: string;
  model_hash?: string;
  file_size_mb?: number;
}

export interface ModelVerifyResponse {
  model_name: string;
  version: string;
  verified: boolean;
  status: 'VALID' | 'CORRUPTED';
  expected_hash: string;
  actual_hash: string;
  file_size_mb: number;
  checkpoint_path: string;
  verified_at: string;
}

export const modelsApi = {
  async list(): Promise<ModelItem[]> {
    const response = await apiClient.get<ModelItem[]>('/api/v1/models');
    return response.data;
  },
  async get(modelName: string): Promise<any> {
    const response = await apiClient.get(`/api/v1/models/${modelName}`);
    return response.data;
  },
  async verify(modelName: string, version?: string): Promise<ModelVerifyResponse> {
    const response = await apiClient.post<ModelVerifyResponse>(
      `/api/v1/models/${modelName}/verify`,
      null,
      { params: version ? { version } : {} }
    );
    return response.data;
  },
  async delete(modelName: string): Promise<any> {
    const response = await apiClient.delete(`/api/v1/models/${modelName}`);
    return response.data;
  },
};


// ============================================================================
// Inference API
// ============================================================================

export interface GenerateRequest {
  prompt: string;
  max_length?: number;
  temperature?: number;
  top_k?: number;
  top_p?: number;
}

export interface GenerateResponse {
  generated_text: string;
  prompt: string;
  tokens_generated: number;
  generation_time_ms: number;
  model_name: string;
}

export interface AttentionInspectRequest {
  text: string;
  layer_idx?: number;
  head_idx?: number;
}

export interface AttentionInspectResponse {
  tokens: string[];
  token_ids: number[];
  num_layers: number;
  num_heads: number;
  selected_layer: number | null;
  selected_head: number | null;
  matrix: number[][];
  all_heads_matrix?: Record<string, number[][]>;
  model_name: string;
}

export interface BeamSearchRequest {
  prompt: string;
  beam_width?: number;
  max_length?: number;
  length_penalty?: number;
  num_return_sequences?: number;
}

export interface BeamHypothesisResponse {
  hypothesis_id: number;
  text: string;
  score: number;
  normalized_score: number;
  length: number;
  tokens: string[];
}

export interface BeamSearchResponse {
  prompt: string;
  beam_width: number;
  hypotheses: BeamHypothesisResponse[];
  model_name: string;
  execution_time_ms: number;
}

export interface NextTokenProbsRequest {
  prompt: string;
  top_k?: number;
  temperature?: number;
}

export interface NextTokenCandidate {
  rank: number;
  token_id: number;
  token_text: string;
  probability: number;
  percentage: number;
  raw_logit: number;
}

export interface NextTokenProbsResponse {
  prompt: string;
  temperature: number;
  candidates: NextTokenCandidate[];
  model_name: string;
}

export const inferenceApi = {
  async load(modelName: string, version?: string): Promise<any> {
    const response = await apiClient.post('/api/v1/inference/load', {
      model_name: modelName,
      version: version,
    });
    return response.data;
  },
  async status(): Promise<{ ready: boolean; model_loaded: boolean; model_name?: string; device?: string }> {
    const response = await apiClient.get('/api/v1/inference/status');
    return response.data;
  },
  async generate(data: GenerateRequest): Promise<GenerateResponse> {
    const response = await apiClient.post<GenerateResponse>('/api/v1/inference/generate', data);
    return response.data;
  },
  async attention(data: AttentionInspectRequest): Promise<AttentionInspectResponse> {
    const response = await apiClient.post<AttentionInspectResponse>('/api/v1/inference/attention', data);
    return response.data;
  },
  async beamSearch(data: BeamSearchRequest): Promise<BeamSearchResponse> {
    const response = await apiClient.post<BeamSearchResponse>('/api/v1/inference/beam-search', data);
    return response.data;
  },
  async nextTokenProbs(data: NextTokenProbsRequest): Promise<NextTokenProbsResponse> {
    const response = await apiClient.post<NextTokenProbsResponse>('/api/v1/inference/next-token-probs', data);
    return response.data;
  },
};


// ============================================================================
// Embeddings API
// ============================================================================

export interface EmbeddingProjectRequest {
  words: string[];
  dimensions?: number;
  model_name?: string;
  normalize?: boolean;
}

export interface ProjectedPoint {
  text: string;
  x: number;
  y: number;
  z?: number | null;
  norm: number;
}

export interface EmbeddingProjectResponse {
  points: ProjectedPoint[];
  dimensions: number;
  explained_variance_ratio: number[];
  d_model: number;
  model_name: string;
}

export interface SimilarityRequest {
  word_a: string;
  word_b: string;
  model_name?: string;
}

export interface SimilarityResponse {
  word_a: string;
  word_b: string;
  cosine_similarity: number;
  angle_degrees: number;
  euclidean_distance: number;
  dot_product: number;
  model_name: string;
}

export interface AnalogyRequest {
  word_a: string;
  word_b: string;
  word_c: string;
  candidates?: string[];
  model_name?: string;
}

export interface AnalogyResponse {
  word_a: string;
  word_b: string;
  word_c: string;
  formula: string;
  top_matches: Array<{ word: string; similarity: number }>;
  model_name: string;
}

export const embeddingsApi = {
  async project(data: EmbeddingProjectRequest): Promise<EmbeddingProjectResponse> {
    const response = await apiClient.post<EmbeddingProjectResponse>('/api/v1/embeddings/project', data);
    return response.data;
  },
  async similarity(data: SimilarityRequest): Promise<SimilarityResponse> {
    const response = await apiClient.post<SimilarityResponse>('/api/v1/embeddings/similarity', data);
    return response.data;
  },
  async analogy(data: AnalogyRequest): Promise<AnalogyResponse> {
    const response = await apiClient.post<AnalogyResponse>('/api/v1/embeddings/analogy', data);
    return response.data;
  },
};

// ============================================================================
// RAG API
// ============================================================================

export interface ChunkItem {
  chunk_id: string;
  text: string;
  chunk_index: number;
  start_char: number;
  end_char: number;
  token_count: number;
  metadata: Record<string, any>;
}

export interface ChunkRequest {
  text: string;
  strategy: 'recursive' | 'sentence' | 'fixed';
  chunk_size: number;
  chunk_overlap: number;
  metadata?: Record<string, any>;
}

export interface ChunkResponse {
  chunks: ChunkItem[];
  total_chunks: number;
  strategy: string;
  total_characters: number;
}

export interface IndexDocumentRequest {
  collection_name?: string;
  document_ids?: string[];
  custom_texts?: Array<{ title: string; text: string }>;
  chunk_strategy?: string;
  chunk_size?: number;
  chunk_overlap?: number;
}

export interface IndexDocumentResponse {
  collection_name: string;
  indexed_chunks: number;
  total_vectors_in_collection: number;
  message: string;
}

export interface SearchResultItem {
  chunk_id: string;
  text: string;
  score: number;
  rank: number;
  metadata: Record<string, any>;
}

export interface SearchRequest {
  query: string;
  collection_name?: string;
  mode?: 'hybrid' | 'dense' | 'sparse';
  top_k?: number;
  alpha?: number;
}

export interface SearchResponse {
  query: string;
  mode: string;
  collection_name: string;
  results: SearchResultItem[];
  count: number;
  latency_ms: number;
}

export interface CitationItem {
  citation_id: number;
  chunk_id: string;
  source_title: string;
  snippet: string;
  score: number;
}

export interface RAGQueryRequest {
  question: string;
  collection_name?: string;
  retrieval_mode?: 'hybrid' | 'dense' | 'sparse';
  top_k?: number;
  alpha?: number;
  max_new_tokens?: number;
  temperature?: number;
}

export interface RAGQueryResponse {
  query: string;
  answer: string;
  retrieved_chunks: SearchResultItem[];
  citations: CitationItem[];
  model_name: string;
  prompt_used: string;
  stats: Record<string, any>;
}

export interface CollectionInfo {
  collection_name: string;
  vector_count: number;
  d_model: number;
  backend: string;
}

export const ragApi = {
  async chunk(data: ChunkRequest): Promise<ChunkResponse> {
    const response = await apiClient.post<ChunkResponse>('/api/v1/rag/chunk', data);
    return response.data;
  },
  async index(data: IndexDocumentRequest): Promise<IndexDocumentResponse> {
    const response = await apiClient.post<IndexDocumentResponse>('/api/v1/rag/index', data);
    return response.data;
  },
  async collections(): Promise<CollectionInfo[]> {
    const response = await apiClient.get<CollectionInfo[]>('/api/v1/rag/collections');
    return response.data;
  },
  async search(data: SearchRequest): Promise<SearchResponse> {
    const response = await apiClient.post<SearchResponse>('/api/v1/rag/search', data);
    return response.data;
  },
  async query(data: RAGQueryRequest): Promise<RAGQueryResponse> {
    const response = await apiClient.post<RAGQueryResponse>('/api/v1/rag/query', data);
    return response.data;
  },
};

// ============================================================================
// Tensor & Math Lab API
// ============================================================================

export interface ShapeAnalyzeRequest {
  shape: number[];
  dtype?: string;
}

export interface ShapeAnalyzeResponse {
  shape: number[];
  rank: number;
  total_elements: number;
  dtype: string;
  element_bytes: number;
  total_bytes: number;
  formatted_memory: string;
  element_strides: number[];
  byte_strides: number[];
  is_contiguous: boolean;
  semantic_interpretation: string;
  memory_comparison: Record<string, { bytes: number; formatted: string; ratio_vs_fp32: number }>;
  dtype_description: string;
}

export interface ReshapeRequest {
  original_shape: number[];
  target_shape: number[];
}

export interface ReshapeResponse {
  success: boolean;
  original_shape: number[];
  target_shape: number[];
  resolved_shape: number[];
  total_elements: number;
  new_strides: number[];
  contiguous: boolean;
  requires_copy: boolean;
  explanation: string;
  error?: string;
}

export interface TransposeRequest {
  shape: number[];
  permutation: number[];
}

export interface TransposeResponse {
  success: boolean;
  original_shape: number[];
  permutation: number[];
  transposed_shape: number[];
  transposed_strides: number[];
  is_contiguous: boolean;
  requires_contiguous_call: boolean;
  note: string;
  error?: string;
}

export interface BroadcastStep {
  axis: number;
  dim_a: number;
  dim_b: number;
  result_dim: number;
  action: string;
  expansion_factor_a: number;
  expansion_factor_b: number;
  compatible: boolean;
}

export interface BroadcastCheckResponse {
  compatible: boolean;
  shape_a: number[];
  shape_b: number[];
  padded_a: number[];
  padded_b: number[];
  result_shape: number[] | null;
  steps: BroadcastStep[];
  error?: string | null;
}

export interface BroadcastSimulateResponse {
  success: boolean;
  operation: string;
  matrix_a: number[][];
  matrix_b: number[][];
  broadcasted_a: number[][];
  broadcasted_b: number[][];
  result_matrix: number[][];
}

export interface MatMulTerm {
  k_index: number;
  a_val: number;
  b_val: number;
  product: number;
}

export interface MatMulResponse {
  success: boolean;
  m: number;
  k: number;
  n: number;
  shape_a: [number, number];
  shape_b: [number, number];
  shape_c: [number, number];
  matrix_a: number[][];
  matrix_b: number[][];
  matrix_c: number[][];
  selected_cell: { row: number; col: number };
  cell_value: number;
  row_vector: number[];
  col_vector: number[];
  pairwise_terms: MatMulTerm[];
  formula_string: string;
  hardware_metrics: {
    total_flops: number;
    read_bytes_fp32: number;
    write_bytes_fp32: number;
    total_memory_bytes: number;
    formatted_memory: string;
    arithmetic_intensity_flops_per_byte: number;
  };
}

export interface ActivationPoint {
  x: number;
  y: number;
  derivative: number;
}

export interface ActivationCurveResponse {
  activation: string;
  formula: string;
  latex: string;
  description: string;
  temperature: number;
  points: ActivationPoint[];
}

export interface SoftmaxElement {
  index: number;
  raw_logit: number;
  scaled_logit: number;
  probability: number;
  percentage: number;
}

export interface SoftmaxTemperatureResponse {
  temperature: number;
  logits: number[];
  elements: SoftmaxElement[];
  entropy_bits: number;
  max_probability: number;
  argmax_index: number;
  interpretation: string;
}

export interface AutogradNode {
  id: string;
  label: string;
  type: 'input' | 'parameter' | 'operation' | 'activation' | 'loss';
  shape: number[];
  forward_val: any;
  grad_val: any;
  layer: number;
}

export interface AutogradEdge {
  from: string;
  to: string;
  label: string;
}

export interface AutogradResponse {
  success: boolean;
  architecture: string;
  loss: number;
  y_target: number[];
  y_pred: number[];
  nodes: AutogradNode[];
  edges: AutogradEdge[];
}

export const tensorLabApi = {
  async analyzeShape(data: ShapeAnalyzeRequest): Promise<ShapeAnalyzeResponse> {
    const response = await apiClient.post<ShapeAnalyzeResponse>('/api/v1/tensor-lab/shape', data);
    return response.data;
  },
  async reshape(data: ReshapeRequest): Promise<ReshapeResponse> {
    const response = await apiClient.post<ReshapeResponse>('/api/v1/tensor-lab/reshape', data);
    return response.data;
  },
  async transpose(data: TransposeRequest): Promise<TransposeResponse> {
    const response = await apiClient.post<TransposeResponse>('/api/v1/tensor-lab/transpose', data);
    return response.data;
  },
  async checkBroadcast(shapeA: number[], shapeB: number[]): Promise<BroadcastCheckResponse> {
    const response = await apiClient.post<BroadcastCheckResponse>('/api/v1/tensor-lab/broadcast', {
      shape_a: shapeA,
      shape_b: shapeB,
    });
    return response.data;
  },
  async simulateBroadcast(matrixA: number[][], matrixB: number[][], operation = 'add'): Promise<BroadcastSimulateResponse> {
    const response = await apiClient.post<BroadcastSimulateResponse>('/api/v1/tensor-lab/broadcast/simulate', {
      matrix_a: matrixA,
      matrix_b: matrixB,
      operation,
    });
    return response.data;
  },
  async matmul(matrixA: number[][], matrixB: number[][], row = 0, col = 0): Promise<MatMulResponse> {
    const response = await apiClient.post<MatMulResponse>('/api/v1/tensor-lab/matmul', {
      matrix_a: matrixA,
      matrix_b: matrixB,
      selected_row: row,
      selected_col: col,
    });
    return response.data;
  },
  async getMatmulSample(m = 3, k = 3, n = 3, preset = 'simple'): Promise<{ matrix_a: number[][]; matrix_b: number[][] }> {
    const response = await apiClient.get('/api/v1/tensor-lab/matmul/sample', {
      params: { m, k, n, preset },
    });
    return response.data;
  },
  async getActivations(): Promise<{ id: string; name: string; llm_usage: string }[]> {
    const response = await apiClient.get('/api/v1/tensor-lab/activations');
    return response.data;
  },
  async getActivationCurve(activation: string, numPoints = 81, temperature = 1.0): Promise<ActivationCurveResponse> {
    const response = await apiClient.post<ActivationCurveResponse>('/api/v1/tensor-lab/activation/curve', {
      activation,
      num_points: numPoints,
      temperature,
    });
    return response.data;
  },
  async computeSoftmaxTemperature(logits: number[], temperature = 1.0): Promise<SoftmaxTemperatureResponse> {
    const response = await apiClient.post<SoftmaxTemperatureResponse>('/api/v1/tensor-lab/activation/softmax', {
      logits,
      temperature,
    });
    return response.data;
  },
  async simulateAutograd(x: number[], yTarget: number[], hiddenDim = 3, activation = 'relu', seed = 42): Promise<AutogradResponse> {
    const response = await apiClient.post<AutogradResponse>('/api/v1/tensor-lab/backprop/simulate', {
      x,
      y_target: yTarget,
      hidden_dim: hiddenDim,
      activation,
      seed,
    });
    return response.data;
  },
};

// ============================================================================
// Transformer Architecture Lab API
// ============================================================================

export interface SinusoidalWave {
  dimension_index: number;
  values: number[];
  type: string;
  wavelength_approx: number;
}

export interface SinusoidalResponse {
  seq_len: number;
  d_model: number;
  pe_matrix: number[][];
  similarity_matrix: number[][];
  waves: SinusoidalWave[];
  description: string;
}

export interface RoPEPosition {
  position: number;
  angle_rad: number;
  angle_deg: number;
  q_rotated: [number, number];
  k_rotated: [number, number];
}

export interface RoPEResponse {
  dim: number;
  max_seq_len: number;
  base: number;
  theta_values: number[];
  positions_data: RoPEPosition[];
  relative_dot_matrix: number[][];
  explanation: string;
}

export interface ALiBiHead {
  head_index: number;
  slope: number;
  bias_matrix: number[][];
}

export interface ALiBiResponse {
  num_heads: number;
  seq_len: number;
  slopes: number[];
  heads: ALiBiHead[];
  description: string;
}

export interface PEComparisonItem {
  name: string;
  authors: string;
  modern_usage: string;
  type: string;
  parameter_overhead: string;
  extrapolation: string;
  mechanism: string;
}

export interface KVCacheHeadGroup {
  kv_head_index: number;
  shared_query_heads: number[];
  query_count: number;
}

export interface KVCacheAnalysisResponse {
  variant_type: string;
  description: string;
  batch_size: number;
  seq_len: number;
  num_query_heads: number;
  num_kv_heads: number;
  head_dim: number;
  num_layers: number;
  dtype: string;
  bytes_per_elem: number;
  queries_per_kv_head: number;
  current_kv_cache: {
    total_bytes: number;
    formatted: string;
    bytes_per_layer: number;
    formatted_per_layer: string;
  };
  comparison: {
    mha_baseline_formatted: string;
    mha_baseline_bytes: number;
    mqa_formatted: string;
    mqa_bytes: number;
    savings_vs_mha_pct: number;
    savings_multiplier: string;
  };
  memory_bandwidth: {
    read_per_step_formatted: string;
    theoretical_throughput_a100: string;
  };
  head_groups: KVCacheHeadGroup[];
}

export interface BlockStage {
  sublayer_name: string;
  shape: number[];
  mean: number;
  std: number;
  min: number;
  max: number;
  description: string;
}

export interface BlockSimulateResponse {
  norm_type: string;
  ffn_type: string;
  norm_placement: string;
  d_model: number;
  d_ff: number;
  stages: BlockStage[];
}

export interface ModelParamsResponse {
  total_parameters: number;
  total_millions: number;
  total_billions: number;
  breakdown: Record<string, number>;
  percentages: Record<string, number>;
  vram_inference: {
    fp16: string;
    fp32: string;
    int8_quantized: string;
    int4_quantized: string;
  };
  vram_training_adamw: string;
}

export interface ArchitecturePreset {
  name: string;
  vocab_size: number;
  d_model: number;
  n_layers: number;
  n_heads: number;
  n_kv_heads: number;
  d_ff: number;
  norm_type: string;
  ffn_type: string;
  tie_word_embeddings: boolean;
}

export const transformerLabApi = {
  async getSinusoidalPE(seqLen = 16, dModel = 32): Promise<SinusoidalResponse> {
    const response = await apiClient.post<SinusoidalResponse>('/api/v1/transformer-lab/positional-encoding/sinusoidal', {
      seq_len: seqLen,
      d_model: dModel,
    });
    return response.data;
  },
  async getRoPE(dim = 16, maxSeqLen = 8, base = 10000.0): Promise<RoPEResponse> {
    const response = await apiClient.post<RoPEResponse>('/api/v1/transformer-lab/positional-encoding/rope', {
      dim,
      max_seq_len: maxSeqLen,
      base,
    });
    return response.data;
  },
  async getALiBi(numHeads = 8, seqLen = 8): Promise<ALiBiResponse> {
    const response = await apiClient.post<ALiBiResponse>('/api/v1/transformer-lab/positional-encoding/alibi', {
      num_heads: numHeads,
      seq_len: seqLen,
    });
    return response.data;
  },
  async comparePE(): Promise<PEComparisonItem[]> {
    const response = await apiClient.get<PEComparisonItem[]>('/api/v1/transformer-lab/positional-encoding/compare');
    return response.data;
  },
  async analyzeAttentionVariants(params: {
    batch_size: number;
    seq_len: number;
    num_query_heads: number;
    num_kv_heads: number;
    head_dim: number;
    num_layers: number;
    dtype?: string;
  }): Promise<KVCacheAnalysisResponse> {
    const response = await apiClient.post<KVCacheAnalysisResponse>('/api/v1/transformer-lab/attention-variants/analyze', params);
    return response.data;
  },
  async simulateBlock(params: {
    norm_type: string;
    ffn_type: string;
    norm_placement: string;
    batch_size?: number;
    seq_len?: number;
    d_model?: number;
    d_ff?: number;
  }): Promise<BlockSimulateResponse> {
    const response = await apiClient.post<BlockSimulateResponse>('/api/v1/transformer-lab/transformer-block/simulate', params);
    return response.data;
  },
  async calculateParams(params: {
    vocab_size: number;
    d_model: number;
    n_layers: number;
    n_heads: number;
    n_kv_heads: number;
    d_ff: number;
    tie_word_embeddings?: boolean;
    ffn_type?: string;
  }): Promise<ModelParamsResponse> {
    const response = await apiClient.post<ModelParamsResponse>('/api/v1/transformer-lab/transformer-block/params', params);
    return response.data;
  },
  async getPresets(): Promise<ArchitecturePreset[]> {
    const response = await apiClient.get<ArchitecturePreset[]>('/api/v1/transformer-lab/transformer-block/presets');
    return response.data;
  },
};

// ============================================================================
// Evaluation & Benchmark Lab API
// ============================================================================

export interface TextMetricInspectRequest {
  candidate: string;
  reference: string;
  max_n?: number;
}

export interface TextMetricInspectResponse {
  candidate_tokens: string[];
  reference_tokens: string[];
  matched_unigrams: string[];
  matched_bigrams: string[];
  matched_trigrams: string[];
  bleu: Record<string, number>;
  brevity_penalty: number;
  candidate_len: number;
  reference_len: number;
  rouge: Record<string, number>;
}

export interface BenchmarkRunRequest {
  model_name: string;
  benchmark_name: string;
  dataset_path?: string;
  max_samples?: number;
  batch_size?: number;
}

export interface BenchmarkResultResponse {
  benchmark_id: string;
  model_name: string;
  benchmark_name: string;
  score: number;
  metrics: Record<string, any>;
  timestamp: string;
  samples_evaluated: number;
}

export interface ModelComparisonRequest {
  model_names: string[];
  benchmark_name: string;
}

export interface ModelComparisonResponse {
  benchmark_name: string;
  comparisons: Array<{
    model_name: string;
    score: number;
    metrics: Record<string, any>;
  }>;
  winner: string;
}

export interface AvailableBenchmarksResponse {
  benchmarks: Record<string, {
    name: string;
    description: string;
    metric: string;
    lower_is_better: boolean;
  }>;
  count: number;
}

export interface BenchmarkSampleQuestion {
  id: string;
  input: string;
  target: string;
  domain: string;
  category: string;
  difficulty?: string;
  numeric_answer?: number;
  steps?: number;
  keywords?: string[];
}

export interface RadarDimensionScore {
  dimension_key: string;
  dimension_name: string;
  score: number;
  raw_metric?: string;
  raw_score?: number;
  name?: string;
}

export interface RadarModelScore {
  model_name: string;
  overall_average: number;
  dimensions: RadarDimensionScore[];
  overall_score?: number;
}

export interface RadarComparisonRequest {
  model_names: string[];
}

export interface RadarComparisonResponse {
  models: RadarModelScore[];
  dimensions: string[];
  winner_by_dimension: Record<string, string>;
  overall_winner: string;
}

export const evaluationApi = {
  async inspectText(payload: TextMetricInspectRequest): Promise<TextMetricInspectResponse> {
    const response = await apiClient.post<TextMetricInspectResponse>('/api/v1/evaluation/inspect-text', payload);
    return response.data;
  },
  async runBenchmark(payload: BenchmarkRunRequest): Promise<BenchmarkResultResponse> {
    const response = await apiClient.post<BenchmarkResultResponse>('/api/v1/evaluation/run', payload);
    return response.data;
  },
  async getResults(params?: {
    model_name?: string;
    benchmark_name?: string;
    limit?: number;
  }): Promise<BenchmarkResultResponse[]> {
    const response = await apiClient.get<BenchmarkResultResponse[]>('/api/v1/evaluation/results', { params });
    return response.data;
  },
  async getResult(benchmarkId: string): Promise<BenchmarkResultResponse> {
    const response = await apiClient.get<BenchmarkResultResponse>(`/api/v1/evaluation/results/${benchmarkId}`);
    return response.data;
  },
  async deleteResult(benchmarkId: string): Promise<{ message: string }> {
    const response = await apiClient.delete<{ message: string }>(`/api/v1/evaluation/results/${benchmarkId}`);
    return response.data;
  },
  async compareModels(payload: ModelComparisonRequest): Promise<ModelComparisonResponse> {
    const response = await apiClient.post<ModelComparisonResponse>('/api/v1/evaluation/compare', payload);
    return response.data;
  },
  async getBenchmarks(): Promise<AvailableBenchmarksResponse> {
    const response = await apiClient.get<AvailableBenchmarksResponse>('/api/v1/evaluation/benchmarks');
    return response.data;
  },
  async getModelMetrics(modelName: string): Promise<Record<string, any>> {
    const response = await apiClient.get<Record<string, any>>(`/api/v1/evaluation/metrics/${modelName}`);
    return response.data;
  },
  async getSampleQuestions(benchmarkName: string): Promise<BenchmarkSampleQuestion[]> {
    const response = await apiClient.get<BenchmarkSampleQuestion[]>('/api/v1/evaluation/benchmarks/sample-questions', {
      params: { benchmark_name: benchmarkName },
    });
    return response.data;
  },
  async getRadarComparison(payload: RadarComparisonRequest): Promise<RadarComparisonResponse> {
    const response = await apiClient.post<RadarComparisonResponse>('/api/v1/evaluation/radar-comparison', payload);
    return response.data;
  },
};

// ============================================================================
// Guided Journey & Knowledge Map API
// ============================================================================

export interface MilestoneQuestion {
  id: string;
  stage_id: string;
  question: string;
  options: string[];
  correct_index: number;
  explanation: string;
  math_intuition?: string;
}

export interface JourneyStage {
  id: string;
  order: number;
  title: string;
  category: string;
  summary: string;
  concepts: string[];
  objectives: string[];
  prerequisites: string[];
  lab_url?: string;
  lab_title?: string;
  questions: MilestoneQuestion[];
}

export interface KnowledgeNode {
  id: string;
  label: string;
  stage_id: string;
  category: string;
  level: number;
  lab_url?: string;
}

export interface KnowledgeEdge {
  id: string;
  source: string;
  target: string;
  label?: string;
}

export interface KnowledgeGraphResponse {
  nodes: KnowledgeNode[];
  edges: KnowledgeEdge[];
}

export interface GlossaryTerm {
  term: string;
  category: string;
  short_def: string;
  detailed_explanation: string;
  formula?: string;
  related_lab_url?: string;
  related_stage_id?: string;
}

export interface CheckQuestionRequest {
  question_id: string;
  selected_option: number;
}

export interface CheckQuestionResponse {
  question_id: string;
  is_correct: boolean;
  selected_option: number;
  correct_index: number;
  explanation: string;
  math_intuition?: string;
}

export const journeyApi = {
  async getCurriculum(): Promise<JourneyStage[]> {
    const response = await apiClient.get<JourneyStage[]>('/api/v1/journey/curriculum');
    return response.data;
  },
  async getStage(stageId: string): Promise<JourneyStage> {
    const response = await apiClient.get<JourneyStage>(`/api/v1/journey/curriculum/${stageId}`);
    return response.data;
  },
  async getGraph(): Promise<KnowledgeGraphResponse> {
    const response = await apiClient.get<KnowledgeGraphResponse>('/api/v1/journey/graph');
    return response.data;
  },
  async getGlossary(params?: { search?: string; category?: string }): Promise<GlossaryTerm[]> {
    const response = await apiClient.get<GlossaryTerm[]>('/api/v1/journey/glossary', { params });
    return response.data;
  },
  async checkQuestion(payload: CheckQuestionRequest): Promise<CheckQuestionResponse> {
    const response = await apiClient.post<CheckQuestionResponse>('/api/v1/journey/check-question', payload);
    return response.data;
  },
};

// ============================================================================
// Systems for AI & Hardware Lab API
// ============================================================================

export interface HardwarePreset {
  name: string;
  category: string;
  peak_tflops: number;
  peak_bandwidth_gbs: number;
  vram_gb: number;
  description: string;
}

export interface RooflinePoint {
  operational_intensity: number;
  attainable_tflops: number;
}

export interface RooflineAnalyzeRequest {
  hardware_name?: string;
  peak_tflops?: number;
  peak_bandwidth_gbs?: number;
  workload_name?: string;
  workload_flops?: number;
  workload_bytes?: number;
  custom_operational_intensity?: number;
}

export interface RooflineAnalyzeResponse {
  hardware: HardwarePreset;
  workload_name: string;
  operational_intensity: number;
  knee_point_intensity: number;
  regime: 'Memory-Bound' | 'Compute-Bound';
  attainable_tflops: number;
  efficiency_pct: number;
  curve_points: RooflinePoint[];
  diagnosis: string;
}

export interface QuantizationSimulateRequest {
  distribution_type?: 'normal' | 'uniform' | 'outlier';
  num_elements?: number;
  outlier_ratio?: number;
  outlier_magnitude?: number;
}

export interface QuantizationFormatResult {
  bits: number;
  dtype: string;
  mse: number;
  mae: number;
  snr_db: number;
  compression_ratio: number;
  original_memory_kb: number;
  quantized_memory_kb: number;
  scale?: number;
  zero_point?: number;
}

export interface HistogramBin {
  bin_center: number;
  count: number;
  density: number;
}

export interface QuantizationSimulateResponse {
  distribution_type: string;
  num_elements: number;
  original_stats: {
    min: number;
    max: number;
    mean: number;
    std: number;
  };
  formats: Record<string, QuantizationFormatResult>;
  histogram_original: HistogramBin[];
  histogram_int8: HistogramBin[];
  histogram_int4: HistogramBin[];
  insights: string[];
}

export interface GPUMemorySimulateRequest {
  param_count_billions: number;
  precision?: 'fp32' | 'fp16' | 'bf16' | 'int8' | 'int4';
  mode?: 'training' | 'inference';
  batch_size?: number;
  sequence_length?: number;
  hidden_size?: number;
  num_layers?: number;
  num_heads?: number;
  num_kv_heads?: number;
  optimizer?: 'adamw' | 'adamw_8bit' | 'sgd';
  activation_checkpointing?: boolean;
  gpu_vram_gb?: number;
}

export interface MemoryComponentBreakdown {
  weights_gb: number;
  gradients_gb: number;
  optimizer_gb: number;
  activations_gb: number;
  kv_cache_gb: number;
  cuda_overhead_gb: number;
}

export interface GPUMemorySimulateResponse {
  mode: 'training' | 'inference';
  param_count_billions: number;
  precision: string;
  gpu_vram_gb: number;
  components: MemoryComponentBreakdown;
  total_memory_gb: number;
  memory_utilization_pct: number;
  fits_in_gpu: boolean;
  oom_risk: 'LOW' | 'WARNING' | 'CRITICAL_OOM';
  recommendations: string[];
}

export const systemsLabApi = {
  async getPresets(): Promise<HardwarePreset[]> {
    const response = await apiClient.get<HardwarePreset[]>('/api/v1/systems-lab/roofline/presets');
    return response.data;
  },
  async analyzeRoofline(payload: RooflineAnalyzeRequest): Promise<RooflineAnalyzeResponse> {
    const response = await apiClient.post<RooflineAnalyzeResponse>('/api/v1/systems-lab/roofline/analyze', payload);
    return response.data;
  },
  async simulateQuantization(payload: QuantizationSimulateRequest): Promise<QuantizationSimulateResponse> {
    const response = await apiClient.post<QuantizationSimulateResponse>('/api/v1/systems-lab/quantization/simulate', payload);
    return response.data;
  },
  async simulateMemory(payload: GPUMemorySimulateRequest): Promise<GPUMemorySimulateResponse> {
    const response = await apiClient.post<GPUMemorySimulateResponse>('/api/v1/systems-lab/memory/simulate', payload);
    return response.data;
  },
};

// ============================================================================
// Synthetic Data Generation & Filtering Lab API
// ============================================================================

export interface SyntheticSample {
  id: string;
  instruction: string;
  input_context?: string;
  response: string;
  paradigm: string;
  domain: string;
  complexity: string;
  metrics: {
    composite_score: number;
    verdict: 'ACCEPT' | 'NEEDS_REVISION' | 'REJECT';
    verdict_reasons?: string[];
    perplexity: number;
    repetition_ratio_2g: number;
    repetition_ratio_4g: number;
    lexical_diversity_ttr: number;
    word_count: number;
    char_count: number;
    pii_count: number;
    pii_matches?: Array<{
      pii_type: string;
      value: string;
      start: number;
      end: number;
      masked_value: string;
    }>;
    subscores?: {
      fluency: number;
      repetition: number;
      length: number;
      formatting: number;
      safety: number;
    };
  };
  rejection_stage?: string | null;
  rejection_reason?: string | null;
}

export interface SyntheticParadigmMeta {
  id: string;
  name: string;
  description: string;
  icon: string;
}

export interface FilterProfileMeta {
  name: string;
  min_perplexity: number;
  max_perplexity: number;
  min_words: number;
  max_words: number;
  max_repetition_ratio: number;
  min_quality_score: number;
  pii_action: string;
  max_jaccard_similarity: number;
}

export interface SyntheticTemplatesResponse {
  paradigms: SyntheticParadigmMeta[];
  domains: Array<{ id: string; name: string }>;
  complexities: string[];
  preset_profiles: Record<string, FilterProfileMeta>;
}

export interface SyntheticGenerateRequest {
  paradigm: string;
  domain: string;
  complexity: string;
  count: number;
  include_edge_cases?: boolean;
}

export interface SyntheticGenerateResponse {
  total_generated: number;
  paradigm: string;
  domain: string;
  complexity: string;
  samples: SyntheticSample[];
}

export interface FilterThresholdsInput {
  min_perplexity: number;
  max_perplexity: number;
  min_words: number;
  max_words: number;
  max_repetition_ratio: number;
  min_quality_score: number;
  pii_action: string;
  max_jaccard_similarity: number;
}

export interface FilterStageMetric {
  stage_name: string;
  input_count: number;
  passed_count: number;
  rejected_count: number;
  retention_rate: number;
}

export interface FilterPipelineResponse {
  total_input_count: number;
  passed_count: number;
  rejected_count: number;
  final_yield_pct: number;
  avg_initial_quality: number;
  avg_final_quality: number;
  funnel_stages: FilterStageMetric[];
  passed_samples: SyntheticSample[];
  rejected_samples: SyntheticSample[];
}

export interface ScoreSampleRequest {
  instruction: string;
  response: string;
  input_context?: string;
}

export interface ExportSyntheticRequest {
  samples: SyntheticSample[];
  dataset_name: string;
  export_format: string;
}

export interface ExportSyntheticResponse {
  success: boolean;
  dataset_name: string;
  export_format: string;
  total_samples: number;
  approx_tokens: number;
  preview_jsonl: string[];
  message: string;
}

export interface IngestToDatasetRequest {
  samples: SyntheticSample[];
  dataset_name: string;
  domain?: string;
  paradigm?: string;
  training_allowed?: boolean;
}

export interface IngestToDatasetResponse {
  success: boolean;
  file_id: string;
  document_count: number;
  dataset_name: string;
  avg_quality_score: number;
  message: string;
}

export const syntheticLabApi = {
  async getTemplates(): Promise<SyntheticTemplatesResponse> {
    const response = await apiClient.get<SyntheticTemplatesResponse>('/api/v1/synthetic-lab/templates');
    return response.data;
  },
  async generateSamples(payload: SyntheticGenerateRequest): Promise<SyntheticGenerateResponse> {
    const response = await apiClient.post<SyntheticGenerateResponse>('/api/v1/synthetic-lab/generate', payload);
    return response.data;
  },
  async filterSamples(payload: {
    samples: SyntheticSample[];
    thresholds?: FilterThresholdsInput;
  }): Promise<FilterPipelineResponse> {
    const response = await apiClient.post<FilterPipelineResponse>('/api/v1/synthetic-lab/filter', payload);
    return response.data;
  },
  async scoreSample(payload: ScoreSampleRequest): Promise<SyntheticSample['metrics']> {
    const response = await apiClient.post<SyntheticSample['metrics']>('/api/v1/synthetic-lab/score-sample', payload);
    return response.data;
  },
  async exportDataset(payload: ExportSyntheticRequest): Promise<ExportSyntheticResponse> {
    const response = await apiClient.post<ExportSyntheticResponse>('/api/v1/synthetic-lab/export', payload);
    return response.data;
  },
async ingestToDataset(payload: IngestToDatasetRequest): Promise<IngestToDatasetResponse> {
    const response = await apiClient.post<IngestToDatasetResponse>('/api/v1/synthetic-lab/ingest-to-dataset', payload);
    return response.data;
  },
};

// ============================================================================
// Neural Network & Backpropagation Lab API
// ============================================================================

export interface MLPNode {
  id: string;
  label: string;
  layer_idx: number;
  neuron_idx: number;
  layer_type: 'input' | 'hidden' | 'output';
  pre_activation_z: number | null;
  post_activation_a: number | null;
  gradient_z: number | null;
  gradient_a: number | null;
  target_val?: number;
  is_dead: boolean;
}

export interface MLPSynapse {
  id: string;
  source_id: string;
  target_id: string;
  source_layer: number;
  target_layer: number;
  weight: number;
  gradient_w: number;
  weight_update: number;
}

export interface ChainRuleStep {
  step: number;
  layer: string;
  formula: string;
  latex: string;
  grad_norm?: number;
  values?: number[];
  bias_grads?: number[];
  description: string;
}

export interface LayerHealth {
  layer_idx: number;
  max_grad: number;
  mean_grad: number;
  status: 'HEALTHY' | 'VANISHING_RISK' | 'EXPLODING_RISK';
}

export interface MLPSimulateRequest {
  inputs: number[];
  targets: number[];
  hidden_dims?: number[];
  activation?: string;
  loss_function?: string;
  learning_rate?: number;
  seed?: number;
}

export interface MLPSimulateResponse {
  loss: number;
  loss_function: string;
  activation: string;
  learning_rate: number;
  architecture: number[];
  predictions: number[];
  targets: number[];
  nodes: MLPNode[];
  synapses: MLPSynapse[];
  chain_rule_steps: ChainRuleStep[];
  layer_health: LayerHealth[];
  dead_neurons: string[];
  overall_health: 'HEALTHY' | 'VANISHING_RISK' | 'EXPLODING_RISK' | 'DEAD_NEURONS_DETECTED';
  activation_derivative_formula: string;
}

export interface OptimizerTrajectoryPoint {
  step: number;
  x: number;
  y: number;
  loss: number;
  grad_norm: number;
}

export interface OptimizerResult {
  name: string;
  color: string;
  description: string;
  trajectory: OptimizerTrajectoryPoint[];
  final_position: [number, number];
  final_loss: number;
  initial_loss: number;
  loss_reduction_pct: number;
  total_distance: number;
  distance_to_optimum: number;
}

export interface LandscapeContours {
  landscape_key: string;
  x_range: [number, number];
  y_range: [number, number];
  x_vals: number[];
  y_vals: number[];
  z_grid: number[][];
  z_min: number;
  z_max: number;
  optimum: [number, number];
}

export interface OptimizerRaceRequest {
  landscape: string;
  optimizers?: string[];
  start_x?: number;
  start_y?: number;
  learning_rate?: number;
  momentum?: number;
  weight_decay?: number;
  steps?: number;
}

export interface OptimizerRaceResponse {
  landscape: {
    key: string;
    name: string;
    formula: string;
    description: string;
    optimum: [number, number];
    start_position: [number, number];
  };
  parameters: {
    learning_rate: number;
    momentum: number;
    weight_decay: number;
    steps: number;
  };
  optimizers: Record<string, OptimizerResult>;
  winner: string;
  winner_name: string;
  contours: LandscapeContours;
}

export interface ActivationCurveData {
  name: string;
  derivative_formula: string;
  fx: number[];
  dfx: number[];
  max_derivative: number;
  vanishing_notes: string[];
}

export interface ActivationCurvesResponse {
  x_vals: number[];
  activations: Record<string, ActivationCurveData>;
}

export interface LandscapesListResponse {
  landscapes: Record<string, {
    key: string;
    name: string;
    formula: string;
    description: string;
    default_start: [number, number];
    optimum: [number, number];
    x_range: [number, number];
    y_range: [number, number];
  }>;
  optimizers: Record<string, {
    name: string;
    color: string;
    description: string;
  }>;
}

export const nnLabApi = {
  async simulateMLP(payload: MLPSimulateRequest): Promise<MLPSimulateResponse> {
    const response = await apiClient.post<MLPSimulateResponse>('/api/v1/nn-lab/mlp/simulate', payload);
    return response.data;
  },
  async raceOptimizers(payload: OptimizerRaceRequest): Promise<OptimizerRaceResponse> {
    const response = await apiClient.post<OptimizerRaceResponse>('/api/v1/nn-lab/optimizers/race', payload);
    return response.data;
  },
  async getLandscapes(): Promise<LandscapesListResponse> {
    const response = await apiClient.get<LandscapesListResponse>('/api/v1/nn-lab/landscapes');
    return response.data;
  },
  async getActivations(numPoints: number = 80): Promise<ActivationCurvesResponse> {
    const response = await apiClient.get<ActivationCurvesResponse>('/api/v1/nn-lab/activations', {
      params: { num_points: numPoints },
    });
    return response.data;
  },
};

// ============================================================================
// Math Lab API
// ============================================================================

export interface MatrixMultiplyRequest {
  matrix_a: number[][];
  matrix_b: number[][];
  detailed_cell?: number[];
}

export interface MatrixOperationRequest {
  matrix: number[][];
}

export interface VectorDotProductRequest {
  vector_a: number[];
  vector_b: number[];
}

export interface DerivativeRequest {
  function_name: string;
  x_min?: number;
  x_max?: number;
  num_points?: number;
}

export interface GradientDescentRequest {
  function_name: string;
  initial_x?: number;
  learning_rate?: number;
  max_iterations?: number;
  tolerance?: number;
}

export interface ChainRuleRequest {
  outer_function: string;
  inner_function: string;
  x_value: number;
}

export interface MatrixOperationResult {
  success: boolean;
  result?: number[][];
  steps?: Array<{
    step: number;
    title: string;
    description: string;
    formula?: string;
    cell?: number[];
    cell_calculation?: Array<{
      k: number;
      a_value: number;
      b_value: number;
      product: number;
      running_sum: number;
    }>;
    flops?: number;
    memory_reads?: number;
    memory_writes?: number;
    final_value?: number;
  }>;
  properties?: {
    shape_a?: number[];
    shape_b?: number[];
    shape_result?: number[];
    result_shape?: number[];
    original_shape?: number[];
    transposed_shape?: number[];
    total_flops?: number;
    result_min?: number;
    result_max?: number;
    result_mean?: number;
    frobenius_norm?: number;
    is_square?: boolean;
    is_symmetric?: boolean;
    determinant?: number;
    condition_number?: number;
    is_well_conditioned?: boolean;
  };
  error?: string;
}

export interface VectorOperationResult {
  success: boolean;
  result?: number[];
  steps?: Array<Record<string, unknown>>;
  properties?: {
    dimension?: number;
    magnitude_a?: number;
    magnitude_b?: number;
    is_orthogonal?: boolean;
    cosine_similarity?: number;
    angle_degrees?: number;
    angle_radians?: number;
  };
  error?: string;
}

export interface DerivativeResult {
  success: boolean;
  x_values: number[];
  y_values: number[];
  dy_dx_values: number[];
  formula: string;
  derivative_formula: string;
  error?: string;
}

export interface GradientDescentStep {
  iteration: number;
  x: number;
  y: number;
  gradient: number;
  step_size: number;
  update_formula: string;
}

export interface GradientDescentResult {
  success: boolean;
  history: GradientDescentStep[];
  final_x: number;
  final_y: number;
  converged: boolean;
  iterations: number;
  error?: string;
}

export interface ChainRuleStep {
  step: number;
  title: string;
  formula: string;
  derivative?: string;
  value_at_x?: number;
  derivative_at_x?: number;
  value_at_g_x?: number;
  derivative_at_g_x?: number;
  calculation?: string;
  result?: number;
  total_derivative?: number;
}

export interface ChainRuleResult {
  success: boolean;
  x_value: number;
  composite_formula: string;
  steps: ChainRuleStep[];
  final_derivative: number;
  error?: string;
}

export interface EigenvaluesResult {
  success: boolean;
  shape?: number[];
  eigenvalues?: number[];
  eigenvectors?: number[][];
  eigen_pairs?: Array<{
    index: number;
    eigenvalue: number | { real: number; imag: number };
    eigenvector: number[] | Array<{ real: number; imag: number }>;
    magnitude: number;
  }>;
  trace?: number;
  determinant?: number;
  is_real?: boolean;
  error?: string;
}

export const mathLabApi = {
  multiplyMatrices: async (request: MatrixMultiplyRequest): Promise<MatrixOperationResult> => {
    const response = await apiClient.post<MatrixOperationResult>('/api/v1/math-lab/matrix/multiply', request);
    return response.data;
  },
  transposeMatrix: async (request: MatrixOperationRequest): Promise<MatrixOperationResult> => {
    const response = await apiClient.post<MatrixOperationResult>('/api/v1/math-lab/matrix/transpose', request);
    return response.data;
  },
  inverseMatrix: async (request: MatrixOperationRequest): Promise<MatrixOperationResult> => {
    const response = await apiClient.post<MatrixOperationResult>('/api/v1/math-lab/matrix/inverse', request);
    return response.data;
  },
  calculateDeterminant: async (request: MatrixOperationRequest): Promise<{ success: boolean; determinant?: number; is_singular?: boolean; error?: string }> => {
    const response = await apiClient.post('/api/v1/math-lab/matrix/determinant', request);
    return response.data;
  },
  calculateEigenvalues: async (request: MatrixOperationRequest): Promise<EigenvaluesResult> => {
    const response = await apiClient.post<EigenvaluesResult>('/api/v1/math-lab/matrix/eigenvalues', request);
    return response.data;
  },
  vectorDotProduct: async (request: VectorDotProductRequest): Promise<VectorOperationResult> => {
    const response = await apiClient.post<VectorOperationResult>('/api/v1/math-lab/vector/dot', request);
    return response.data;
  },
  getSupportedFunctions: async () => {
    const response = await apiClient.get('/api/v1/math-lab/calculus/functions');
    return response.data;
  },
  computeDerivative: async (request: DerivativeRequest): Promise<DerivativeResult> => {
    const response = await apiClient.post<DerivativeResult>('/api/v1/math-lab/calculus/derivative', request);
    return response.data;
  },
  simulateGradientDescent: async (request: GradientDescentRequest): Promise<GradientDescentResult> => {
    const response = await apiClient.post<GradientDescentResult>('/api/v1/math-lab/calculus/gradient-descent', request);
    return response.data;
  },
  demonstrateChainRule: async (request: ChainRuleRequest): Promise<ChainRuleResult> => {
    const response = await apiClient.post<ChainRuleResult>('/api/v1/math-lab/calculus/chain-rule', request);
    return response.data;
  },
  getMatrixPresets: async () => {
    const response = await apiClient.get('/api/v1/math-lab/presets/matrices');
    return response.data;
  },
  getVectorPresets: async () => {
    const response = await apiClient.get('/api/v1/math-lab/presets/vectors');
    return response.data;
  },
};

// ============================================================================
// Export unified API client
// ============================================================================

export const api = {
  files: filesApi,
  datasets: datasetsApi,
  datasetCompiler: datasetCompilerApi,
  training: trainingApi,
  models: modelsApi,
  inference: inferenceApi,
  embeddings: embeddingsApi,
  rag: ragApi,
  tensorLab: tensorLabApi,
  transformerLab: transformerLabApi,
  evaluation: evaluationApi,
  journey: journeyApi,
  systemsLab: systemsLabApi,
  syntheticLab: syntheticLabApi,
  nnLab: nnLabApi,
  mathLab: mathLabApi,
  system: systemApi,
};

export default api;

