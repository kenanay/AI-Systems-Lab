/**
 * Frontend Type Definitions
 * 
 * Backend API'den gelen veri tipleri.
 */

// File types
export interface FileRecord {
  file_id: string;
  original_name: string;
  relative_path: string;
  mime_type: string;
  size_bytes: number;
  sha256: string;
  parser_name?: string;
  parser_version?: string;
  schema_version?: string;
  security_level: string;
  pii_detected: boolean;
  license?: string;
  copyright_status?: string;
  training_allowed: boolean;
  created_at: string;
  dataset_version?: string;
  source?: string;
  language?: string;
  quality_score?: number;
}

export interface FileUploadResponse {
  file_id: string;
  filename: string;
  size_bytes: number;
  mime_type: string;
  sha256: string;
  is_duplicate: boolean;
  message: string;
}

// Document types
export interface DocumentRecord {
  document_id: string;
  file_id: string;
  title?: string;
  text: string;
  language?: string;
  char_count?: number;
  word_count?: number;
  line_count?: number;
  quality_score?: number;
  is_empty: boolean;
  is_duplicate: boolean;
  created_at: string;
}

export interface DocumentPreview {
  document_id: string;
  file_id: string;
  title?: string;
  text_preview: string;
  full_text_length: number;
  language?: string;
  char_count?: number;
  word_count?: number;
  created_at: string;
}

// Statistics types
export interface IngestionStats {
  total_files: number;
  total_documents: number;
  total_size_bytes: number;
  files_by_type: Record<string, number>;
  files_by_language: Record<string, number>;
  avg_quality_score?: number;
  pii_detected_count: number;
  training_allowed_count: number;
}

// API Response types
export interface ApiResponse<T> {
  data?: T;
  error?: string;
  message?: string;
}

export interface BatchProcessResponse {
  status: string;
  results: {
    total: number;
    successful: number;
    failed: number;
    skipped: number;
    errors: Array<{
      file_id: string;
      error: string;
    }>;
  };
  message: string;
}

export interface ExportResponse {
  status: string;
  output_path?: string;
  file_size_bytes?: number;
  security_filters?: {
    pii_filtered: boolean;
    training_allowed_required: boolean;
    min_quality_score: number;
  };
  message: string;
}

// UI types
export interface UploadProgress {
  file_id?: string;
  filename: string;
  progress: number;
  status: 'pending' | 'uploading' | 'processing' | 'success' | 'error';
  error?: string;
}
