/**
 * File Upload Component
 * 
 * Drag & drop, multi-file upload, progress tracking.
 */

'use client';

import { useCallback, useState } from 'react';
import { Upload, X, FileText, CheckCircle, AlertCircle } from 'lucide-react';
import { useUploadFile, useProcessFile } from '@/hooks/useFiles';
import { formatFileSize, getFileIcon } from '@/lib/utils';
import type { UploadProgress } from '@/types';

interface FileUploadProps {
  onUploadComplete?: (fileId: string) => void;
  autoProcess?: boolean;
}

export function FileUpload({ onUploadComplete, autoProcess = false }: FileUploadProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [uploads, setUploads] = useState<UploadProgress[]>([]);

  const uploadMutation = useUploadFile();
  const processMutation = useProcessFile();

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback(
    async (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);

      const files = Array.from(e.dataTransfer.files);
      await handleFiles(files);
    },
    []
  );

  const handleFileInput = useCallback(
    async (e: React.ChangeEvent<HTMLInputElement>) => {
      const files = e.target.files ? Array.from(e.target.files) : [];
      await handleFiles(files);
      
      // Reset input
      e.target.value = '';
    },
    []
  );

  const handleFiles = async (files: File[]) => {
    for (const file of files) {
      // Upload state ekle
      const uploadState: UploadProgress = {
        filename: file.name,
        progress: 0,
        status: 'pending',
      };

      setUploads((prev) => [...prev, uploadState]);

      try {
        // 1. Upload başla
        setUploads((prev) =>
          prev.map((u) =>
            u.filename === file.name
              ? { ...u, status: 'uploading', progress: 50 }
              : u
          )
        );

        const response = await uploadMutation.mutateAsync(file);

        // 2. Upload tamamlandı
        setUploads((prev) =>
          prev.map((u) =>
            u.filename === file.name
              ? {
                  ...u,
                  file_id: response.file_id,
                  status: response.is_duplicate ? 'success' : 'uploading',
                  progress: response.is_duplicate ? 100 : 75,
                }
              : u
          )
        );

        // Duplicate ise process etme
        if (response.is_duplicate) {
          onUploadComplete?.(response.file_id);
          continue;
        }

        // 3. Auto-process ise parse et
        if (autoProcess) {
          setUploads((prev) =>
            prev.map((u) =>
              u.filename === file.name
                ? { ...u, status: 'processing', progress: 90 }
                : u
            )
          );

          await processMutation.mutateAsync(response.file_id);
        }

        // 4. Başarılı
        setUploads((prev) =>
          prev.map((u) =>
            u.filename === file.name
              ? { ...u, status: 'success', progress: 100 }
              : u
          )
        );

        onUploadComplete?.(response.file_id);

      } catch (error: any) {
        console.error('Upload error:', error);

        setUploads((prev) =>
          prev.map((u) =>
            u.filename === file.name
              ? {
                  ...u,
                  status: 'error',
                  error: error.response?.data?.detail || error.message || 'Upload failed',
                }
              : u
          )
        );
      }
    }
  };

  const removeUpload = (filename: string) => {
    setUploads((prev) => prev.filter((u) => u.filename !== filename));
  };

  const clearCompleted = () => {
    setUploads((prev) => prev.filter((u) => u.status !== 'success'));
  };

  return (
    <div className="w-full">
      {/* Drop Zone */}
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className={`
          relative border-2 border-dashed rounded-lg p-8 text-center transition-colors
          ${
            isDragging
              ? 'border-blue-500 bg-blue-50'
              : 'border-gray-300 bg-gray-50 hover:border-gray-400'
          }
        `}
      >
        <input
          type="file"
          multiple
          onChange={handleFileInput}
          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
          accept=".txt,.md,.pdf"
        />

        <Upload className="mx-auto h-12 w-12 text-gray-400 mb-4" />
        
        <p className="text-lg font-medium text-gray-700 mb-2">
          Dosyaları sürükle ve bırak
        </p>
        <p className="text-sm text-gray-500 mb-4">
          veya tıklayarak seç
        </p>
        
        <div className="flex items-center justify-center gap-4 text-xs text-gray-400">
          <span>TXT</span>
          <span>•</span>
          <span>MD</span>
          <span>•</span>
          <span>PDF</span>
        </div>

        {autoProcess && (
          <div className="mt-4 text-xs text-blue-600">
            ✓ Otomatik parse etme aktif
          </div>
        )}
      </div>

      {/* Upload List */}
      {uploads.length > 0 && (
        <div className="mt-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-medium text-gray-700">
              Yüklenenler ({uploads.length})
            </h3>
            {uploads.some((u) => u.status === 'success') && (
              <button
                onClick={clearCompleted}
                className="text-xs text-gray-500 hover:text-gray-700"
              >
                Tamamlananları temizle
              </button>
            )}
          </div>

          <div className="space-y-3">
            {uploads.map((upload, index) => (
              <UploadItem
                key={`${upload.filename}-${index}`}
                upload={upload}
                onRemove={() => removeUpload(upload.filename)}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// Upload Item Component
interface UploadItemProps {
  upload: UploadProgress;
  onRemove: () => void;
}

function UploadItem({ upload, onRemove }: UploadItemProps) {
  const getStatusIcon = () => {
    switch (upload.status) {
      case 'success':
        return <CheckCircle className="h-5 w-5 text-green-600" />;
      case 'error':
        return <AlertCircle className="h-5 w-5 text-red-600" />;
      case 'uploading':
      case 'processing':
        return (
          <div className="h-5 w-5 border-2 border-blue-600 border-t-transparent rounded-full animate-spin" />
        );
      default:
        return <FileText className="h-5 w-5 text-gray-400" />;
    }
  };

  const getStatusText = () => {
    switch (upload.status) {
      case 'pending':
        return 'Bekliyor...';
      case 'uploading':
        return 'Yükleniyor...';
      case 'processing':
        return 'Parse ediliyor...';
      case 'success':
        return 'Tamamlandı';
      case 'error':
        return upload.error || 'Hata';
      default:
        return '';
    }
  };

  const getStatusColor = () => {
    switch (upload.status) {
      case 'success':
        return 'text-green-600';
      case 'error':
        return 'text-red-600';
      case 'uploading':
      case 'processing':
        return 'text-blue-600';
      default:
        return 'text-gray-500';
    }
  };

  return (
    <div className="bg-white border border-gray-200 rounded-lg p-4">
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-start gap-3 flex-1 min-w-0">
          {getStatusIcon()}
          
          <div className="flex-1 min-w-0">
            <p className="text-sm font-medium text-gray-900 truncate">
              {upload.filename}
            </p>
            <p className={`text-xs mt-1 ${getStatusColor()}`}>
              {getStatusText()}
            </p>
          </div>
        </div>

        {(upload.status === 'success' || upload.status === 'error') && (
          <button
            onClick={onRemove}
            className="text-gray-400 hover:text-gray-600"
          >
            <X className="h-4 w-4" />
          </button>
        )}
      </div>

      {/* Progress Bar */}
      {(upload.status === 'uploading' || upload.status === 'processing') && (
        <div className="mt-3">
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div
              className="bg-blue-600 h-2 rounded-full transition-all duration-300"
              style={{ width: `${upload.progress}%` }}
            />
          </div>
        </div>
      )}
    </div>
  );
}
