/**
 * Document Viewer Component
 * 
 * Doküman içeriğini göster.
 */

'use client';

import { useDocumentByFile } from '@/hooks/useDatasets';
import { useFile } from '@/hooks/useFiles';
import { FileText, Loader2, AlertCircle } from 'lucide-react';
import { formatNumber } from '@/lib/utils';

interface DocumentViewerProps {
  fileId: string;
}

export function DocumentViewer({ fileId }: DocumentViewerProps) {
  const { data: file, isLoading: fileLoading } = useFile(fileId);
  const { data: document, isLoading: docLoading, error: docError } = useDocumentByFile(
    fileId,
    !!file?.parser_name // Sadece parse edilmiş dosyalar için çağır
  );
  
  if (fileLoading) {
    return (
      <div className="flex items-center justify-center py-12">
        <Loader2 className="h-8 w-8 text-blue-600 animate-spin" />
      </div>
    );
  }

  if (!file) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-6 text-center">
        <AlertCircle className="mx-auto h-8 w-8 text-red-600 mb-2" />
        <p className="text-red-800">Dosya bulunamadı</p>
      </div>
    );
  }

  // Parse edilmemiş dosya
  if (!file.parser_name) {
    return (
      <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-6 text-center">
        <FileText className="mx-auto h-12 w-12 text-yellow-600 mb-4" />
        <p className="text-yellow-800 font-medium mb-2">
          Bu dosya henüz parse edilmedi
        </p>
        <p className="text-sm text-yellow-700">
          Dosyayı parse etmek için Files tablosundan "Parse" butonuna basın.
        </p>
      </div>
    );
  }

  // Document yükleniyor
  if (docLoading) {
    return (
      <div className="space-y-4">
        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <h3 className="text-lg font-semibold text-gray-900 mb-2">
            {file.original_name}
          </h3>
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <span className="text-gray-500">Parser:</span>{' '}
              <span className="text-gray-900 font-medium">
                {file.parser_name} v{file.parser_version}
              </span>
            </div>
            <div>
              <span className="text-gray-500">Dil:</span>{' '}
              <span className="text-gray-900 font-medium">
                {file.language || '-'}
              </span>
            </div>
          </div>
        </div>
        
        <div className="flex items-center justify-center py-8">
          <Loader2 className="h-6 w-6 text-blue-600 animate-spin mr-2" />
          <span className="text-gray-600">Doküman yükleniyor...</span>
        </div>
      </div>
    );
  }

  // Document yükleme hatası
  if (docError || !document) {
    return (
      <div className="space-y-4">
        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <h3 className="text-lg font-semibold text-gray-900 mb-2">
            {file.original_name}
          </h3>
        </div>
        
        <div className="bg-red-50 border border-red-200 rounded-lg p-6 text-center">
          <AlertCircle className="mx-auto h-8 w-8 text-red-600 mb-2" />
          <p className="text-red-800">Doküman yüklenemedi</p>
          <p className="text-sm text-red-600 mt-1">
            {(docError as any)?.response?.data?.detail || 'Bilinmeyen hata'}
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* File Info */}
      <div className="bg-white border border-gray-200 rounded-lg p-4">
        <h3 className="text-lg font-semibold text-gray-900 mb-2">
          {document.title || file.original_name}
        </h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
          <div>
            <span className="text-gray-500">Parser:</span>{' '}
            <span className="text-gray-900 font-medium">
              {file.parser_name} v{file.parser_version}
            </span>
          </div>
          <div>
            <span className="text-gray-500">Dil:</span>{' '}
            <span className="text-gray-900 font-medium">
              {document.language || '-'}
            </span>
          </div>
          <div>
            <span className="text-gray-500">Kelime:</span>{' '}
            <span className="text-gray-900 font-medium">
              {formatNumber(document.word_count || 0)}
            </span>
          </div>
          <div>
            <span className="text-gray-500">Kalite:</span>{' '}
            <span className="text-gray-900 font-medium">
              {document.quality_score ? `${(document.quality_score * 100).toFixed(0)}%` : '-'}
            </span>
          </div>
        </div>
      </div>

      {/* Document Content */}
      <div className="bg-white border border-gray-200 rounded-lg p-6">
        <div className="flex items-center justify-between mb-4">
          <h4 className="text-sm font-semibold text-gray-700 uppercase">
            Doküman İçeriği
          </h4>
          <span className="text-xs text-gray-500">
            {formatNumber(document.char_count || 0)} karakter
          </span>
        </div>
        
        <div className="bg-gray-50 border border-gray-200 rounded p-4 max-h-96 overflow-y-auto">
          <pre className="text-sm text-gray-800 whitespace-pre-wrap font-sans">
            {document.text}
          </pre>
        </div>
        
        {document.is_empty && (
          <p className="text-sm text-yellow-600 mt-2">
            ⚠️ Bu doküman boş olarak işaretlenmiş
          </p>
        )}
        
        {document.is_duplicate && (
          <p className="text-sm text-orange-600 mt-2">
            ⚠️ Bu doküman duplicate olarak işaretlenmiş
          </p>
        )}
      </div>
    </div>
  );
}
