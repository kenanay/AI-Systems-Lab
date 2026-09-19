/**
 * Files Table Component
 * 
 * Dataset'teki dosyaları listele, filtrele, sırala.
 */

'use client';

import { useState } from 'react';
import { Trash2, Eye, FileText, Filter, PlayCircle, Loader2 } from 'lucide-react';
import { useFiles, useDeleteFile, useProcessFile } from '@/hooks/useFiles';
import {
  formatFileSize,
  formatRelativeTime,
  getFileIcon,
  getQualityColor,
} from '@/lib/utils';
import type { FileRecord } from '@/types';

interface FilesTableProps {
  onViewDocument?: (fileId: string) => void;
}

export function FilesTable({ onViewDocument }: FilesTableProps) {
  const [languageFilter, setLanguageFilter] = useState<string>('');
  const [sortBy, setSortBy] = useState<'date' | 'size' | 'quality'>('date');
  
  const { data: files, isLoading, error } = useFiles();
  const deleteMutation = useDeleteFile();
  const processMutation = useProcessFile();

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-center">
        <p className="text-red-800">Dosyalar yüklenirken hata oluştu</p>
      </div>
    );
  }

  if (!files || files.length === 0) {
    return (
      <div className="bg-gray-50 border border-gray-200 rounded-lg p-12 text-center">
        <FileText className="mx-auto h-12 w-12 text-gray-400 mb-4" />
        <p className="text-gray-600 mb-2">Henüz dosya yok</p>
        <p className="text-sm text-gray-500">
          Başlamak için dosya yükle
        </p>
      </div>
    );
  }

  // Filtreleme
  let filteredFiles = [...files];
  
  if (languageFilter) {
    filteredFiles = filteredFiles.filter(
      (f) => f.language === languageFilter
    );
  }

  // Sıralama
  filteredFiles.sort((a, b) => {
    switch (sortBy) {
      case 'size':
        return b.size_bytes - a.size_bytes;
      case 'quality':
        return (b.quality_score || 0) - (a.quality_score || 0);
      case 'date':
      default:
        return new Date(b.created_at).getTime() - new Date(a.created_at).getTime();
    }
  });

  // Benzersiz diller
  const languages = Array.from(new Set(files.map((f) => f.language).filter(Boolean)));

  const handleDelete = async (fileId: string) => {
    if (confirm('Bu dosyayı silmek istediğinizden emin misiniz?')) {
      try {
        await deleteMutation.mutateAsync(fileId);
      } catch (error) {
        console.error('Delete error:', error);
        alert('Dosya silinirken hata oluştu');
      }
    }
  };

  const handleProcess = async (fileId: string) => {
    try {
      await processMutation.mutateAsync(fileId);
      // Başarılı mesajı göster
      alert('Dosya başarıyla parse edildi!');
    } catch (error: any) {
      console.error('Process error:', error);
      const errorMsg = error.response?.data?.detail || error.message || 'Parse işlemi başarısız';
      alert(`Hata: ${errorMsg}`);
    }
  };

  return (
    <div>
      {/* Filters */}
      <div className="mb-4 flex flex-wrap items-center gap-4">
        <div className="flex items-center gap-2">
          <Filter className="h-4 w-4 text-gray-500" />
          <span className="text-sm text-gray-700">Filtreler:</span>
        </div>

        {/* Language Filter */}
        <select
          value={languageFilter}
          onChange={(e) => setLanguageFilter(e.target.value)}
          className="text-sm border border-gray-300 rounded-lg px-3 py-1.5 focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
        >
          <option value="">Tüm Diller</option>
          {languages.map((lang) => (
            <option key={lang} value={lang}>
              {lang}
            </option>
          ))}
        </select>

        {/* Sort By */}
        <select
          value={sortBy}
          onChange={(e) => setSortBy(e.target.value as any)}
          className="text-sm border border-gray-300 rounded-lg px-3 py-1.5 focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
        >
          <option value="date">Tarihe göre</option>
          <option value="size">Boyuta göre</option>
          <option value="quality">Kaliteye göre</option>
        </select>

        <div className="ml-auto text-sm text-gray-600">
          {filteredFiles.length} / {files.length} dosya
        </div>
      </div>

      {/* Table */}
      <div className="bg-white border border-gray-200 rounded-lg overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-gray-50 border-b border-gray-200">
              <tr>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Dosya
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Boyut
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Dil
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Kalite
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Tarih
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Durum
                </th>
                <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                  İşlemler
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200">
              {filteredFiles.map((file) => (
                <FileRow
                  key={file.file_id}
                  file={file}
                  onView={() => onViewDocument?.(file.file_id)}
                  onDelete={() => handleDelete(file.file_id)}
                  onProcess={() => handleProcess(file.file_id)}
                  isProcessing={processMutation.isPending}
                />
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

// File Row Component
interface FileRowProps {
  file: FileRecord;
  onView: () => void;
  onDelete: () => void;
  onProcess: () => void;
  isProcessing: boolean;
}

function FileRow({ file, onView, onDelete, onProcess, isProcessing }: FileRowProps) {
  return (
    <tr className="hover:bg-gray-50 transition-colors">
      {/* Dosya adı */}
      <td className="px-4 py-3">
        <div className="flex items-center gap-3">
          <span className="text-2xl">{getFileIcon(file.mime_type)}</span>
          <div className="min-w-0">
            <p className="text-sm font-medium text-gray-900 truncate">
              {file.original_name}
            </p>
            <p className="text-xs text-gray-500 font-mono truncate">
              {file.sha256.substring(0, 12)}...
            </p>
          </div>
        </div>
      </td>

      {/* Boyut */}
      <td className="px-4 py-3 text-sm text-gray-700">
        {formatFileSize(file.size_bytes)}
      </td>

      {/* Dil */}
      <td className="px-4 py-3">
        {file.language ? (
          <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
            {file.language}
          </span>
        ) : (
          <span className="text-xs text-gray-400">-</span>
        )}
      </td>

      {/* Kalite */}
      <td className="px-4 py-3">
        {file.quality_score !== null && file.quality_score !== undefined ? (
          <span className={`text-sm font-medium ${getQualityColor(file.quality_score)}`}>
            {(file.quality_score * 100).toFixed(0)}%
          </span>
        ) : (
          <span className="text-xs text-gray-400">-</span>
        )}
      </td>

      {/* Tarih */}
      <td className="px-4 py-3 text-sm text-gray-600">
        {formatRelativeTime(file.created_at)}
      </td>

      {/* Durum */}
      <td className="px-4 py-3">
        <div className="flex flex-col gap-1">
          {file.parser_name && (
            <span className="inline-flex items-center text-xs text-green-700">
              ✓ Parse edildi
            </span>
          )}
          {file.pii_detected && (
            <span className="inline-flex items-center text-xs text-red-700">
              ⚠️ PII
            </span>
          )}
          {file.training_allowed && (
            <span className="inline-flex items-center text-xs text-blue-700">
              ✓ Training OK
            </span>
          )}
        </div>
      </td>

      {/* İşlemler */}
      <td className="px-4 py-3 text-right">
        <div className="flex items-center justify-end gap-2">
          {/* Parse butonu - sadece parse edilmemiş dosyalar için */}
          {!file.parser_name && (
            <button
              onClick={onProcess}
              disabled={isProcessing}
              className="p-1.5 text-green-600 hover:bg-green-50 rounded transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              title="Parse Et"
            >
              {isProcessing ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <PlayCircle className="h-4 w-4" />
              )}
            </button>
          )}
          
          <button
            onClick={onView}
            className="p-1.5 text-blue-600 hover:bg-blue-50 rounded transition-colors"
            title="Görüntüle"
          >
            <Eye className="h-4 w-4" />
          </button>
          <button
            onClick={onDelete}
            className="p-1.5 text-red-600 hover:bg-red-50 rounded transition-colors"
            title="Sil"
          >
            <Trash2 className="h-4 w-4" />
          </button>
        </div>
      </td>
    </tr>
  );
}
