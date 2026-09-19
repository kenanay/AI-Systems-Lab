/**
 * Metadata Panel Component
 * 
 * Dosya ve doküman metadata'sını göster.
 */

'use client';

import { useFile } from '@/hooks/useFiles';
import { Info, Shield, FileCheck, Clock } from 'lucide-react';
import {
  formatFileSize,
  formatDate,
  getQualityColor,
  formatNumber,
} from '@/lib/utils';

interface MetadataPanelProps {
  fileId: string;
}

export function MetadataPanel({ fileId }: MetadataPanelProps) {
  const { data: file, isLoading } = useFile(fileId);

  if (isLoading) {
    return (
      <div className="animate-pulse space-y-4">
        <div className="h-20 bg-gray-200 rounded" />
        <div className="h-20 bg-gray-200 rounded" />
        <div className="h-20 bg-gray-200 rounded" />
      </div>
    );
  }

  if (!file) {
    return null;
  }

  return (
    <div className="space-y-4">
      {/* Genel Bilgiler */}
      <div className="bg-white border border-gray-200 rounded-lg p-4">
        <div className="flex items-center gap-2 mb-3">
          <Info className="h-5 w-5 text-blue-600" />
          <h3 className="font-semibold text-gray-900">Genel Bilgiler</h3>
        </div>
        
        <div className="space-y-2 text-sm">
          <MetadataRow label="Dosya Adı" value={file.original_name} />
          <MetadataRow label="Dosya Boyutu" value={formatFileSize(file.size_bytes)} />
          <MetadataRow label="MIME Type" value={file.mime_type} />
          <MetadataRow
            label="SHA-256"
            value={
              <span className="font-mono text-xs">{file.sha256.substring(0, 16)}...</span>
            }
          />
          <MetadataRow label="Oluşturma Tarihi" value={formatDate(file.created_at)} />
        </div>
      </div>

      {/* Parse Bilgileri */}
      {file.parser_name && (
        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <div className="flex items-center gap-2 mb-3">
            <FileCheck className="h-5 w-5 text-green-600" />
            <h3 className="font-semibold text-gray-900">Parse Bilgileri</h3>
          </div>
          
          <div className="space-y-2 text-sm">
            <MetadataRow label="Parser" value={`${file.parser_name} v${file.parser_version}`} />
            <MetadataRow label="Dil" value={file.language || '-'} />
            <MetadataRow
              label="Kalite Skoru"
              value={
                file.quality_score !== null && file.quality_score !== undefined ? (
                  <span className={`font-medium ${getQualityColor(file.quality_score)}`}>
                    {(file.quality_score * 100).toFixed(1)}%
                  </span>
                ) : (
                  '-'
                )
              }
            />
            <MetadataRow label="Schema Versiyon" value={file.schema_version} />
          </div>
        </div>
      )}

      {/* Güvenlik Bilgileri */}
      <div className="bg-white border border-gray-200 rounded-lg p-4">
        <div className="flex items-center gap-2 mb-3">
          <Shield className="h-5 w-5 text-purple-600" />
          <h3 className="font-semibold text-gray-900">Güvenlik & İzinler</h3>
        </div>
        
        <div className="space-y-2 text-sm">
          <MetadataRow
            label="Security Level"
            value={
              <span className="inline-flex items-center px-2 py-1 rounded text-xs font-medium bg-gray-100 text-gray-800">
                {file.security_level}
              </span>
            }
          />
          <MetadataRow
            label="PII Tespit"
            value={
              <span
                className={`inline-flex items-center px-2 py-1 rounded text-xs font-medium ${
                  file.pii_detected
                    ? 'bg-red-100 text-red-800'
                    : 'bg-green-100 text-green-800'
                }`}
              >
                {file.pii_detected ? '⚠️ Tespit Edildi' : '✓ Yok'}
              </span>
            }
          />
          <MetadataRow
            label="Training İzni"
            value={
              <span
                className={`inline-flex items-center px-2 py-1 rounded text-xs font-medium ${
                  file.training_allowed
                    ? 'bg-green-100 text-green-800'
                    : 'bg-gray-100 text-gray-800'
                }`}
              >
                {file.training_allowed ? '✓ İzinli' : '✗ İzinsiz'}
              </span>
            }
          />
          <MetadataRow label="Lisans" value={file.license || '-'} />
          <MetadataRow label="Telif Hakları" value={file.copyright_status || '-'} />
        </div>
      </div>

      {/* Data Lineage */}
      <div className="bg-white border border-gray-200 rounded-lg p-4">
        <div className="flex items-center gap-2 mb-3">
          <Clock className="h-5 w-5 text-orange-600" />
          <h3 className="font-semibold text-gray-900">Data Lineage</h3>
        </div>
        
        <div className="space-y-2 text-sm">
          <MetadataRow label="File ID" value={<span className="font-mono text-xs">{file.file_id}</span>} />
          <MetadataRow label="Dataset Version" value={file.dataset_version || '-'} />
          <MetadataRow label="Source" value={file.source || '-'} />
          <MetadataRow label="Relative Path" value={<span className="font-mono text-xs">{file.relative_path}</span>} />
        </div>
      </div>
    </div>
  );
}

// Metadata Row Helper
interface MetadataRowProps {
  label: string;
  value: React.ReactNode;
}

function MetadataRow({ label, value }: MetadataRowProps) {
  return (
    <div className="flex justify-between items-start gap-4">
      <span className="text-gray-600 flex-shrink-0">{label}:</span>
      <span className="text-gray-900 font-medium text-right">{value}</span>
    </div>
  );
}
