/**
 * Dataset Explorer Page
 */

'use client';

import { useState } from 'react';
import Link from 'next/link';
import { ArrowLeft, BarChart3, Download, Package } from 'lucide-react';
import { FilesTable } from '@/components/FilesTable';
import { DocumentViewer } from '@/components/DocumentViewer';
import { MetadataPanel } from '@/components/MetadataPanel';
import { useDatasetStats, useExportParquet, useExportPretraining } from '@/hooks/useDatasets';
import { formatFileSize, formatNumber } from '@/lib/utils';

export default function DatasetExplorerPage() {
  const [selectedFileId, setSelectedFileId] = useState<string | null>(null);
  
  const { data: stats, isLoading: statsLoading } = useDatasetStats();
  const exportParquetMutation = useExportParquet();
  const exportPretrainingMutation = useExportPretraining();

  const handleExportParquet = async () => {
    try {
      const result = await exportParquetMutation.mutateAsync();
      alert(`✓ Parquet export tamamlandı!\n${result.message}`);
    } catch (error: any) {
      alert(`✗ Export başarısız: ${error.response?.data?.detail || error.message}`);
    }
  };

  const handleExportPretraining = async () => {
    try {
      const result = await exportPretrainingMutation.mutateAsync(0.5);
      alert(
        `✓ Pretraining dataset oluşturuldu!\n` +
        `Dosya: ${result.output_path}\n` +
        `Boyut: ${formatFileSize(result.file_size_bytes || 0)}\n` +
        `Güvenlik filtreleri uygulandı.`
      );
    } catch (error: any) {
      alert(`✗ Export başarısız: ${error.response?.data?.detail || error.message}`);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white border-b border-gray-200">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Link
                href="/"
                className="text-gray-600 hover:text-gray-900 transition-colors"
              >
                <ArrowLeft className="h-5 w-5" />
              </Link>
              <div>
                <h1 className="text-2xl font-bold text-gray-900">
                  Dataset Explorer
                </h1>
                <p className="text-sm text-gray-500">
                  Canonical dataset'ini keşfet ve yönet
                </p>
              </div>
            </div>

            {/* Export Actions */}
            <div className="flex items-center gap-2">
              <button
                onClick={handleExportParquet}
                disabled={exportParquetMutation.isPending}
                className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed text-sm font-medium"
              >
                <Download className="h-4 w-4" />
                {exportParquetMutation.isPending ? 'Export ediliyor...' : 'Parquet Export'}
              </button>
              
              <button
                onClick={handleExportPretraining}
                disabled={exportPretrainingMutation.isPending}
                className="flex items-center gap-2 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed text-sm font-medium"
              >
                <Package className="h-4 w-4" />
                {exportPretrainingMutation.isPending ? 'Export ediliyor...' : 'Pretraining Export'}
              </button>
            </div>
          </div>
        </div>
      </div>

      <div className="container mx-auto px-4 py-8">
        {/* Stats Cards */}
        {stats && (
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
            <StatCard
              title="Toplam Dosya"
              value={formatNumber(stats.total_files)}
              icon={<BarChart3 className="h-5 w-5" />}
              color="blue"
            />
            <StatCard
              title="Toplam Doküman"
              value={formatNumber(stats.total_documents)}
              icon={<BarChart3 className="h-5 w-5" />}
              color="green"
            />
            <StatCard
              title="Toplam Boyut"
              value={formatFileSize(stats.total_size_bytes)}
              icon={<BarChart3 className="h-5 w-5" />}
              color="purple"
            />
            <StatCard
              title="Ort. Kalite"
              value={
                stats.avg_quality_score
                  ? `${(stats.avg_quality_score * 100).toFixed(0)}%`
                  : '-'
              }
              icon={<BarChart3 className="h-5 w-5" />}
              color="orange"
            />
          </div>
        )}

        {/* Main Content */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Files Table */}
          <div className="lg:col-span-2">
            <FilesTable onViewDocument={setSelectedFileId} />
          </div>

          {/* Sidebar */}
          <div className="lg:col-span-1">
            {selectedFileId ? (
              <div className="space-y-6">
                <DocumentViewer fileId={selectedFileId} />
                <MetadataPanel fileId={selectedFileId} />
              </div>
            ) : (
              <div className="bg-white border border-gray-200 rounded-lg p-8 text-center">
                <BarChart3 className="mx-auto h-12 w-12 text-gray-400 mb-4" />
                <p className="text-gray-600">
                  Detayları görmek için bir dosya seçin
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

// Stat Card Component
interface StatCardProps {
  title: string;
  value: string | number;
  icon: React.ReactNode;
  color: 'blue' | 'green' | 'purple' | 'orange';
}

function StatCard({ title, value, icon, color }: StatCardProps) {
  const colorClasses = {
    blue: 'bg-blue-100 text-blue-600',
    green: 'bg-green-100 text-green-600',
    purple: 'bg-purple-100 text-purple-600',
    orange: 'bg-orange-100 text-orange-600',
  };

  return (
    <div className="bg-white border border-gray-200 rounded-lg p-6">
      <div className="flex items-center justify-between mb-2">
        <p className="text-sm text-gray-600">{title}</p>
        <div className={`p-2 rounded-lg ${colorClasses[color]}`}>{icon}</div>
      </div>
      <p className="text-2xl font-bold text-gray-900">{value}</p>
    </div>
  );
}
