/**
 * File Upload Page
 */

'use client';

import { useState } from 'react';
import Link from 'next/link';
import { ArrowLeft, Info } from 'lucide-react';
import { FileUpload } from '@/components/FileUpload';
import { useFiles } from '@/hooks/useFiles';
import { formatFileSize, formatRelativeTime, getFileIcon } from '@/lib/utils';

export default function UploadPage() {
  const [uploadedCount, setUploadedCount] = useState(0);
  const { data: files, isLoading } = useFiles();

  const handleUploadComplete = (fileId: string) => {
    setUploadedCount((prev) => prev + 1);
    console.log('Upload complete:', fileId);
  };

  return (
    <div className="min-h-[calc(100vh-4rem)] bg-slate-50 text-slate-900 pb-12">
      {/* Header */}
      <div className="bg-white border-b border-slate-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center gap-4">
            <Link
              href="/"
              className="text-slate-600 hover:text-slate-900 transition-colors"
            >
              <ArrowLeft className="h-5 w-5" />
            </Link>
            <div>
              <h1 className="text-2xl font-bold text-slate-900">
                Dosya Yükle
              </h1>
              <p className="text-sm text-slate-500">
                TXT, MD, PDF formatlarını destekler
              </p>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Upload Section */}
          <div className="lg:col-span-2">
            <FileUpload
              onUploadComplete={handleUploadComplete}
              autoProcess={true}
            />

            {uploadedCount > 0 && (
              <div className="mt-6 bg-green-50 border border-green-200 rounded-lg p-4">
                <p className="text-sm text-green-800">
                  ✓ {uploadedCount} dosya başarıyla yüklendi ve parse edildi!
                </p>
              </div>
            )}
          </div>

          {/* Info Sidebar */}
          <div className="lg:col-span-1">
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
              <div className="flex items-center gap-2 mb-4">
                <Info className="h-5 w-5 text-blue-600" />
                <h2 className="text-lg font-semibold text-gray-900">
                  Nasıl Çalışır?
                </h2>
              </div>

              <div className="space-y-4 text-sm text-gray-700">
                <div>
                  <p className="font-medium mb-1">1. Dosya Yükle</p>
                  <p className="text-gray-600">
                    Desteklenen formatlardan (TXT, MD, PDF) dosya seç veya sürükle-bırak.
                  </p>
                </div>

                <div>
                  <p className="font-medium mb-1">2. MIME Type Detection</p>
                  <p className="text-gray-600">
                    Dosya tipi otomatik algılanır ve doğrulanır.
                  </p>
                </div>

                <div>
                  <p className="font-medium mb-1">3. SHA-256 Hash</p>
                  <p className="text-gray-600">
                    Dosya içeriğinden benzersiz hash üretilir. Duplicate dosyalar tespit edilir.
                  </p>
                </div>

                <div>
                  <p className="font-medium mb-1">4. Parse & Normalize</p>
                  <p className="text-gray-600">
                    Metin çıkarılır, temizlenir ve normalize edilir.
                  </p>
                </div>

                <div>
                  <p className="font-medium mb-1">5. Quality Scoring</p>
                  <p className="text-gray-600">
                    İçerik kalitesi otomatik olarak değerlendirilir.
                  </p>
                </div>

                <div>
                  <p className="font-medium mb-1">6. DocumentRecord</p>
                  <p className="text-gray-600">
                    Canonical dataset formatında kaydedilir.
                  </p>
                </div>
              </div>

              <div className="mt-6 pt-6 border-t border-gray-200">
                <Link
                  href="/dataset-explorer"
                  className="block w-full text-center bg-blue-600 text-white rounded-lg px-4 py-2 hover:bg-blue-700 transition-colors text-sm font-medium"
                >
                  Dataset Explorer'a Git →
                </Link>
              </div>
            </div>

            {/* Recent Files */}
            {files && files.length > 0 && (
              <div className="mt-6 bg-white rounded-lg shadow-sm border border-gray-200 p-6">
                <h2 className="text-lg font-semibold text-gray-900 mb-4">
                  Son Yüklenenler
                </h2>
                <div className="space-y-3">
                  {files.slice(0, 5).map((file) => (
                    <div
                      key={file.file_id}
                      className="flex items-start gap-3 text-sm"
                    >
                      <span className="text-xl">{getFileIcon(file.mime_type)}</span>
                      <div className="flex-1 min-w-0">
                        <p className="font-medium text-gray-900 truncate">
                          {file.original_name}
                        </p>
                        <p className="text-xs text-gray-500">
                          {formatFileSize(file.size_bytes)} • {formatRelativeTime(file.created_at)}
                        </p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
