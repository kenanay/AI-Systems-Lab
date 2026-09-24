'use client';

/**
 * Dataset Compiler Page
 * 
 * Documents'ı training-ready Parquet dataset'e compile eder.
 * 
 * Özellikler:
 * - Compilation form (dataset name, version, source selection)
 * - Tokenizer seçimi
 * - Compilation parameters
 * - Job tracking (progress, status)
 * - Dataset version browser
 */

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { datasetCompilerApi, listTokenizers, type TokenizerRecord } from '@/lib/api';
import type {
  CompilationJobRequest,
  CompilationJob,
  DatasetVersion,
} from '@/lib/api';

export default function DatasetCompilerPage() {
  // Tabs
  const [activeTab, setActiveTab] = useState<'compile' | 'jobs' | 'versions'>('compile');

  // Compilation Form
  const [jobName, setJobName] = useState('');
  const [datasetName, setDatasetName] = useState('');
  const [datasetVersion, setDatasetVersion] = useState('');
  const [autoVersion, setAutoVersion] = useState(true);
  const [selectedFileIds, setSelectedFileIds] = useState<string[]>([]);
  const [selectedDocumentIds, setSelectedDocumentIds] = useState<string[]>([]);
  const [selectedTokenizerId, setSelectedTokenizerId] = useState('');

  // Compilation Parameters
  const [minQuality, setMinQuality] = useState(0.5);
  const [maxQuality, setMaxQuality] = useState(1.0);
  const [minLength, setMinLength] = useState(10);
  const [maxLength, setMaxLength] = useState(100000);
  const [allowPii, setAllowPii] = useState(false);
  const [maskPii, setMaskPii] = useState(true);
  const [requireTrainingAllowed, setRequireTrainingAllowed] = useState(true);
  const [removeDuplicates, setRemoveDuplicates] = useState(true);

  // Data
  const [tokenizers, setTokenizers] = useState<TokenizerRecord[]>([]);
  const [jobs, setJobs] = useState<CompilationJob[]>([]);
  const [versions, setVersions] = useState<DatasetVersion[]>([]);

  // UI State
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');

  // Load tokenizers
  useEffect(() => {
    loadTokenizers();
  }, []);

  // Pre-fill from URL query parameters (e.g. from Synthetic Lab bridge)
  useEffect(() => {
    if (typeof window !== 'undefined') {
      const params = new URLSearchParams(window.location.search);
      const fileIdParam = params.get('file_id');
      const datasetNameParam = params.get('dataset_name');
      if (fileIdParam) {
        setSelectedFileIds((prev) => (prev.includes(fileIdParam) ? prev : [...prev, fileIdParam]));
      }
      if (datasetNameParam) {
        setDatasetName(datasetNameParam);
        setJobName(`compile_${datasetNameParam}`);
      }
    }
  }, []);

  // Auto-refresh jobs when on jobs tab
  useEffect(() => {
    if (activeTab === 'jobs') {
      loadJobs();
      const interval = setInterval(loadJobs, 3000); // Refresh every 3s
      return () => clearInterval(interval);
    }
  }, [activeTab]);

  // Load versions when on versions tab
  useEffect(() => {
    if (activeTab === 'versions') {
      loadVersions();
    }
  }, [activeTab]);

  const loadTokenizers = async () => {
    try {
      const data = await listTokenizers(true); // Only active tokenizers
      setTokenizers(data);
      if (data.length > 0 && !selectedTokenizerId) {
        setSelectedTokenizerId(data[0].tokenizer_id);
      }
    } catch (err) {
      console.error('Tokenizer yükleme hatası:', err);
    }
  };

  const loadJobs = async () => {
    try {
      const data = await datasetCompilerApi.listJobs();
      setJobs(data);
    } catch (err) {
      console.error('Job yükleme hatası:', err);
    }
  };

  const loadVersions = async () => {
    try {
      const data = await datasetCompilerApi.listVersions();
      setVersions(data);
    } catch (err) {
      console.error('Version yükleme hatası:', err);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setSuccess('');

    // Validation
    if (!jobName.trim()) {
      setError('Job ismi gerekli');
      return;
    }

    if (!datasetName.trim()) {
      setError('Dataset ismi gerekli');
      return;
    }

    if (!selectedTokenizerId) {
      setError('Tokenizer seçimi gerekli');
      return;
    }

    if (selectedFileIds.length === 0 && selectedDocumentIds.length === 0) {
      setError('En az bir file veya document seçilmeli');
      return;
    }

    setLoading(true);

    try {
      const request: CompilationJobRequest = {
        job_name: jobName,
        dataset_name: datasetName,
        dataset_version: autoVersion ? undefined : datasetVersion,
        file_ids: selectedFileIds.length > 0 ? selectedFileIds : undefined,
        document_ids: selectedDocumentIds.length > 0 ? selectedDocumentIds : undefined,
        tokenizer_id: selectedTokenizerId,
        compilation_params: {
          min_quality_score: minQuality,
          max_quality_score: maxQuality,
          min_length: minLength,
          max_length: maxLength,
          allow_pii: allowPii,
          mask_pii: maskPii,
          require_training_allowed: requireTrainingAllowed,
          remove_duplicates: removeDuplicates,
        },
      };

      const job = await datasetCompilerApi.createJob(request);
      setSuccess(`Compilation job oluşturuldu: ${job.job_id}`);
      
      // Switch to jobs tab
      setActiveTab('jobs');
      loadJobs();

      // Reset form
      setJobName('');
      setDatasetName('');
      setDatasetVersion('');
      setSelectedFileIds([]);
      setSelectedDocumentIds([]);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Compilation job oluşturulamadı');
    } finally {
      setLoading(false);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'COMPLETED':
        return 'text-green-600 bg-green-50';
      case 'RUNNING':
        return 'text-blue-600 bg-blue-50';
      case 'FAILED':
        return 'text-red-600 bg-red-50';
      case 'CANCELLED':
        return 'text-gray-600 bg-gray-50';
      default:
        return 'text-yellow-600 bg-yellow-50';
    }
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <h1 className="text-3xl font-bold mb-6">Dataset Compiler</h1>

      {/* Tabs */}
      <div className="border-b border-gray-200 mb-6">
        <nav className="-mb-px flex space-x-8">
          <button
            onClick={() => setActiveTab('compile')}
            className={`${
              activeTab === 'compile'
                ? 'border-blue-500 text-blue-600'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            } whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm`}
          >
            Yeni Compilation
          </button>
          <button
            onClick={() => setActiveTab('jobs')}
            className={`${
              activeTab === 'jobs'
                ? 'border-blue-500 text-blue-600'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            } whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm`}
          >
            Compilation Jobs
          </button>
          <button
            onClick={() => setActiveTab('versions')}
            className={`${
              activeTab === 'versions'
                ? 'border-blue-500 text-blue-600'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            } whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm`}
          >
            Dataset Versions
          </button>
        </nav>
      </div>

      {/* Compile Tab */}
      {activeTab === 'compile' && (
        <div className="max-w-4xl">
          <form onSubmit={handleSubmit} className="space-y-6">
            {/* Basic Info */}
            <div className="bg-white shadow rounded-lg p-6">
              <h2 className="text-xl font-semibold mb-4">Temel Bilgiler</h2>

              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Job İsmi *
                  </label>
                  <input
                    type="text"
                    value={jobName}
                    onChange={(e) => setJobName(e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    placeholder="örn: Turkish Education Dataset Compilation"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Dataset İsmi *
                  </label>
                  <input
                    type="text"
                    value={datasetName}
                    onChange={(e) => setDatasetName(e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    placeholder="örn: turkish_education"
                  />
                </div>

                <div>
                  <label className="flex items-center space-x-2">
                    <input
                      type="checkbox"
                      checked={autoVersion}
                      onChange={(e) => setAutoVersion(e.target.checked)}
                      className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                    />
                    <span className="text-sm text-gray-700">
                      Otomatik version (PATCH bump)
                    </span>
                  </label>
                </div>

                {!autoVersion && (
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Dataset Version
                    </label>
                    <input
                      type="text"
                      value={datasetVersion}
                      onChange={(e) => setDatasetVersion(e.target.value)}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                      placeholder="örn: 1.0.0"
                    />
                  </div>
                )}

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Tokenizer *
                  </label>
                  <select
                    value={selectedTokenizerId}
                    onChange={(e) => setSelectedTokenizerId(e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
                    <option value="">Tokenizer seç...</option>
                    {tokenizers.map((t) => (
                      <option key={t.tokenizer_id} value={t.tokenizer_id}>
                        {t.name} (vocab: {t.vocab_size})
                      </option>
                    ))}
                  </select>
                  {tokenizers.length === 0 && (
                    <p className="mt-1 text-sm text-gray-500">
                      Henüz trained tokenizer yok. Önce tokenizer training yapın.
                    </p>
                  )}
                </div>
              </div>
            </div>

            {/* Source Selection */}
            <div className="bg-white shadow rounded-lg p-6">
              <h2 className="text-xl font-semibold mb-4">Kaynak Seçimi</h2>

              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    File IDs (virgülle ayırın)
                  </label>
                  <textarea
                    value={selectedFileIds.join(', ')}
                    onChange={(e) =>
                      setSelectedFileIds(
                        e.target.value
                          .split(',')
                          .map((id) => id.trim())
                          .filter(Boolean)
                      )
                    }
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    rows={3}
                    placeholder="örn: file_abc123, file_def456"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Document IDs (virgülle ayırın)
                  </label>
                  <textarea
                    value={selectedDocumentIds.join(', ')}
                    onChange={(e) =>
                      setSelectedDocumentIds(
                        e.target.value
                          .split(',')
                          .map((id) => id.trim())
                          .filter(Boolean)
                      )
                    }
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    rows={3}
                    placeholder="örn: doc_abc123, doc_def456"
                  />
                </div>

                <p className="text-sm text-gray-500">
                  * En az bir file veya document ID gerekli
                </p>
              </div>
            </div>

            {/* Compilation Parameters */}
            <div className="bg-white shadow rounded-lg p-6">
              <h2 className="text-xl font-semibold mb-4">Compilation Parametreleri</h2>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Min Quality Score
                  </label>
                  <input
                    type="number"
                    value={minQuality}
                    onChange={(e) => setMinQuality(parseFloat(e.target.value))}
                    step="0.1"
                    min="0"
                    max="1"
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Max Quality Score
                  </label>
                  <input
                    type="number"
                    value={maxQuality}
                    onChange={(e) => setMaxQuality(parseFloat(e.target.value))}
                    step="0.1"
                    min="0"
                    max="1"
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Min Length (chars)
                  </label>
                  <input
                    type="number"
                    value={minLength}
                    onChange={(e) => setMinLength(parseInt(e.target.value))}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Max Length (chars)
                  </label>
                  <input
                    type="number"
                    value={maxLength}
                    onChange={(e) => setMaxLength(parseInt(e.target.value))}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                </div>
              </div>

              <div className="mt-4 space-y-2">
                <label className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    checked={maskPii}
                    onChange={(e) => setMaskPii(e.target.checked)}
                    className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                  />
                  <span className="text-sm text-gray-700">
                    🛡️ <strong>PII Maskeleme (Anonimleştirme):</strong> TC Kimlik, Telefon, E-posta ve IBAN verilerini maskele
                  </span>
                </label>

                <label className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    checked={allowPii}
                    onChange={(e) => setAllowPii(e.target.checked)}
                    className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                  />
                  <span className="text-sm text-gray-700">
                    PII içeren dokümanları filtresiz dahil et (⚠️ Güvenlik riski)
                  </span>
                </label>

                <label className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    checked={requireTrainingAllowed}
                    onChange={(e) => setRequireTrainingAllowed(e.target.checked)}
                    className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                  />
                  <span className="text-sm text-gray-700">
                    Sadece training_allowed=true dokümanları
                  </span>
                </label>

                <label className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    checked={removeDuplicates}
                    onChange={(e) => setRemoveDuplicates(e.target.checked)}
                    className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                  />
                  <span className="text-sm text-gray-700">
                    Duplicate dokümanları kaldır (SHA-256)
                  </span>
                </label>
              </div>
            </div>

            {/* Messages */}
            {error && (
              <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded">
                {error}
              </div>
            )}

            {success && (
              <div className="bg-green-50 border border-green-200 text-green-700 px-4 py-3 rounded">
                {success}
              </div>
            )}

            {/* Submit */}
            <button
              type="submit"
              disabled={loading}
              className="w-full bg-blue-600 text-white py-3 px-4 rounded-md hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed font-medium"
            >
              {loading ? 'Compilation Başlatılıyor...' : 'Compilation Başlat'}
            </button>
          </form>
        </div>
      )}

      {/* Jobs Tab */}
      {activeTab === 'jobs' && (
        <div>
          <div className="bg-white shadow rounded-lg overflow-hidden">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Job
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Status
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Progress
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Created
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Result
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {jobs.map((job) => (
                  <tr key={job.job_id}>
                    <td className="px-6 py-4">
                      <div className="text-sm font-medium text-gray-900">
                        {job.job_name}
                      </div>
                      <div className="text-sm text-gray-500 font-mono">
                        {job.job_id}
                      </div>
                    </td>
                    <td className="px-6 py-4">
                      <span
                        className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${getStatusColor(
                          job.status
                        )}`}
                      >
                        {job.status}
                      </span>
                    </td>
                    <td className="px-6 py-4">
                      <div className="flex items-center">
                        <div className="w-32 bg-gray-200 rounded-full h-2 mr-2">
                          <div
                            className="bg-blue-600 h-2 rounded-full"
                            style={{ width: `${job.progress * 100}%` }}
                          ></div>
                        </div>
                        <span className="text-sm text-gray-700">
                          {(job.progress * 100).toFixed(0)}%
                        </span>
                      </div>
                    </td>
                    <td className="px-6 py-4 text-sm text-gray-500">
                      {new Date(job.created_at).toLocaleString('tr-TR')}
                    </td>
                    <td className="px-6 py-4 text-sm">
                      {job.result_metadata?.dataset_id && (
                        <div className="space-y-1">
                          <div className="text-green-600 font-mono text-xs">
                            ✓ {job.result_metadata.dataset_id}
                          </div>
                          {Boolean(job.result_metadata?.stats?.pii_masked_count) && (
                            <div className="text-[11px] text-amber-600 font-medium">
                              🛡️ {job.result_metadata.stats.pii_masked_count} PII maskelendi
                            </div>
                          )}
                          <Link
                            href={`/training?dataset_id=${encodeURIComponent(job.result_metadata.dataset_id)}&dataset_name=${encodeURIComponent(job.job_name)}`}
                            className="inline-flex items-center gap-1 px-2 py-0.5 bg-indigo-50 text-indigo-700 hover:bg-indigo-100 border border-indigo-200 text-[11px] font-semibold rounded transition"
                          >
                            🚀 Eğitime Aktar ➔
                          </Link>
                        </div>
                      )}
                      {job.error && (
                        <div className="text-red-600 text-xs">{job.error}</div>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>

            {jobs.length === 0 && (
              <div className="text-center py-12 text-gray-500">
                Henüz compilation job yok
              </div>
            )}
          </div>
        </div>
      )}

      {/* Dataset Versions Tab */}
      {activeTab === 'versions' && (
        <div className="bg-white shadow rounded-lg overflow-hidden">
          <div className="p-6 border-b border-gray-200">
            <h2 className="text-xl font-semibold">Dataset Versiyonları</h2>
            <p className="text-sm text-gray-500 mt-1">
              Compile edilmiş Parquet dataset versiyonları ve eğitim paketleri
            </p>
          </div>

          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Dataset
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Version
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Documents
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Tokens
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Size
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {versions.map((version) => (
                  <tr key={version.dataset_id}>
                    <td className="px-6 py-4">
                      <div className="text-sm font-medium text-gray-900">
                        {version.name}
                      </div>
                      <div className="text-sm text-gray-500 font-mono">
                        {version.dataset_id}
                      </div>
                    </td>
                    <td className="px-6 py-4 text-sm text-gray-700 font-mono">
                      v{version.version}
                    </td>
                    <td className="px-6 py-4 text-sm text-gray-700">
                      {version.num_documents.toLocaleString()}
                    </td>
                    <td className="px-6 py-4 text-sm text-gray-700">
                      {version.total_tokens?.toLocaleString() || '-'}
                    </td>
                    <td className="px-6 py-4 text-sm text-gray-700">
                      {version.file_size_bytes
                        ? `${(version.file_size_bytes / 1024 / 1024).toFixed(2)} MB`
                        : '-'}
                    </td>
                    <td className="px-6 py-4 text-sm">
                      <div className="flex items-center gap-3 flex-wrap">
                        <Link
                          href={`/training?dataset_id=${encodeURIComponent(version.dataset_id)}&dataset_name=${encodeURIComponent(version.name)}`}
                          className="inline-flex items-center gap-1 px-2.5 py-1 bg-indigo-600 text-white text-xs font-semibold rounded hover:bg-indigo-700 transition shadow-2xs"
                        >
                          🚀 Model Eğit
                        </Link>
                        <a
                          href={datasetCompilerApi.getDownloadUrl(version.dataset_id)}
                          className="text-blue-600 hover:text-blue-800 text-xs font-medium"
                          download
                        >
                          ⬇ İndir
                        </a>
                        <a
                          href={datasetCompilerApi.getMetadataDownloadUrl(
                            version.dataset_id
                          )}
                          className="text-gray-600 hover:text-gray-800 text-xs font-medium"
                          download
                        >
                          📄 Metadata
                        </a>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>

            {versions.length === 0 && (
              <div className="text-center py-12 text-gray-500">
                Henüz compiled dataset version yok
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
