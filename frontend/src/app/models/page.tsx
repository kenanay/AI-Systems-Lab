"use client";

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { modelsApi, ModelItem, ModelVerifyResponse } from '@/lib/api';

interface ModelVersion {
  version: string;
  created_at: string;
  model_hash?: string;
  file_size_mb?: number;
  environment?: string;
  metrics?: {
    loss?: number;
    perplexity?: number;
    [key: string]: any;
  };
  training_config?: {
    epochs?: number;
    lr?: number;
    [key: string]: any;
  };
}

interface Model {
  name: string;
  versions: ModelVersion[];
  latest_version?: string;
  description?: string;
  tags?: string[];
  architecture?: string;
  parameters?: number;
  model_hash?: string;
  file_size_mb?: number;
}

interface VerificationState {
  verified: boolean;
  status: 'VALID' | 'CORRUPTED';
  expected_hash: string;
  actual_hash: string;
  file_size_mb?: number;
  checkpoint_path?: string;
  verified_at?: string;
  error_message?: string;
}

export default function ModelsPage() {
  const [models, setModels] = useState<Model[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [deleteConfirm, setDeleteConfirm] = useState<string | null>(null);

  // Verification state tracking: key is `${modelName}:${version}`
  const [verifying, setVerifying] = useState<Record<string, boolean>>({});
  const [verifyResults, setVerifyResults] = useState<Record<string, VerificationState>>({});
  const [copiedHash, setCopiedHash] = useState<string | null>(null);

  useEffect(() => {
    fetchModels();
  }, []);

  const fetchModels = async () => {
    try {
      setLoading(true);
      const data = await modelsApi.list();
      
      const rawList: ModelItem[] = Array.isArray(data) 
        ? data 
        : (Array.isArray((data as any)?.models) ? (data as any).models : []);

      // Group by model_name
      const modelMap = new Map<string, Model>();
      for (const item of rawList) {
        const name = item.model_name || 'unnamed';
        const versionObj: ModelVersion = {
          version: item.version,
          created_at: item.created_at,
          model_hash: item.model_hash,
          file_size_mb: item.file_size_mb,
          environment: item.environment,
          metrics: item.metrics,
          training_config: item.training_config,
        };

        if (!modelMap.has(name)) {
          modelMap.set(name, {
            name: name,
            versions: [versionObj],
            latest_version: item.version,
            description: item.description,
            tags: item.tags,
            architecture: item.architecture,
            parameters: item.parameters,
            model_hash: item.model_hash,
            file_size_mb: item.file_size_mb,
          });
        } else {
          const existing = modelMap.get(name)!;
          existing.versions.push(versionObj);
          if (!existing.model_hash && item.model_hash) {
            existing.model_hash = item.model_hash;
          }
        }
      }

      setModels(Array.from(modelMap.values()));
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Model listesi yüklenirken hata oluştu');
    } finally {
      setLoading(false);
    }
  };

  const handleVerifyModel = async (modelName: string, version?: string) => {
    const key = `${modelName}:${version || 'latest'}`;
    try {
      setVerifying((prev) => ({ ...prev, [key]: true }));
      const res = await modelsApi.verify(modelName, version);
      setVerifyResults((prev) => ({
        ...prev,
        [key]: {
          verified: res.verified,
          status: res.status,
          expected_hash: res.expected_hash,
          actual_hash: res.actual_hash,
          file_size_mb: res.file_size_mb,
          checkpoint_path: res.checkpoint_path,
          verified_at: res.verified_at,
        },
      }));
    } catch (err: any) {
      const msg = err?.response?.data?.detail || err.message || 'Doğrulama başarısız oldu';
      setVerifyResults((prev) => ({
        ...prev,
        [key]: {
          verified: false,
          status: 'CORRUPTED',
          expected_hash: '',
          actual_hash: '',
          error_message: msg,
        },
      }));
    } finally {
      setVerifying((prev) => ({ ...prev, [key]: false }));
    }
  };

  const handleCopyHash = (hash: string) => {
    navigator.clipboard.writeText(hash);
    setCopiedHash(hash);
    setTimeout(() => setCopiedHash(null), 2000);
  };

  const handleDelete = async (modelName: string) => {
    try {
      await modelsApi.delete(modelName);
      await fetchModels();
      setDeleteConfirm(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Model silinirken hata oluştu');
    }
  };

  const filteredModels = models.filter(
    (model) =>
      model.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      (model.description && model.description.toLowerCase().includes(searchQuery.toLowerCase()))
  );

  const getLatestMetrics = (model: Model) => {
    if (model.versions && model.versions.length > 0) {
      return model.versions[0].metrics || {};
    }
    return {};
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white shadow">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-gray-900">Model Hub</h1>
              <p className="mt-1 text-sm text-gray-500">
                Trained model registry - versiyonlama, SHA-256 bütünlük doğrulaması ve üretim yönetimi
              </p>
            </div>
            <div className="flex space-x-3">
              <button
                onClick={fetchModels}
                className="px-4 py-2 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 transition-colors"
              >
                🔄 Yenile
              </button>
              <Link
                href="/training"
                className="px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 transition-colors"
              >
                + Yeni Model Eğit
              </Link>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Search */}
        <div className="mb-6">
          <input
            type="text"
            placeholder="Model adı veya açıklama ile ara..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          />
        </div>

        {/* Stats */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
          <div className="bg-white rounded-lg shadow p-6 border-l-4 border-blue-500">
            <div className="text-sm font-medium text-gray-500">Toplam Model</div>
            <div className="mt-2 text-3xl font-bold text-gray-900">{models.length}</div>
          </div>
          <div className="bg-white rounded-lg shadow p-6 border-l-4 border-indigo-500">
            <div className="text-sm font-medium text-gray-500">Toplam Versiyon</div>
            <div className="mt-2 text-3xl font-bold text-gray-900">
              {models.reduce((sum, m) => sum + (m.versions?.length || 0), 0)}
            </div>
          </div>
          <div className="bg-white rounded-lg shadow p-6 border-l-4 border-emerald-500">
            <div className="text-sm font-medium text-gray-500">Arama Sonucu</div>
            <div className="mt-2 text-3xl font-bold text-gray-900">{filteredModels.length}</div>
          </div>
        </div>

        {/* Loading State */}
        {loading && (
          <div className="text-center py-12">
            <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
            <p className="mt-4 text-gray-600">Modeller yükleniyor...</p>
          </div>
        )}

        {/* Error State */}
        {error && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6">
            <div className="flex">
              <div className="flex-shrink-0">
                <span className="text-red-400">⚠️</span>
              </div>
              <div className="ml-3">
                <h3 className="text-sm font-medium text-red-800">Hata</h3>
                <div className="mt-2 text-sm text-red-700">{error}</div>
              </div>
            </div>
          </div>
        )}

        {/* Empty State */}
        {!loading && models.length === 0 && (
          <div className="text-center py-12 bg-white rounded-lg shadow">
            <div className="text-6xl mb-4">📦</div>
            <h3 className="text-lg font-medium text-gray-900">Henüz model bulunmuyor</h3>
            <p className="mt-2 text-sm text-gray-500">
              İlk modelinizi eğitmek ve registry'ye kaydetmek için Training Lab'e gidin.
            </p>
            <div className="mt-6">
              <Link
                href="/training"
                className="inline-flex items-center px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700"
              >
                Model Eğit
              </Link>
            </div>
          </div>
        )}

        {/* Models List */}
        {!loading && filteredModels.length > 0 && (
          <div className="space-y-4">
            {filteredModels.map((model) => {
              const latestMetrics = getLatestMetrics(model);
              const isExpanded = selectedModel === model.name;
              const verifyKey = `${model.name}:latest`;
              const verification = verifyResults[verifyKey];
              const isVerifying = verifying[verifyKey];

              return (
                <div key={model.name} className="bg-white rounded-lg shadow hover:shadow-md transition-shadow border border-gray-100">
                  <div className="p-6">
                    <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
                      <div className="flex-1">
                        <div className="flex items-center space-x-3 flex-wrap gap-y-1">
                          <h3 className="text-lg font-semibold text-gray-900">{model.name}</h3>
                          {model.latest_version && (
                            <span className="px-2 py-0.5 text-xs font-medium bg-blue-100 text-blue-800 rounded">
                              v{model.latest_version}
                            </span>
                          )}
                          {model.file_size_mb !== undefined && (
                            <span className="px-2 py-0.5 text-xs font-medium bg-gray-100 text-gray-600 rounded">
                              {model.file_size_mb} MB
                            </span>
                          )}
                          {model.architecture && (
                            <span className="px-2 py-0.5 text-xs font-mono bg-purple-50 text-purple-700 border border-purple-200 rounded">
                              {model.architecture}
                            </span>
                          )}
                        </div>

                        {model.description && (
                          <p className="mt-1 text-sm text-gray-500">{model.description}</p>
                        )}

                        {/* SHA-256 Fingerprint Badge */}
                        <div className="mt-2 flex items-center space-x-2 text-xs text-gray-600 font-mono">
                          <span className="font-semibold text-gray-700">SHA-256:</span>
                          {model.model_hash ? (
                            <div className="inline-flex items-center gap-1.5 bg-slate-50 border border-slate-200 px-2 py-0.5 rounded text-slate-800">
                              <span title={model.model_hash}>
                                {model.model_hash.slice(0, 10)}...{model.model_hash.slice(-8)}
                              </span>
                              <button
                                onClick={() => handleCopyHash(model.model_hash!)}
                                className="text-gray-400 hover:text-gray-700 transition-colors"
                                title="Tam hash'i kopyala"
                              >
                                {copiedHash === model.model_hash ? '✓' : '📋'}
                              </button>
                            </div>
                          ) : (
                            <span className="text-gray-400 italic">Doğrulama hash'i henüz yok</span>
                          )}
                        </div>

                        {/* Verification Status Alert */}
                        {verification && (
                          <div className="mt-2">
                            {verification.verified ? (
                              <span className="inline-flex items-center gap-1.5 px-2.5 py-1 text-xs font-medium rounded-md bg-emerald-50 text-emerald-700 border border-emerald-200">
                                <span className="font-bold">✓</span>
                                SHA-256 Bütünlüğü Doğrulandı ({verification.file_size_mb} MB)
                              </span>
                            ) : (
                              <span className="inline-flex items-center gap-1.5 px-2.5 py-1 text-xs font-medium rounded-md bg-red-50 text-red-700 border border-red-200">
                                <span className="font-bold">✕</span>
                                Bütünlük Doğrulanamadı: {verification.error_message || 'Bozulmuş Checkpoint (Hash Uyuşmuyor)'}
                              </span>
                            )}
                          </div>
                        )}

                        {/* Tags */}
                        {model.tags && model.tags.length > 0 && (
                          <div className="mt-2 flex flex-wrap gap-2">
                            {model.tags.map((tag, idx) => (
                              <span
                                key={idx}
                                className="px-2 py-0.5 text-xs bg-gray-100 text-gray-700 rounded"
                              >
                                {tag}
                              </span>
                            ))}
                          </div>
                        )}
                      </div>

                      {/* Actions */}
                      <div className="flex items-center flex-wrap gap-2">
                        {latestMetrics.perplexity && (
                          <div className="text-right mr-3 hidden sm:block">
                            <div className="text-xs text-gray-500">Perplexity</div>
                            <div className="text-lg font-semibold text-gray-900">
                              {latestMetrics.perplexity.toFixed(2)}
                            </div>
                          </div>
                        )}

                        {/* SHA-256 Verify Button */}
                        <button
                          onClick={() => handleVerifyModel(model.name)}
                          disabled={isVerifying}
                          className="px-3 py-2 text-sm border border-emerald-300 text-emerald-800 bg-emerald-50 hover:bg-emerald-100 rounded-md font-medium inline-flex items-center gap-1.5 disabled:opacity-50 transition-colors shadow-sm"
                          title="Model dosyasının SHA-256 hash bütünlüğünü doğrula"
                        >
                          {isVerifying ? (
                            <>
                              <span className="animate-spin h-3.5 w-3.5 border-2 border-emerald-600 border-t-transparent rounded-full" />
                              Doğrulanıyor...
                            </>
                          ) : (
                            <>
                              🛡️ Doğrula (SHA-256)
                            </>
                          )}
                        </button>

                        <button
                          onClick={() => setSelectedModel(isExpanded ? null : model.name)}
                          className="px-3 py-2 text-sm border border-gray-300 rounded-md hover:bg-gray-50 transition-colors"
                        >
                          {isExpanded ? '▲ Gizle' : '▼ Versiyonlar'}
                        </button>

                        <Link
                          href={`/playground?model=${encodeURIComponent(model.name)}`}
                          className="px-3 py-2 text-sm bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors shadow-sm"
                        >
                          🎮 Test Et
                        </Link>

                        <Link
                          href={`/evaluation?model=${encodeURIComponent(model.name)}`}
                          className="px-3 py-2 text-sm bg-indigo-600 text-white rounded-md hover:bg-indigo-700 transition-colors shadow-sm"
                        >
                          📊 Değerlendir
                        </Link>

                        {deleteConfirm === model.name ? (
                          <div className="flex space-x-2">
                            <button
                              onClick={() => handleDelete(model.name)}
                              className="px-3 py-2 text-sm bg-red-600 text-white rounded-md hover:bg-red-700 transition-colors"
                            >
                              ✓ Onayla
                            </button>
                            <button
                              onClick={() => setDeleteConfirm(null)}
                              className="px-3 py-2 text-sm border border-gray-300 rounded-md hover:bg-gray-50 transition-colors"
                            >
                              ✕ İptal
                            </button>
                          </div>
                        ) : (
                          <button
                            onClick={() => setDeleteConfirm(model.name)}
                            className="px-3 py-2 text-sm text-red-600 border border-red-300 rounded-md hover:bg-red-50 transition-colors"
                          >
                            🗑️ Sil
                          </button>
                        )}
                      </div>
                    </div>

                    {/* Expanded Details */}
                    {isExpanded && (
                      <div className="mt-6 pt-6 border-t border-gray-200">
                        <h4 className="text-sm font-semibold text-gray-900 mb-4">Kayıtlı Versiyonlar & Hash Doğrulamaları</h4>
                        <div className="space-y-3">
                          {model.versions && model.versions.map((version, idx) => {
                            const verKey = `${model.name}:${version.version}`;
                            const verState = verifyResults[verKey];
                            const verLoading = verifying[verKey];

                            return (
                              <div
                                key={idx}
                                className="p-4 bg-gray-50 border border-gray-200 rounded-lg"
                              >
                                <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 mb-2">
                                  <div className="flex items-center gap-2">
                                    <span className="font-mono text-sm font-semibold text-gray-900">v{version.version}</span>
                                    {version.file_size_mb && (
                                      <span className="text-xs text-gray-500">({version.file_size_mb} MB)</span>
                                    )}
                                    {version.environment && (
                                      <span className="text-xs px-2 py-0.5 bg-blue-50 text-blue-700 rounded border border-blue-200">
                                        {version.environment}
                                      </span>
                                    )}
                                  </div>

                                  <div className="flex items-center gap-3">
                                    <span className="text-xs text-gray-500">
                                      {new Date(version.created_at).toLocaleDateString('tr-TR')}
                                    </span>
                                    <button
                                      onClick={() => handleVerifyModel(model.name, version.version)}
                                      disabled={verLoading}
                                      className="text-xs px-2.5 py-1 border border-emerald-300 text-emerald-700 bg-white hover:bg-emerald-50 rounded font-medium disabled:opacity-50 transition-colors shadow-2xs"
                                    >
                                      {verLoading ? 'Doğrulanıyor...' : '🛡️ Doğrula'}
                                    </button>
                                  </div>
                                </div>

                                {version.model_hash && (
                                  <div className="text-xs text-gray-600 font-mono mb-2 flex items-center gap-1.5 flex-wrap">
                                    <span className="font-semibold text-gray-700">SHA-256:</span>
                                    <span className="text-gray-800 bg-white px-2 py-0.5 rounded border border-gray-200 select-all">
                                      {version.model_hash}
                                    </span>
                                  </div>
                                )}

                                {verState && (
                                  <div className="mb-3">
                                    {verState.verified ? (
                                      <span className="inline-flex items-center gap-1 text-xs text-emerald-700 bg-emerald-100 px-2 py-0.5 rounded font-medium">
                                        ✓ Doğrulandı: {verState.actual_hash.slice(0, 16)}... ({verState.file_size_mb} MB)
                                      </span>
                                    ) : (
                                      <span className="inline-flex items-center gap-1 text-xs text-red-700 bg-red-100 px-2 py-0.5 rounded font-medium">
                                        ✕ Doğrulanamadı: {verState.error_message || 'Bozuk / Uyuşmuyor'}
                                      </span>
                                    )}
                                  </div>
                                )}

                                {version.metrics && (
                                  <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm mt-3 pt-2 border-t border-gray-200">
                                    {Object.entries(version.metrics).map(([key, value]) => (
                                      <div key={key}>
                                        <div className="text-xs text-gray-500">{key}</div>
                                        <div className="font-medium text-gray-900">
                                          {typeof value === 'number' ? value.toFixed(4) : String(value)}
                                        </div>
                                      </div>
                                    ))}
                                  </div>
                                )}

                                {version.training_config && (
                                  <div className="mt-2 text-xs text-gray-600">
                                    <span className="font-medium text-gray-700">Config:</span>{' '}
                                    {Object.entries(version.training_config)
                                      .slice(0, 4)
                                      .map(([k, v]) => `${k}=${v}`)
                                      .join(', ')}
                                  </div>
                                )}
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {/* No Results */}
        {!loading && models.length > 0 && filteredModels.length === 0 && (
          <div className="text-center py-12 bg-white rounded-lg shadow">
            <div className="text-4xl mb-4">🔍</div>
            <h3 className="text-lg font-medium text-gray-900">Sonuç bulunamadı</h3>
            <p className="mt-2 text-sm text-gray-500">
              "{searchQuery}" aramasıyla eşleşen model bulunamadı.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
