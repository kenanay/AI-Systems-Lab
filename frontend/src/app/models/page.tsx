"use client";

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { modelsApi, ModelItem, ModelVerifyResponse, ModelExportRecord } from '@/lib/api';

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

  // Export & Quantization Modal State
  const [exportModalOpen, setExportModalOpen] = useState(false);
  const [exportTargetModel, setExportTargetModel] = useState<Model | null>(null);
  const [exportTargetVersion, setExportTargetVersion] = useState<string>('1.0.0');
  const [exportFormat, setExportFormat] = useState<'onnx' | 'torchscript' | 'gguf'>('onnx');
  const [exportQuantization, setExportQuantization] = useState<'none' | 'fp16' | 'int8' | 'int4'>('none');
  const [exportLoading, setExportLoading] = useState(false);
  const [exportError, setExportError] = useState<string | null>(null);
  const [exportResult, setExportResult] = useState<ModelExportRecord | null>(null);
  const [modelExportsList, setModelExportsList] = useState<ModelExportRecord[]>([]);
  const [copiedSnippet, setCopiedSnippet] = useState(false);

  useEffect(() => {
    fetchModels();
  }, []);

  const openExportModal = async (model: Model, version?: string) => {
    const ver = version || model.latest_version || (model.versions?.[0]?.version) || '1.0.0';
    setExportTargetModel(model);
    setExportTargetVersion(ver);
    setExportFormat('onnx');
    setExportQuantization('none');
    setExportError(null);
    setExportResult(null);
    setExportModalOpen(true);

    try {
      const list = await modelsApi.listExports(model.name, ver);
      setModelExportsList(list);
    } catch {
      setModelExportsList([]);
    }
  };

  const handleExportSubmit = async () => {
    if (!exportTargetModel) return;
    try {
      setExportLoading(true);
      setExportError(null);
      const res = await modelsApi.exportModel(exportTargetModel.name, {
        version: exportTargetVersion,
        export_format: exportFormat,
        quantization: exportQuantization,
      });
      setExportResult(res);
      // Refresh exports list
      const list = await modelsApi.listExports(exportTargetModel.name, exportTargetVersion);
      setModelExportsList(list);
    } catch (err: any) {
      const msg = err?.response?.data?.detail || err.message || 'Dışa aktarma işlemi başarısız oldu';
      setExportError(msg);
    } finally {
      setExportLoading(false);
    }
  };

  const handleCopySnippet = (snippet: string) => {
    navigator.clipboard.writeText(snippet);
    setCopiedSnippet(true);
    setTimeout(() => setCopiedSnippet(false), 2000);
  };

  const getEstimatedSize = (baseSizeMb: number, quant: string) => {
    const factorMap: Record<string, number> = {
      none: 1.0,
      fp16: 0.5,
      int8: 0.265,
      int4: 0.138,
    };
    const factor = factorMap[quant] || 1.0;
    return (baseSizeMb * factor).toFixed(2);
  };

  const getSavedPercent = (quant: string) => {
    const pctMap: Record<string, string> = {
      none: '%0 (Kayıpsız FP32)',
      fp16: '%50 RAM Tasarrufu',
      int8: '%73.5 RAM Tasarrufu',
      int4: '%86.2 RAM Tasarrufu',
    };
    return pctMap[quant] || '%0';
  };

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

                        {/* Export & Quantize Button */}
                        <button
                          onClick={() => openExportModal(model, model.latest_version)}
                          className="px-3 py-2 text-sm border border-purple-300 text-purple-800 bg-purple-50 hover:bg-purple-100 rounded-md font-medium inline-flex items-center gap-1.5 transition-colors shadow-sm"
                          title="Modeli ONNX, GGUF veya TorchScript formatında kuantize edip dışa aktar"
                        >
                          📦 Dışa Aktar & Kuantize Et
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
                                    <button
                                      onClick={() => openExportModal(model, version.version)}
                                      className="text-xs px-2.5 py-1 border border-purple-300 text-purple-700 bg-white hover:bg-purple-50 rounded font-medium transition-colors shadow-2xs"
                                      title="Bu versiyonu dışa aktar"
                                    >
                                      📦 Dışa Aktar
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

      {/* ========================================================================= */}
      {/* Model Export & Quantization Modal                                         */}
      {/* ========================================================================= */}
      {exportModalOpen && exportTargetModel && (
        <div className="fixed inset-0 z-50 overflow-y-auto bg-black/60 backdrop-blur-xs flex items-center justify-center p-4">
          <div className="bg-white rounded-2xl shadow-2xl max-w-2xl w-full border border-gray-200 overflow-hidden my-8">
            {/* Header */}
            <div className="bg-gradient-to-r from-purple-700 via-indigo-700 to-blue-700 px-6 py-5 text-white flex items-center justify-between">
              <div>
                <div className="flex items-center gap-2">
                  <span className="text-xl">📦</span>
                  <h3 className="text-lg font-bold">Model Dışa Aktarma & Kuantizasyon</h3>
                  <span className="px-2 py-0.5 text-xs font-mono bg-white/20 rounded-md">
                    v{exportTargetVersion}
                  </span>
                </div>
                <p className="text-xs text-purple-100 mt-1">
                  {exportTargetModel.name} modelini üretim ortamları, Ollama veya C++ için dönüştürün.
                </p>
              </div>
              <button
                onClick={() => setExportModalOpen(false)}
                className="text-white/80 hover:text-white bg-white/10 hover:bg-white/20 rounded-full p-1.5 transition-colors"
                title="Kapat"
              >
                ✕
              </button>
            </div>

            <div className="p-6 space-y-6 max-h-[75vh] overflow-y-auto">
              {/* Error Banner */}
              {exportError && (
                <div className="bg-red-50 border border-red-200 text-red-800 text-sm p-3.5 rounded-lg flex items-center gap-2">
                  <span>⚠️</span>
                  <span>{exportError}</span>
                </div>
              )}

              {/* 1. Format Selection */}
              <div>
                <label className="block text-xs font-bold uppercase tracking-wider text-gray-700 mb-2">
                  1. Dışa Aktarma Formatı
                </label>
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                  <button
                    type="button"
                    onClick={() => setExportFormat('onnx')}
                    className={`p-3.5 rounded-xl border text-left transition-all ${
                      exportFormat === 'onnx'
                        ? 'border-purple-600 bg-purple-50/60 ring-2 ring-purple-500/20'
                        : 'border-gray-200 hover:border-gray-300 bg-white'
                    }`}
                  >
                    <div className="flex items-center justify-between mb-1">
                      <span className="font-bold text-sm text-gray-900">ONNX</span>
                      <span className="text-xs font-mono bg-purple-100 text-purple-800 px-1.5 py-0.5 rounded">.onnx</span>
                    </div>
                    <p className="text-xs text-gray-500">
                      ONNX Runtime, Triton & TensorRT için dinamik batch/seq grafı.
                    </p>
                  </button>

                  <button
                    type="button"
                    onClick={() => setExportFormat('gguf')}
                    className={`p-3.5 rounded-xl border text-left transition-all ${
                      exportFormat === 'gguf'
                        ? 'border-purple-600 bg-purple-50/60 ring-2 ring-purple-500/20'
                        : 'border-gray-200 hover:border-gray-300 bg-white'
                    }`}
                  >
                    <div className="flex items-center justify-between mb-1">
                      <span className="font-bold text-sm text-gray-900">GGUF v3</span>
                      <span className="text-xs font-mono bg-indigo-100 text-indigo-800 px-1.5 py-0.5 rounded">.gguf</span>
                    </div>
                    <p className="text-xs text-gray-500">
                      Ollama & llama.cpp ile yerel PC / Mac üzerinde doğrudan çalıştırma.
                    </p>
                  </button>

                  <button
                    type="button"
                    onClick={() => setExportFormat('torchscript')}
                    className={`p-3.5 rounded-xl border text-left transition-all ${
                      exportFormat === 'torchscript'
                        ? 'border-purple-600 bg-purple-50/60 ring-2 ring-purple-500/20'
                        : 'border-gray-200 hover:border-gray-300 bg-white'
                    }`}
                  >
                    <div className="flex items-center justify-between mb-1">
                      <span className="font-bold text-sm text-gray-900">TorchScript</span>
                      <span className="text-xs font-mono bg-blue-100 text-blue-800 px-1.5 py-0.5 rounded">.pt</span>
                    </div>
                    <p className="text-xs text-gray-500">
                      PyTorch JIT ile C++ LibTorch & Python-sız üretim çıkarımı.
                    </p>
                  </button>
                </div>
              </div>

              {/* 2. Quantization Selection */}
              <div>
                <label className="block text-xs font-bold uppercase tracking-wider text-gray-700 mb-2">
                  2. Kuantizasyon & Hassasiyet
                </label>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                  <button
                    type="button"
                    onClick={() => setExportQuantization('none')}
                    className={`p-3 rounded-lg border text-left transition-all ${
                      exportQuantization === 'none'
                        ? 'border-purple-600 bg-purple-50/50 ring-2 ring-purple-500/20'
                        : 'border-gray-200 hover:border-gray-300'
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-semibold text-gray-900">FP32 (Standart)</span>
                      <span className="text-xs font-medium text-gray-500">Kayıpsız</span>
                    </div>
                    <p className="text-xs text-gray-500 mt-1">
                      Orijinal 32-bit kayan nokta ağırlıkları. Maksimum doğruluk.
                    </p>
                  </button>

                  <button
                    type="button"
                    onClick={() => setExportQuantization('fp16')}
                    className={`p-3 rounded-lg border text-left transition-all ${
                      exportQuantization === 'fp16'
                        ? 'border-purple-600 bg-purple-50/50 ring-2 ring-purple-500/20'
                        : 'border-gray-200 hover:border-gray-300'
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-semibold text-gray-900">FP16 (Yarı Hassasiyet)</span>
                      <span className="text-xs font-semibold text-emerald-600">%50 Tasarruf</span>
                    </div>
                    <p className="text-xs text-gray-500 mt-1">
                      16-bit float. GPU çıkarımında 2x hızlanma ve %50 bellek düşüşü.
                    </p>
                  </button>

                  <button
                    type="button"
                    onClick={() => setExportQuantization('int8')}
                    className={`p-3 rounded-lg border text-left transition-all ${
                      exportQuantization === 'int8'
                        ? 'border-purple-600 bg-purple-50/50 ring-2 ring-purple-500/20'
                        : 'border-gray-200 hover:border-gray-300'
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-semibold text-gray-900">INT8 (Dinamik)</span>
                      <span className="text-xs font-semibold text-emerald-600">%73.5 Tasarruf</span>
                    </div>
                    <p className="text-xs text-gray-500 mt-1">
                      Linear katman ağırlıkları 8-bit. CPU için ideal 2x-3x hızlanma.
                    </p>
                  </button>

                  <button
                    type="button"
                    onClick={() => setExportQuantization('int4')}
                    className={`p-3 rounded-lg border text-left transition-all ${
                      exportQuantization === 'int4'
                        ? 'border-purple-600 bg-purple-50/50 ring-2 ring-purple-500/20'
                        : 'border-gray-200 hover:border-gray-300'
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-semibold text-gray-900">INT4 (Blok Ağırlık)</span>
                      <span className="text-xs font-semibold text-purple-600 font-bold">%86.2 Tasarruf</span>
                    </div>
                    <p className="text-xs text-gray-500 mt-1">
                      AWQ / GPTQ tarzı 4-bit blok ölçekleme. Ultra düşük VRAM tüketimi.
                    </p>
                  </button>
                </div>
              </div>

              {/* 3. Live RAM & Size Estimation Card */}
              <div className="p-4 bg-gradient-to-r from-slate-50 to-purple-50/30 rounded-xl border border-slate-200/80">
                <div className="text-xs font-bold uppercase tracking-wider text-slate-600 mb-2">
                  Tahmini Bellek & Dosya Boyutu Etkisi
                </div>
                <div className="grid grid-cols-3 gap-4 text-center">
                  <div className="bg-white p-2.5 rounded-lg border border-slate-200">
                    <div className="text-xs text-slate-500">Mevcut Boyut</div>
                    <div className="text-sm font-bold text-slate-800">
                      {exportTargetModel.file_size_mb || 52.4} MB
                    </div>
                  </div>
                  <div className="bg-white p-2.5 rounded-lg border border-purple-200">
                    <div className="text-xs text-purple-700">Tahmini Yeni Boyut</div>
                    <div className="text-sm font-bold text-purple-900">
                      ~{getEstimatedSize(exportTargetModel.file_size_mb || 52.4, exportQuantization)} MB
                    </div>
                  </div>
                  <div className="bg-white p-2.5 rounded-lg border border-emerald-200">
                    <div className="text-xs text-emerald-700">Bellek Tasarrufu</div>
                    <div className="text-sm font-bold text-emerald-700">
                      {getSavedPercent(exportQuantization)}
                    </div>
                  </div>
                </div>
              </div>

              {/* Export Trigger Button */}
              <div>
                <button
                  type="button"
                  onClick={handleExportSubmit}
                  disabled={exportLoading}
                  className="w-full py-3 px-4 bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-700 hover:to-indigo-700 text-white font-medium rounded-xl shadow-md transition-all flex items-center justify-center gap-2 disabled:opacity-50"
                >
                  {exportLoading ? (
                    <>
                      <span className="animate-spin h-4 w-4 border-2 border-white border-t-transparent rounded-full" />
                      Dışa Aktarılıyor & Kuantize Ediliyor...
                    </>
                  ) : (
                    <>
                      🚀 Dışa Aktarmayı Başlat ({exportFormat.toUpperCase()} - {exportQuantization.toUpperCase()})
                    </>
                  )}
                </button>
              </div>

              {/* Export Result & Download */}
              {exportResult && (
                <div className="p-4 bg-emerald-50 border border-emerald-200 rounded-xl space-y-3 animate-in fade-in duration-200">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <span className="text-emerald-700 font-bold">✓ Dışa Aktarma Tamamlandı</span>
                      <span className="text-xs font-mono bg-emerald-200/60 text-emerald-900 px-2 py-0.5 rounded">
                        {exportResult.file_size_mb} MB ({exportResult.compression_ratio}x küçültüldü)
                      </span>
                    </div>
                    <a
                      href={modelsApi.getDownloadUrl(exportTargetModel.name, exportResult.file_name, exportTargetVersion)}
                      download={exportResult.file_name}
                      className="px-3 py-1.5 bg-emerald-600 hover:bg-emerald-700 text-white text-xs font-medium rounded-lg shadow-sm transition-colors flex items-center gap-1.5"
                    >
                      ⬇️ İndir ({exportResult.file_name})
                    </a>
                  </div>

                  <div className="text-xs font-mono text-slate-700 flex items-center gap-1.5 flex-wrap">
                    <span className="font-semibold text-slate-800">SHA-256:</span>
                    <span className="bg-white px-2 py-0.5 rounded border border-emerald-200 select-all">
                      {exportResult.sha256}
                    </span>
                  </div>

                  {/* Deployment Snippet */}
                  {exportResult.deployment_snippet && (
                    <div className="mt-3">
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-xs font-bold text-slate-700">Dağıtım & Çalıştırma Kodu</span>
                        <button
                          type="button"
                          onClick={() => handleCopySnippet(exportResult.deployment_snippet)}
                          className="text-xs text-purple-700 hover:text-purple-900 font-medium"
                        >
                          {copiedSnippet ? '✓ Kopyalandı' : '📋 Kodu Kopyala'}
                        </button>
                      </div>
                      <pre className="p-3 bg-slate-900 text-slate-100 rounded-lg text-xs font-mono overflow-x-auto whitespace-pre">
                        {exportResult.deployment_snippet}
                      </pre>
                    </div>
                  )}
                </div>
              )}

              {/* Previous Exports History */}
              {modelExportsList && modelExportsList.length > 0 && (
                <div className="pt-4 border-t border-gray-200">
                  <h4 className="text-xs font-bold uppercase tracking-wider text-gray-700 mb-3">
                    Kayıtlı Dışa Aktarımlar ({modelExportsList.length})
                  </h4>
                  <div className="space-y-2 max-h-44 overflow-y-auto">
                    {modelExportsList.map((rec, i) => (
                      <div
                        key={i}
                        className="p-3 bg-gray-50 border border-gray-200 rounded-lg flex items-center justify-between text-xs"
                      >
                        <div className="space-y-0.5">
                          <div className="font-mono font-semibold text-gray-800">
                            {rec.file_name}
                          </div>
                          <div className="text-gray-500 flex items-center gap-2">
                            <span>Format: {rec.export_format.toUpperCase()}</span>
                            <span>•</span>
                            <span>Kuantizasyon: {rec.quantization.toUpperCase()}</span>
                            <span>•</span>
                            <span>{rec.file_size_mb} MB</span>
                          </div>
                        </div>
                        <a
                          href={modelsApi.getDownloadUrl(exportTargetModel.name, rec.file_name, rec.version)}
                          download={rec.file_name}
                          className="px-2.5 py-1 text-xs border border-gray-300 hover:bg-gray-100 text-gray-700 rounded transition-colors"
                        >
                          ⬇️ İndir
                        </a>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
