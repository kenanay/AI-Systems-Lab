"use client";

import { useState, useEffect } from 'react';
import Link from 'next/link';

interface ModelVersion {
  version: string;
  created_at: string;
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
}

export default function ModelsPage() {
  const [models, setModels] = useState<Model[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [deleteConfirm, setDeleteConfirm] = useState<string | null>(null);

  useEffect(() => {
    fetchModels();
  }, []);

  const fetchModels = async () => {
    try {
      setLoading(true);
      const response = await fetch('http://localhost:8000/api/v1/models');
      if (!response.ok) throw new Error('Failed to fetch models');
      const data = await response.json();
      setModels(data.models || []);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load models');
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async (modelName: string) => {
    try {
      const response = await fetch(`http://localhost:8000/api/v1/models/${modelName}`, {
        method: 'DELETE',
      });
      
      if (!response.ok) throw new Error('Failed to delete model');
      
      // Refresh models list
      fetchModels();
      setDeleteConfirm(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete model');
    }
  };

  const filteredModels = models.filter(model =>
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
                Trained model'lerin registry'si - versiyonlama, metrikler ve yönetim
              </p>
            </div>
            <div className="flex space-x-3">
              <button
                onClick={fetchModels}
                className="px-4 py-2 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50"
              >
                🔄 Yenile
              </button>
              <Link
                href="/training"
                className="px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700"
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
            placeholder="Model ara..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          />
        </div>

        {/* Stats */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
          <div className="bg-white rounded-lg shadow p-6">
            <div className="text-sm font-medium text-gray-500">Toplam Model</div>
            <div className="mt-2 text-3xl font-bold text-gray-900">{models.length}</div>
          </div>
          <div className="bg-white rounded-lg shadow p-6">
            <div className="text-sm font-medium text-gray-500">Toplam Versiyon</div>
            <div className="mt-2 text-3xl font-bold text-gray-900">
              {models.reduce((sum, m) => sum + (m.versions?.length || 0), 0)}
            </div>
          </div>
          <div className="bg-white rounded-lg shadow p-6">
            <div className="text-sm font-medium text-gray-500">Arama Sonucu</div>
            <div className="mt-2 text-3xl font-bold text-gray-900">{filteredModels.length}</div>
          </div>
        </div>

        {/* Loading State */}
        {loading && (
          <div className="text-center py-12">
            <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
            <p className="mt-4 text-gray-600">Model'ler yükleniyor...</p>
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
          <div className="text-center py-12">
            <div className="text-6xl mb-4">📦</div>
            <h3 className="text-lg font-medium text-gray-900">Henüz model yok</h3>
            <p className="mt-2 text-sm text-gray-500">
              İlk modelinizi eğitmek için Training Lab'e gidin.
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
              
              return (
                <div key={model.name} className="bg-white rounded-lg shadow hover:shadow-md transition-shadow">
                  <div className="p-6">
                    <div className="flex items-center justify-between">
                      <div className="flex-1">
                        <div className="flex items-center space-x-3">
                          <h3 className="text-lg font-semibold text-gray-900">{model.name}</h3>
                          {model.latest_version && (
                            <span className="px-2 py-1 text-xs font-medium bg-blue-100 text-blue-800 rounded">
                              v{model.latest_version}
                            </span>
                          )}
                        </div>
                        {model.description && (
                          <p className="mt-1 text-sm text-gray-500">{model.description}</p>
                        )}
                        
                        {/* Tags */}
                        {model.tags && model.tags.length > 0 && (
                          <div className="mt-2 flex flex-wrap gap-2">
                            {model.tags.map((tag, idx) => (
                              <span
                                key={idx}
                                className="px-2 py-1 text-xs bg-gray-100 text-gray-700 rounded"
                              >
                                {tag}
                              </span>
                            ))}
                          </div>
                        )}
                      </div>

                      {/* Actions */}
                      <div className="flex items-center space-x-2 ml-4">
                        {latestMetrics.perplexity && (
                          <div className="text-right mr-4">
                            <div className="text-xs text-gray-500">Perplexity</div>
                            <div className="text-lg font-semibold text-gray-900">
                              {latestMetrics.perplexity.toFixed(2)}
                            </div>
                          </div>
                        )}
                        
                        <button
                          onClick={() => setSelectedModel(isExpanded ? null : model.name)}
                          className="px-3 py-2 text-sm border border-gray-300 rounded-md hover:bg-gray-50"
                        >
                          {isExpanded ? '▲ Gizle' : '▼ Detay'}
                        </button>
                        
                        <Link
                          href={`/playground?model=${model.name}`}
                          className="px-3 py-2 text-sm bg-blue-600 text-white rounded-md hover:bg-blue-700"
                        >
                          🎮 Test Et
                        </Link>
                        
                        {deleteConfirm === model.name ? (
                          <div className="flex space-x-2">
                            <button
                              onClick={() => handleDelete(model.name)}
                              className="px-3 py-2 text-sm bg-red-600 text-white rounded-md hover:bg-red-700"
                            >
                              ✓ Onayla
                            </button>
                            <button
                              onClick={() => setDeleteConfirm(null)}
                              className="px-3 py-2 text-sm border border-gray-300 rounded-md hover:bg-gray-50"
                            >
                              ✕ İptal
                            </button>
                          </div>
                        ) : (
                          <button
                            onClick={() => setDeleteConfirm(model.name)}
                            className="px-3 py-2 text-sm text-red-600 border border-red-300 rounded-md hover:bg-red-50"
                          >
                            🗑️ Sil
                          </button>
                        )}
                      </div>
                    </div>

                    {/* Expanded Details */}
                    {isExpanded && (
                      <div className="mt-6 pt-6 border-t border-gray-200">
                        <h4 className="text-sm font-medium text-gray-900 mb-4">Versiyonlar</h4>
                        <div className="space-y-3">
                          {model.versions && model.versions.map((version, idx) => (
                            <div
                              key={idx}
                              className="p-4 bg-gray-50 rounded-lg"
                            >
                              <div className="flex items-center justify-between mb-2">
                                <span className="font-mono text-sm font-medium">v{version.version}</span>
                                <span className="text-xs text-gray-500">
                                  {new Date(version.created_at).toLocaleDateString('tr-TR')}
                                </span>
                              </div>
                              
                              {version.metrics && (
                                <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
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
                                <div className="mt-3 text-xs text-gray-600">
                                  <span className="font-medium">Config:</span>{' '}
                                  {Object.entries(version.training_config)
                                    .slice(0, 3)
                                    .map(([k, v]) => `${k}=${v}`)
                                    .join(', ')}
                                </div>
                              )}
                            </div>
                          ))}
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
          <div className="text-center py-12">
            <div className="text-4xl mb-4">🔍</div>
            <h3 className="text-lg font-medium text-gray-900">Sonuç bulunamadı</h3>
            <p className="mt-2 text-sm text-gray-500">
              "{searchQuery}" için model bulunamadı.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
