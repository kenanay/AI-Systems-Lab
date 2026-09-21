'use client';

import React, { useState, useEffect } from 'react';
import Link from 'next/link';
import { useQuery, useMutation } from '@tanstack/react-query';
import {
  MessageSquare,
  Sparkles,
  Sliders,
  Send,
  Cpu,
  Layers,
  Zap,
  CheckCircle2,
  Clock,
  RefreshCw,
  HelpCircle
} from 'lucide-react';
import { api, ModelItem, GenerateRequest, GenerateResponse } from '@/lib/api';

interface GenerationRecord {
  prompt: string;
  generated_text: string;
  tokens_generated: number;
  time_ms: number;
  model_name: string;
  timestamp: string;
}

export default function PlaygroundPage() {
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [prompt, setPrompt] = useState<string>('Yapay zeka sistemleri');
  const [temperature, setTemperature] = useState<number>(0.8);
  const [maxLength, setMaxLength] = useState<number>(50);
  const [topK, setTopK] = useState<number>(50);
  const [topP, setTopP] = useState<number>(0.9);
  const [history, setHistory] = useState<GenerationRecord[]>([]);

  // Fetch models
  const { data: models, isLoading: modelsLoading } = useQuery({
    queryKey: ['models-list'],
    queryFn: () => api.models.list(),
  });

  // Fetch inference status
  const { data: inferenceStatus, refetch: refetchStatus } = useQuery({
    queryKey: ['inference-status'],
    queryFn: () => api.inference.status(),
    refetchInterval: 3000,
  });

  // Auto-select first model if available
  useEffect(() => {
    if (models && models.length > 0 && !selectedModel) {
      setSelectedModel(models[0].model_name);
    }
  }, [models, selectedModel]);

  // Mutations
  const loadMutation = useMutation({
    mutationFn: (modelName: string) => api.inference.load(modelName),
    onSuccess: () => {
      refetchStatus();
    },
  });

  const generateMutation = useMutation({
    mutationFn: (data: GenerateRequest) => api.inference.generate(data),
    onSuccess: (res: GenerateResponse) => {
      const record: GenerationRecord = {
        prompt: res.prompt,
        generated_text: res.generated_text,
        tokens_generated: res.tokens_generated,
        time_ms: res.generation_time_ms,
        model_name: res.model_name,
        timestamp: new Date().toLocaleTimeString(),
      };
      setHistory((prev) => [record, ...prev]);
    },
  });

  const handleLoadModel = () => {
    if (selectedModel) {
      loadMutation.mutate(selectedModel);
    }
  };

  const handleGenerate = (e: React.FormEvent) => {
    e.preventDefault();
    if (!prompt.trim()) return;

    generateMutation.mutate({
      prompt,
      max_length: maxLength,
      temperature,
      top_k: topK,
      top_p: topP,
    });
  };

  const examplePrompts = [
    'Yapay zeka sistemleri yerel',
    'Transformer mimarisi attention',
    'Türkiye geometri ve matematik',
    'Derin öğrenme modelleri',
  ];

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 pb-16">
      {/* Header */}
      <header className="bg-white border-b border-slate-200 px-6 py-4 flex items-center justify-between sticky top-0 z-30 shadow-sm">
        <div className="flex items-center space-x-3">
          <Link href="/" className="text-xl font-bold text-indigo-600 hover:opacity-80">
            🤖 Local AI Research Lab
          </Link>
          <span className="text-slate-300">/</span>
          <h1 className="text-lg font-semibold text-slate-800">Model Playground (Test &amp; Sohbet)</h1>
        </div>
        <div className="flex items-center space-x-3">
          <Link
            href="/training"
            className="px-4 py-2 bg-indigo-50 text-indigo-700 hover:bg-indigo-100 rounded-lg text-sm font-medium transition"
          >
            🏋️ Eğitim Stüdyosu'na Git →
          </Link>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 py-8">
        {/* Status Bar */}
        <div className="bg-white rounded-xl p-4 mb-6 border border-slate-200 shadow-sm flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center space-x-3">
            <div
              className={`w-3.5 h-3.5 rounded-full ${
                inferenceStatus?.model_loaded ? 'bg-emerald-500 animate-pulse' : 'bg-amber-400'
              }`}
            />
            <span className="text-sm font-semibold text-slate-800">
              {inferenceStatus?.model_loaded
                ? `Aktif Model: ${inferenceStatus.model_name}`
                : 'Bellekte Yüklü Model Yok'}
            </span>
            <span className="text-xs px-2 py-0.5 bg-slate-100 text-slate-600 rounded">
              Cihaz: {inferenceStatus?.device?.toUpperCase() || 'CPU'}
            </span>
          </div>

          <div className="flex items-center space-x-2">
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="px-3 py-1.5 border border-slate-300 rounded-lg text-xs bg-white focus:ring-2 focus:ring-indigo-500"
            >
              <option value="">Model Seçin...</option>
              {models?.map((m: any) => (
                <option key={m.model_name} value={m.model_name}>
                  {m.model_name} (v{m.version})
                </option>
              ))}
            </select>
            <button
              onClick={handleLoadModel}
              disabled={!selectedModel || loadMutation.isPending}
              className="px-4 py-1.5 bg-indigo-600 hover:bg-indigo-700 text-white rounded-lg text-xs font-semibold shadow transition disabled:opacity-50"
            >
              {loadMutation.isPending ? 'Yükleniyor...' : 'Modeli Yükle'}
            </button>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          {/* Sol Kolon: Parametreler */}
          <div className="lg:col-span-4 space-y-6">
            <div className="bg-white rounded-xl p-6 border border-slate-200 shadow-sm space-y-5">
              <h3 className="text-sm font-bold text-slate-900 uppercase tracking-wider flex items-center space-x-2">
                <Sliders className="w-4 h-4 text-indigo-600" />
                <span>Çıkarım Parametreleri</span>
              </h3>

              {/* Temperature */}
              <div>
                <div className="flex justify-between text-xs font-semibold text-slate-700 mb-1">
                  <span>Temperature (Sıcaklık)</span>
                  <span className="text-indigo-600">{temperature}</span>
                </div>
                <input
                  type="range"
                  min="0.1"
                  max="1.8"
                  step="0.1"
                  value={temperature}
                  onChange={(e) => setTemperature(parseFloat(e.target.value))}
                  className="w-full accent-indigo-600"
                />
                <p className="text-[11px] text-slate-500 mt-1">
                  Düşük değer daha deterministik (tutarlı), yüksek değer daha yaratıcı metin üretir.
                </p>
              </div>

              {/* Max Length */}
              <div>
                <div className="flex justify-between text-xs font-semibold text-slate-700 mb-1">
                  <span>Max Tokens</span>
                  <span className="text-indigo-600">{maxLength}</span>
                </div>
                <input
                  type="range"
                  min="10"
                  max="200"
                  step="5"
                  value={maxLength}
                  onChange={(e) => setMaxLength(parseInt(e.target.value))}
                  className="w-full accent-indigo-600"
                />
              </div>

              {/* Top-P */}
              <div>
                <div className="flex justify-between text-xs font-semibold text-slate-700 mb-1">
                  <span>Top-P (Nucleus)</span>
                  <span className="text-indigo-600">{topP}</span>
                </div>
                <input
                  type="range"
                  min="0.1"
                  max="1.0"
                  step="0.05"
                  value={topP}
                  onChange={(e) => setTopP(parseFloat(e.target.value))}
                  className="w-full accent-indigo-600"
                />
              </div>

              {/* Top-K */}
              <div>
                <div className="flex justify-between text-xs font-semibold text-slate-700 mb-1">
                  <span>Top-K</span>
                  <span className="text-indigo-600">{topK}</span>
                </div>
                <input
                  type="range"
                  min="1"
                  max="100"
                  step="5"
                  value={topK}
                  onChange={(e) => setTopK(parseInt(e.target.value))}
                  className="w-full accent-indigo-600"
                />
              </div>

              {/* Örnek İstemler */}
              <div className="pt-2 border-t border-slate-100">
                <p className="text-xs font-bold text-slate-700 mb-2 flex items-center space-x-1">
                  <Sparkles className="w-3.5 h-3.5 text-amber-500" />
                  <span>Örnek Başlangıçlar</span>
                </p>
                <div className="flex flex-wrap gap-1.5">
                  {examplePrompts.map((p, i) => (
                    <button
                      key={i}
                      type="button"
                      onClick={() => setPrompt(p)}
                      className="px-2.5 py-1 bg-slate-100 hover:bg-indigo-50 hover:text-indigo-700 text-slate-600 rounded text-xs transition"
                    >
                      {p}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Sağ Kolon: Giriş ve Sonuç */}
          <div className="lg:col-span-8 space-y-6">
            {/* Metin Giriş Kartı */}
            <div className="bg-white rounded-xl p-6 border border-slate-200 shadow-sm space-y-4">
              <form onSubmit={handleGenerate} className="space-y-4">
                <div>
                  <label className="block text-xs font-bold text-slate-700 uppercase tracking-wider mb-1.5">
                    Giriş İstemi (Prompt)
                  </label>
                  <textarea
                    rows={3}
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                    placeholder="Modelin devam ettirmesini istediğiniz metni yazın..."
                    className="w-full p-3 border border-slate-300 rounded-lg text-sm focus:ring-2 focus:ring-indigo-500 focus:outline-none"
                  />
                </div>

                <div className="flex items-center justify-between">
                  <span className="text-xs text-slate-500">
                    {prompt.length} karakter
                  </span>
                  <button
                    type="submit"
                    disabled={generateMutation.isPending || !inferenceStatus?.model_loaded}
                    className="px-6 py-2.5 bg-indigo-600 hover:bg-indigo-700 text-white font-semibold rounded-lg shadow transition flex items-center space-x-2 disabled:opacity-50"
                  >
                    <Send className="w-4 h-4" />
                    <span>{generateMutation.isPending ? 'Üretiliyor...' : 'Metin Üret'}</span>
                  </button>
                </div>
              </form>

              {!inferenceStatus?.model_loaded && (
                <div className="p-3 bg-amber-50 rounded-lg border border-amber-200 text-xs text-amber-800 flex items-center space-x-2">
                  <HelpCircle className="w-4 h-4 text-amber-600 shrink-0" />
                  <span>
                    Metin üretmek için lütfen yukarıdaki menüden bir model seçip <strong>"Modeli Yükle"</strong> butonuna basın.
                  </span>
                </div>
              )}
            </div>

            {/* Çıkarım Geçmişi */}
            <div className="space-y-4">
              <h3 className="text-sm font-bold text-slate-900 uppercase tracking-wider flex items-center space-x-2">
                <MessageSquare className="w-4 h-4 text-indigo-600" />
                <span>Üretim Sonuçları ({history.length})</span>
              </h3>

              {history.length > 0 ? (
                history.map((item, idx) => (
                  <div
                    key={idx}
                    className="bg-white rounded-xl p-5 border border-slate-200 shadow-sm space-y-3"
                  >
                    <div className="flex items-center justify-between text-xs text-slate-400 border-b border-slate-100 pb-2">
                      <span className="font-semibold text-slate-600">{item.model_name}</span>
                      <div className="flex items-center space-x-3">
                        <span className="flex items-center space-x-1">
                          <Zap className="w-3 h-3 text-amber-500" />
                          <span>{item.tokens_generated} token</span>
                        </span>
                        <span className="flex items-center space-x-1">
                          <Clock className="w-3 h-3 text-slate-400" />
                          <span>{item.time_ms.toFixed(1)} ms</span>
                        </span>
                        <span>{item.timestamp}</span>
                      </div>
                    </div>

                    <div className="text-sm space-y-2">
                      <p className="text-slate-500 font-mono text-xs bg-slate-50 p-2 rounded">
                        <strong>Prompt:</strong> {item.prompt}
                      </p>
                      <p className="text-slate-900 font-medium leading-relaxed">
                        {item.generated_text}
                      </p>
                    </div>
                  </div>
                ))
              ) : (
                <div className="bg-white rounded-xl p-12 border border-slate-200 text-center shadow-sm">
                  <Sparkles className="w-10 h-10 text-slate-300 mx-auto mb-2" />
                  <p className="text-sm font-bold text-slate-800">Henüz Metin Üretilmedi</p>
                  <p className="text-xs text-slate-500 mt-1">
                    Yukarıdaki istem alanına bir cümle yazıp "Metin Üret" butonuna basın.
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
