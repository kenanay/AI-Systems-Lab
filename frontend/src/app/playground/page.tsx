'use client';

import React, { useState, useEffect, useRef } from 'react';
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
  HelpCircle,
  GitBranch,
  BarChart3,
  Play,
  Square,
  Copy,
  Check,
  RotateCcw,
  ArrowRight,
  Terminal,
  Activity,
  ChevronRight,
  Maximize2
} from 'lucide-react';
import {
  api,
  API_BASE_URL,
  ModelItem,
  GenerateRequest,
  GenerateResponse,
  BeamSearchResponse,
  BeamHypothesisResponse,
  NextTokenProbsResponse,
  NextTokenCandidate,
} from '@/lib/api';

type PlaygroundMode = 'streaming' | 'beam_search' | 'logits' | 'standard';

interface GenerationRecord {
  id: string;
  prompt: string;
  generated_text: string;
  tokens_generated: number;
  time_ms: number;
  model_name: string;
  mode: string;
  timestamp: string;
}

export default function PlaygroundPage() {
  // Mode selection
  const [mode, setMode] = useState<PlaygroundMode>('streaming');

  // Model & Prompt state
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [prompt, setPrompt] = useState<string>('Yapay zeka sistemleri');

  // Standard / Streaming parameters
  const [temperature, setTemperature] = useState<number>(0.7);
  const [maxLength, setMaxLength] = useState<number>(60);
  const [topK, setTopK] = useState<number>(50);
  const [topP, setTopP] = useState<number>(0.9);

  // Beam Search parameters
  const [beamWidth, setBeamWidth] = useState<number>(3);
  const [lengthPenalty, setLengthPenalty] = useState<number>(1.0);
  const [numReturnSeqs, setNumReturnSeqs] = useState<number>(3);
  const [beamResults, setBeamResults] = useState<BeamSearchResponse | null>(null);
  const [isBeamLoading, setIsBeamLoading] = useState<boolean>(false);

  // Next Token Logits parameters & state
  const [logitsTopK, setLogitsTopK] = useState<number>(8);
  const [logitsTemp, setLogitsTemp] = useState<number>(0.8);
  const [logitsResults, setLogitsResults] = useState<NextTokenProbsResponse | null>(null);
  const [isLogitsLoading, setIsLogitsLoading] = useState<boolean>(false);

  // Streaming state
  const [isStreaming, setIsStreaming] = useState<boolean>(false);
  const [streamedText, setStreamedText] = useState<string>('');
  const [streamTokensCount, setStreamTokensCount] = useState<number>(0);
  const [streamTokensPerSec, setStreamTokensPerSec] = useState<number>(0);
  const [streamElapsedTime, setStreamElapsedTime] = useState<number>(0);
  const abortControllerRef = useRef<AbortController | null>(null);
  const streamTimerRef = useRef<NodeJS.Timeout | null>(null);

  // History and copy feedback
  const [history, setHistory] = useState<GenerationRecord[]>([]);
  const [copiedId, setCopiedId] = useState<string | null>(null);

  // Fetch available models
  const { data: models } = useQuery({
    queryKey: ['models-list'],
    queryFn: () => api.models.list(),
  });

  // Fetch inference status
  const { data: inferenceStatus, refetch: refetchStatus } = useQuery({
    queryKey: ['inference-status'],
    queryFn: () => api.inference.status(),
    refetchInterval: 3000,
  });

  // Model loading mutation
  const loadMutation = useMutation({
    mutationFn: (modelName: string) => api.inference.load(modelName),
    onSuccess: () => {
      refetchStatus();
    },
  });

  // Standard generation mutation
  const generateMutation = useMutation({
    mutationFn: (data: GenerateRequest) => api.inference.generate(data),
    onSuccess: (res: GenerateResponse) => {
      const record: GenerationRecord = {
        id: Math.random().toString(36).substring(7),
        prompt: res.prompt,
        generated_text: res.generated_text,
        tokens_generated: res.tokens_generated,
        time_ms: res.generation_time_ms,
        model_name: res.model_name,
        mode: 'Standart',
        timestamp: new Date().toLocaleTimeString(),
      };
      setHistory((prev) => [record, ...prev]);
    },
  });

  // Auto-select model from URL query or first available model
  useEffect(() => {
    if (typeof window !== 'undefined') {
      const urlModel = new URLSearchParams(window.location.search).get('model');
      if (urlModel) {
        setSelectedModel(urlModel);
        if (inferenceStatus && inferenceStatus.model_name !== urlModel && !loadMutation.isPending) {
          loadMutation.mutate(urlModel);
        }
        return;
      }
    }
    if (models && models.length > 0 && !selectedModel) {
      setSelectedModel(models[0].model_name);
    }
  }, [models, selectedModel, inferenceStatus?.model_name]);

  // Clean up streaming on unmount
  useEffect(() => {
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
      if (streamTimerRef.current) {
        clearInterval(streamTimerRef.current);
      }
    };
  }, []);

  const handleLoadModel = () => {
    if (selectedModel) {
      loadMutation.mutate(selectedModel);
    }
  };

  const handleCopyText = (text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  // --- 1. Streaming Generator Handler ---
  const handleStartStream = async () => {
    if (!prompt.trim() || isStreaming) return;

    setIsStreaming(true);
    setStreamedText('');
    setStreamTokensCount(0);
    setStreamTokensPerSec(0);
    setStreamElapsedTime(0);

    const controller = new AbortController();
    abortControllerRef.current = controller;

    const startTime = performance.now();
    let accumulatedText = '';
    let tokensReceived = 0;

    // Periodic elapsed timer
    streamTimerRef.current = setInterval(() => {
      const elapsedSec = (performance.now() - startTime) / 1000;
      setStreamElapsedTime(Math.round(elapsedSec * 10) / 10);
    }, 100);

    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/inference/generate/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt,
          max_length: maxLength,
          temperature,
          top_k: topK > 0 ? topK : undefined,
          top_p: topP < 1.0 ? topP : undefined,
        }),
        signal: controller.signal,
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const reader = response.body?.getReader();
      if (!reader) throw new Error('SSE Akış okuyucu başlatılamadı.');

      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const events = buffer.split('\n\n');
        buffer = events.pop() || '';

        for (const event of events) {
          const trimmed = event.trim();
          if (trimmed.startsWith('data: ')) {
            try {
              const data = JSON.parse(trimmed.slice(6));
              if (data.token) {
                tokensReceived += 1;
                accumulatedText += data.token;
                setStreamedText(accumulatedText);
                setStreamTokensCount(tokensReceived);

                const currentElapsed = (performance.now() - startTime) / 1000;
                if (currentElapsed > 0.05) {
                  setStreamTokensPerSec(Math.round((tokensReceived / currentElapsed) * 10) / 10);
                }
              }
              if (data.done) {
                const totalElapsedMs = performance.now() - startTime;
                const record: GenerationRecord = {
                  id: Math.random().toString(36).substring(7),
                  prompt,
                  generated_text: accumulatedText,
                  tokens_generated: tokensReceived,
                  time_ms: Math.round(totalElapsedMs),
                  model_name: data.model_name || inferenceStatus?.model_name || 'GPT-Model',
                  mode: 'Akışlı (SSE)',
                  timestamp: new Date().toLocaleTimeString(),
                };
                setHistory((prev) => [record, ...prev]);
              }
            } catch (jsonErr) {
              console.warn('SSE Parse error', jsonErr);
            }
          }
        }
      }
    } catch (err: any) {
      if (err.name !== 'AbortError') {
        console.error('Streaming error:', err);
        setStreamedText((prev) => prev + `\n[Akış Hatası: ${err.message || 'Bilinmeyen hata'}]`);
      }
    } finally {
      setIsStreaming(false);
      if (streamTimerRef.current) {
        clearInterval(streamTimerRef.current);
      }
    }
  };

  const handleStopStream = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    setIsStreaming(false);
    if (streamTimerRef.current) {
      clearInterval(streamTimerRef.current);
    }
  };

  // --- 2. Beam Search Handler ---
  const handleRunBeamSearch = async () => {
    if (!prompt.trim() || isBeamLoading) return;

    setIsBeamLoading(true);
    try {
      const res = await api.inference.beamSearch({
        prompt,
        beam_width: beamWidth,
        max_length: maxLength,
        length_penalty: lengthPenalty,
        num_return_sequences: Math.min(numReturnSeqs, beamWidth),
      });
      setBeamResults(res);

      if (res.hypotheses.length > 0) {
        const topHypo = res.hypotheses[0];
        const record: GenerationRecord = {
          id: Math.random().toString(36).substring(7),
          prompt: res.prompt,
          generated_text: topHypo.text,
          tokens_generated: topHypo.length,
          time_ms: res.execution_time_ms,
          model_name: res.model_name,
          mode: `Beam Search (K=${res.beam_width})`,
          timestamp: new Date().toLocaleTimeString(),
        };
        setHistory((prev) => [record, ...prev]);
      }
    } catch (err: any) {
      console.error('Beam Search error:', err);
    } finally {
      setIsBeamLoading(false);
    }
  };

  // --- 3. Next Token Logits Handler ---
  const handleInspectLogits = async (inputPrompt?: string) => {
    const textToInspect = inputPrompt !== undefined ? inputPrompt : prompt;
    if (!textToInspect.trim() || isLogitsLoading) return;

    setIsLogitsLoading(true);
    try {
      const res = await api.inference.nextTokenProbs({
        prompt: textToInspect,
        top_k: logitsTopK,
        temperature: logitsTemp,
      });
      setLogitsResults(res);
    } catch (err: any) {
      console.error('Logits inspect error:', err);
    } finally {
      setIsLogitsLoading(false);
    }
  };

  // Step-by-step interactive decoding
  const handleAppendTokenAndAdvance = (candidate: NextTokenCandidate) => {
    const nextPrompt = prompt + candidate.token_text;
    setPrompt(nextPrompt);
    handleInspectLogits(nextPrompt);
  };

  // --- 4. Standard Generate Handler ---
  const handleStandardGenerate = (e: React.FormEvent) => {
    e.preventDefault();
    if (!prompt.trim()) return;

    generateMutation.mutate({
      prompt,
      max_length: maxLength,
      temperature,
      top_k: topK > 0 ? topK : undefined,
      top_p: topP < 1.0 ? topP : undefined,
    });
  };

  const examplePrompts = [
    { tag: 'Algoritma', text: '### Talimat:\nİkili arama (Binary Search) algoritmasının mantığını ve O(log n) karmaşıklığını açıklayın.\n\n### Yanıt:' },
    { tag: 'Kodlama', text: '### Talimat:\nPython ile bir dizideki tekil elemanları bulan optimize bir fonksiyon yazın.\n\n### Yanıt:' },
    { tag: 'Mimari', text: 'Transformer mimarisinde Multi-Head Attention mekanizması neden tekil dikkate göre daha güçlüdür?' },
    { tag: 'Optimizasyon', text: 'Büyük dil modellerinde LoRA (Low-Rank Adaptation) tekniğinin parametre verimliliği avantajları:' },
    { tag: 'Sistem', text: 'Dağıtık derin öğrenmede Pipeline Parallelism ve Tensor Parallelism arasındaki temel farklar:' }
  ];

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 pb-16">
      {/* Top Header */}
      <header className="bg-white border-b border-slate-200 px-6 py-4 flex items-center justify-between sticky top-0 z-30 shadow-sm">
        <div className="flex items-center space-x-3">
          <Link href="/" className="text-xl font-bold text-indigo-600 hover:opacity-80 transition">
            🤖 Local AI Research Lab
          </Link>
          <span className="text-slate-300">/</span>
          <div className="flex items-center space-x-2">
            <h1 className="text-lg font-bold text-slate-800">Model Playground &amp; Decoding Lab</h1>
            <span className="px-2 py-0.5 bg-indigo-100 text-indigo-800 rounded-full text-xs font-semibold">
              v2.0
            </span>
          </div>
        </div>
        <div className="flex items-center space-x-3">
          <Link
            href="/attention-lab"
            className="px-3 py-1.5 bg-slate-100 hover:bg-slate-200 text-slate-700 rounded-lg text-xs font-medium transition flex items-center space-x-1"
          >
            <Activity className="w-3.5 h-3.5 text-indigo-600" />
            <span>Attention Lab</span>
          </Link>
          <Link
            href="/training"
            className="px-3.5 py-1.5 bg-indigo-50 text-indigo-700 hover:bg-indigo-100 rounded-lg text-xs font-semibold transition"
          >
            🏋️ Eğitim Stüdyosu →
          </Link>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 py-6">
        {/* Model Status Bar */}
        <div className="bg-white rounded-xl p-4 mb-6 border border-slate-200 shadow-sm flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center space-x-3">
            <div
              className={`w-3.5 h-3.5 rounded-full ${
                inferenceStatus?.model_loaded ? 'bg-emerald-500 ring-4 ring-emerald-100 animate-pulse' : 'bg-amber-400'
              }`}
            />
            <div>
              <div className="text-sm font-bold text-slate-800 flex items-center space-x-2">
                <span>{inferenceStatus?.model_loaded ? `Aktif: ${inferenceStatus.model_name}` : 'Hazırda Model Bekleniyor'}</span>
                {inferenceStatus?.model_loaded && (
                  <span className="px-2 py-0.5 bg-emerald-50 text-emerald-700 border border-emerald-200 rounded text-[11px] font-medium">
                    Çıkarıma Hazır
                  </span>
                )}
              </div>
              <p className="text-xs text-slate-500">
                Donanım Hızlandırıcı: <span className="font-semibold text-slate-700">{inferenceStatus?.device?.toUpperCase() || 'CPU'}</span>
              </p>
            </div>
          </div>

          <div className="flex items-center space-x-2">
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="px-3 py-1.5 border border-slate-300 rounded-lg text-xs bg-white text-slate-700 focus:ring-2 focus:ring-indigo-500 focus:outline-none"
            >
              <option value="">Kayıtlı Model Seçin...</option>
              {models?.map((m: any) => (
                <option key={m.model_name} value={m.model_name}>
                  {m.model_name} (v{m.version || '1.0'})
                </option>
              ))}
            </select>
            <button
              onClick={handleLoadModel}
              disabled={!selectedModel || loadMutation.isPending}
              className="px-4 py-1.5 bg-indigo-600 hover:bg-indigo-700 text-white rounded-lg text-xs font-semibold shadow transition disabled:opacity-50 flex items-center space-x-1.5"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${loadMutation.isPending ? 'animate-spin' : ''}`} />
              <span>{loadMutation.isPending ? 'Yükleniyor...' : 'Modeli Belleğe Al'}</span>
            </button>
          </div>
        </div>

        {/* Mode Selector Tabs */}
        <div className="flex items-center space-x-2 mb-6 border-b border-slate-200 pb-3 overflow-x-auto">
          <button
            onClick={() => setMode('streaming')}
            className={`px-4 py-2 rounded-lg text-xs font-bold transition flex items-center space-x-2 whitespace-nowrap ${
              mode === 'streaming'
                ? 'bg-indigo-600 text-white shadow'
                : 'bg-white text-slate-600 hover:bg-slate-100 border border-slate-200'
            }`}
          >
            <Zap className="w-4 h-4 text-amber-300" />
            <span>⚡ Gerçek Zamanlı Akış (Streaming SSE)</span>
          </button>

          <button
            onClick={() => setMode('beam_search')}
            className={`px-4 py-2 rounded-lg text-xs font-bold transition flex items-center space-x-2 whitespace-nowrap ${
              mode === 'beam_search'
                ? 'bg-indigo-600 text-white shadow'
                : 'bg-white text-slate-600 hover:bg-slate-100 border border-slate-200'
            }`}
          >
            <GitBranch className="w-4 h-4 text-emerald-300" />
            <span>🌳 Işın Araması (Beam Search)</span>
          </button>

          <button
            onClick={() => {
              setMode('logits');
              if (!logitsResults) {
                handleInspectLogits();
              }
            }}
            className={`px-4 py-2 rounded-lg text-xs font-bold transition flex items-center space-x-2 whitespace-nowrap ${
              mode === 'logits'
                ? 'bg-indigo-600 text-white shadow'
                : 'bg-white text-slate-600 hover:bg-slate-100 border border-slate-200'
            }`}
          >
            <BarChart3 className="w-4 h-4 text-cyan-300" />
            <span>📊 Token Olasılık Dağılımı (Logits)</span>
          </button>

          <button
            onClick={() => setMode('standard')}
            className={`px-4 py-2 rounded-lg text-xs font-bold transition flex items-center space-x-2 whitespace-nowrap ${
              mode === 'standard'
                ? 'bg-indigo-600 text-white shadow'
                : 'bg-white text-slate-600 hover:bg-slate-100 border border-slate-200'
            }`}
          >
            <MessageSquare className="w-4 h-4 text-slate-400" />
            <span>📜 Toplu Çıkarım &amp; Geçmiş</span>
          </button>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          {/* Sol Kolon: Parametreler & Örnekler */}
          <div className="lg:col-span-4 space-y-6">
            <div className="bg-white rounded-xl p-5 border border-slate-200 shadow-sm space-y-5">
              <div className="flex items-center justify-between border-b border-slate-100 pb-3">
                <h3 className="text-xs font-bold text-slate-900 uppercase tracking-wider flex items-center space-x-2">
                  <Sliders className="w-4 h-4 text-indigo-600" />
                  <span>
                    {mode === 'beam_search'
                      ? 'Işın Parametreleri'
                      : mode === 'logits'
                      ? 'Logits Parametreleri'
                      : 'Decoding Hiperparametreleri'}
                  </span>
                </h3>
                <span className="text-[11px] px-2 py-0.5 bg-slate-100 font-mono text-slate-600 rounded">
                  Mod: {mode}
                </span>
              </div>

              {/* Mode Specific Controls */}
              {mode === 'beam_search' ? (
                <>
                  {/* Beam Width */}
                  <div>
                    <div className="flex justify-between text-xs font-semibold text-slate-700 mb-1">
                      <span>Beam Width (Işın Genişliği K)</span>
                      <span className="text-indigo-600 font-bold">{beamWidth}</span>
                    </div>
                    <input
                      type="range"
                      min="2"
                      max="6"
                      step="1"
                      value={beamWidth}
                      onChange={(e) => setBeamWidth(parseInt(e.target.value))}
                      className="w-full accent-indigo-600"
                    />
                    <p className="text-[11px] text-slate-500 mt-1">
                      Aynı anda takip edilen en yüksek olasılıklı hipotez dalı sayısı.
                    </p>
                  </div>

                  {/* Length Penalty */}
                  <div>
                    <div className="flex justify-between text-xs font-semibold text-slate-700 mb-1">
                      <span>Length Penalty (Uzunluk Cezası α)</span>
                      <span className="text-indigo-600 font-bold">{lengthPenalty.toFixed(1)}</span>
                    </div>
                    <input
                      type="range"
                      min="0.5"
                      max="2.0"
                      step="0.1"
                      value={lengthPenalty}
                      onChange={(e) => setLengthPenalty(parseFloat(e.target.value))}
                      className="w-full accent-indigo-600"
                    />
                    <p className="text-[11px] text-slate-500 mt-1">
                      1.0 = Dengeli normalizasyon, &gt;1.0 daha uzun cümleleri ödüllendirir.
                    </p>
                  </div>

                  {/* Max Length */}
                  <div>
                    <div className="flex justify-between text-xs font-semibold text-slate-700 mb-1">
                      <span>Max Tokens</span>
                      <span className="text-indigo-600 font-bold">{maxLength}</span>
                    </div>
                    <input
                      type="range"
                      min="10"
                      max="120"
                      step="5"
                      value={maxLength}
                      onChange={(e) => setMaxLength(parseInt(e.target.value))}
                      className="w-full accent-indigo-600"
                    />
                  </div>

                  {/* Return Sequences */}
                  <div>
                    <div className="flex justify-between text-xs font-semibold text-slate-700 mb-1">
                      <span>Döndürülecek Hipotez Sayısı</span>
                      <span className="text-indigo-600 font-bold">{Math.min(numReturnSeqs, beamWidth)}</span>
                    </div>
                    <input
                      type="range"
                      min="1"
                      max={beamWidth}
                      step="1"
                      value={Math.min(numReturnSeqs, beamWidth)}
                      onChange={(e) => setNumReturnSeqs(parseInt(e.target.value))}
                      className="w-full accent-indigo-600"
                    />
                  </div>
                </>
              ) : mode === 'logits' ? (
                <>
                  {/* Logits Temperature */}
                  <div>
                    <div className="flex justify-between text-xs font-semibold text-slate-700 mb-1">
                      <span>Temperature (Softmax Sıcaklığı)</span>
                      <span className="text-cyan-600 font-bold">{logitsTemp.toFixed(1)}</span>
                    </div>
                    <input
                      type="range"
                      min="0.1"
                      max="2.0"
                      step="0.1"
                      value={logitsTemp}
                      onChange={(e) => {
                        const val = parseFloat(e.target.value);
                        setLogitsTemp(val);
                      }}
                      className="w-full accent-cyan-600"
                    />
                    <p className="text-[11px] text-slate-500 mt-1">
                      Düşük değer olasılık tepe noktasını sivriltir; yüksek değer dağılımı yayvanlaştırır.
                    </p>
                  </div>

                  {/* Top-K Logits */}
                  <div>
                    <div className="flex justify-between text-xs font-semibold text-slate-700 mb-1">
                      <span>Görüntülenecek Aday Sayısı (Top-K)</span>
                      <span className="text-cyan-600 font-bold">{logitsTopK}</span>
                    </div>
                    <input
                      type="range"
                      min="3"
                      max="15"
                      step="1"
                      value={logitsTopK}
                      onChange={(e) => setLogitsTopK(parseInt(e.target.value))}
                      className="w-full accent-cyan-600"
                    />
                  </div>
                </>
              ) : (
                <>
                  {/* Temperature */}
                  <div>
                    <div className="flex justify-between text-xs font-semibold text-slate-700 mb-1">
                      <span>Temperature (Sıcaklık)</span>
                      <span className="text-indigo-600 font-bold">{temperature}</span>
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
                      0.2 = Çok tutarlı/deterministik, 1.2+ = Yaratıcı ve çeşitlilik dolu.
                    </p>
                  </div>

                  {/* Max Length */}
                  <div>
                    <div className="flex justify-between text-xs font-semibold text-slate-700 mb-1">
                      <span>Max Tokens</span>
                      <span className="text-indigo-600 font-bold">{maxLength}</span>
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
                      <span>Top-P (Nucleus Sampling)</span>
                      <span className="text-indigo-600 font-bold">{topP}</span>
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
                      <span>Top-K Filtresi</span>
                      <span className="text-indigo-600 font-bold">{topK}</span>
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
                </>
              )}

              {/* Quick Prompts */}
              <div className="pt-3 border-t border-slate-100">
                <p className="text-xs font-bold text-slate-700 mb-2 flex items-center space-x-1">
                  <Sparkles className="w-3.5 h-3.5 text-amber-500" />
                  <span>Hazır İstem Şablonları</span>
                </p>
                <div className="space-y-1.5">
                  {examplePrompts.map((p, i) => (
                    <button
                      key={i}
                      type="button"
                      onClick={() => {
                        setPrompt(p.text);
                        if (mode === 'logits') {
                          handleInspectLogits(p.text);
                        }
                      }}
                      className="w-full text-left p-2 bg-slate-50 hover:bg-indigo-50 hover:text-indigo-700 rounded-lg text-xs transition border border-slate-100 flex items-center justify-between group"
                    >
                      <span className="truncate pr-2">{p.text.split('\n')[0].replace('### Talimat:\n', '')}</span>
                      <span className="shrink-0 text-[10px] px-1.5 py-0.5 bg-white group-hover:bg-indigo-100 text-slate-500 rounded border border-slate-200">
                        {p.tag}
                      </span>
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Sağ Kolon: Giriş Kartı & Sonuçlar */}
          <div className="lg:col-span-8 space-y-6">
            {/* Metin Giriş Kartı */}
            <div className="bg-white rounded-xl p-5 border border-slate-200 shadow-sm space-y-4">
              <div>
                <div className="flex items-center justify-between mb-1.5">
                  <label className="text-xs font-bold text-slate-700 uppercase tracking-wider flex items-center space-x-1.5">
                    <span>Giriş İstemi (Prompt Context)</span>
                    {mode === 'logits' && (
                      <span className="text-[10px] lowercase px-2 py-0.5 bg-cyan-50 text-cyan-700 border border-cyan-200 rounded">
                        son token inceleniyor
                      </span>
                    )}
                  </label>
                  <span className="text-xs text-slate-400 font-mono">
                    {prompt.length} karakter • ~{Math.ceil(prompt.length / 4)} token
                  </span>
                </div>
                <textarea
                  rows={3}
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  placeholder="Modelin devam ettirmesini istediğiniz metni veya talimatı yazın..."
                  className="w-full p-3 border border-slate-300 rounded-lg text-sm font-sans focus:ring-2 focus:ring-indigo-500 focus:outline-none leading-relaxed"
                />
              </div>

              {/* Action Buttons depending on Mode */}
              <div className="flex flex-wrap items-center justify-between gap-3 pt-1">
                <button
                  type="button"
                  onClick={() => setPrompt('')}
                  className="text-xs text-slate-400 hover:text-slate-600 flex items-center space-x-1 transition"
                >
                  <RotateCcw className="w-3.5 h-3.5" />
                  <span>Temizle</span>
                </button>

                <div className="flex items-center space-x-2">
                  {mode === 'streaming' && (
                    <>
                      {isStreaming ? (
                        <button
                          type="button"
                          onClick={handleStopStream}
                          className="px-5 py-2.5 bg-rose-600 hover:bg-rose-700 text-white font-semibold rounded-lg shadow transition flex items-center space-x-2 animate-pulse text-xs"
                        >
                          <Square className="w-4 h-4 fill-white" />
                          <span>Üretimi Durdur</span>
                        </button>
                      ) : (
                        <button
                          type="button"
                          onClick={handleStartStream}
                          disabled={!prompt.trim()}
                          className="px-6 py-2.5 bg-indigo-600 hover:bg-indigo-700 text-white font-semibold rounded-lg shadow transition flex items-center space-x-2 disabled:opacity-50 text-xs"
                        >
                          <Zap className="w-4 h-4 text-amber-300" />
                          <span>Akışlı Üret (SSE)</span>
                        </button>
                      )}
                    </>
                  )}

                  {mode === 'beam_search' && (
                    <button
                      type="button"
                      onClick={handleRunBeamSearch}
                      disabled={isBeamLoading || !prompt.trim()}
                      className="px-6 py-2.5 bg-emerald-600 hover:bg-emerald-700 text-white font-semibold rounded-lg shadow transition flex items-center space-x-2 disabled:opacity-50 text-xs"
                    >
                      <GitBranch className="w-4 h-4" />
                      <span>{isBeamLoading ? 'Işınlar Taranıyor...' : 'Işın Araması Yap (K-Beams)'}</span>
                    </button>
                  )}

                  {mode === 'logits' && (
                    <button
                      type="button"
                      onClick={() => handleInspectLogits()}
                      disabled={isLogitsLoading || !prompt.trim()}
                      className="px-6 py-2.5 bg-cyan-600 hover:bg-cyan-700 text-white font-semibold rounded-lg shadow transition flex items-center space-x-2 disabled:opacity-50 text-xs"
                    >
                      <BarChart3 className="w-4 h-4" />
                      <span>{isLogitsLoading ? 'Hesaplanıyor...' : 'Sonraki Token Dağılımını Al'}</span>
                    </button>
                  )}

                  {mode === 'standard' && (
                    <button
                      type="button"
                      onClick={handleStandardGenerate}
                      disabled={generateMutation.isPending || !prompt.trim()}
                      className="px-6 py-2.5 bg-indigo-600 hover:bg-indigo-700 text-white font-semibold rounded-lg shadow transition flex items-center space-x-2 disabled:opacity-50 text-xs"
                    >
                      <Send className="w-4 h-4" />
                      <span>{generateMutation.isPending ? 'Üretiliyor...' : 'Klasik Üret'}</span>
                    </button>
                  )}
                </div>
              </div>
            </div>

            {/* MODE 1: STREAMING OUTPUT PANEL */}
            {mode === 'streaming' && (
              <div className="bg-slate-900 rounded-xl p-5 border border-slate-800 shadow-xl text-slate-100 space-y-4">
                <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                  <div className="flex items-center space-x-2">
                    <Terminal className="w-4 h-4 text-emerald-400" />
                    <span className="text-xs font-mono font-semibold text-slate-300">
                      Gerçek Zamanlı Token Akışı
                    </span>
                    {isStreaming && (
                      <span className="flex items-center space-x-1.5 px-2 py-0.5 bg-emerald-950 border border-emerald-800 text-emerald-300 rounded text-[11px] font-mono">
                        <span className="w-2 h-2 rounded-full bg-emerald-400 animate-ping" />
                        <span>Akıyor</span>
                      </span>
                    )}
                  </div>

                  <div className="flex items-center space-x-3 text-xs font-mono">
                    <div className="px-2.5 py-1 bg-slate-800 rounded border border-slate-700 flex items-center space-x-1.5">
                      <Zap className="w-3.5 h-3.5 text-amber-400" />
                      <span className="text-amber-300 font-bold">{streamTokensPerSec}</span>
                      <span className="text-slate-400 text-[11px]">tok/s</span>
                    </div>

                    <div className="px-2.5 py-1 bg-slate-800 rounded border border-slate-700 flex items-center space-x-1.5">
                      <Clock className="w-3.5 h-3.5 text-slate-400" />
                      <span className="text-slate-300">{streamElapsedTime}s</span>
                    </div>

                    <div className="px-2.5 py-1 bg-slate-800 rounded border border-slate-700">
                      <span className="text-slate-400">{streamTokensCount} token</span>
                    </div>

                    {streamedText && (
                      <button
                        onClick={() => handleCopyText(streamedText, 'stream-active')}
                        className="p-1.5 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded border border-slate-700 transition"
                        title="Metni Kopyala"
                      >
                        {copiedId === 'stream-active' ? (
                          <Check className="w-3.5 h-3.5 text-emerald-400" />
                        ) : (
                          <Copy className="w-3.5 h-3.5" />
                        )}
                      </button>
                    )}
                  </div>
                </div>

                {/* Typewriter Stream Box */}
                <div className="min-h-[160px] max-h-[360px] overflow-y-auto font-mono text-sm leading-relaxed p-4 bg-slate-950/60 rounded-lg border border-slate-800/80 whitespace-pre-wrap">
                  {streamedText ? (
                    <span>
                      <span className="text-slate-500 select-none">{prompt}</span>{' '}
                      <span className="text-emerald-300 font-semibold">{streamedText}</span>
                      {isStreaming && (
                        <span className="inline-block w-2 h-4 ml-0.5 bg-emerald-400 animate-pulse align-middle" />
                      )}
                    </span>
                  ) : (
                    <span className="text-slate-600 italic">
                      İsteminizi yazıp "Akışlı Üret (SSE)" butonuna bastığınızda, modelin ürettiği her token burada anlık daktilo animasyonuyla akacaktır.
                    </span>
                  )}
                </div>
              </div>
            )}

            {/* MODE 2: BEAM SEARCH RESULTS */}
            {mode === 'beam_search' && (
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <h3 className="text-sm font-bold text-slate-900 uppercase tracking-wider flex items-center space-x-2">
                    <GitBranch className="w-4 h-4 text-emerald-600" />
                    <span>Işın Araması Hipotezleri ({beamResults?.hypotheses.length || 0})</span>
                  </h3>
                  {beamResults && (
                    <span className="text-xs text-slate-500 font-mono">
                      Hesaplama: {beamResults.execution_time_ms.toFixed(1)} ms • K={beamResults.beam_width}
                    </span>
                  )}
                </div>

                {beamResults?.hypotheses && beamResults.hypotheses.length > 0 ? (
                  beamResults.hypotheses.map((hypo, idx) => (
                    <div
                      key={hypo.hypothesis_id}
                      className={`bg-white rounded-xl p-5 border transition shadow-sm space-y-3 ${
                        idx === 0
                          ? 'border-emerald-300 ring-2 ring-emerald-100'
                          : 'border-slate-200'
                      }`}
                    >
                      <div className="flex items-center justify-between border-b border-slate-100 pb-2">
                        <div className="flex items-center space-x-2">
                          <span
                            className={`px-2 py-0.5 rounded text-xs font-bold font-mono ${
                              idx === 0
                                ? 'bg-amber-100 text-amber-800 border border-amber-300'
                                : idx === 1
                                ? 'bg-slate-200 text-slate-700'
                                : 'bg-slate-100 text-slate-600'
                            }`}
                          >
                            Hipotez #{idx + 1}
                          </span>
                          {idx === 0 && (
                            <span className="px-2 py-0.5 bg-emerald-100 text-emerald-800 rounded text-[11px] font-semibold">
                              ★ En Yüksek Skorlu Dal
                            </span>
                          )}
                        </div>

                        <div className="flex items-center space-x-3 text-xs font-mono">
                          <span className="text-emerald-700 bg-emerald-50 px-2 py-0.5 rounded border border-emerald-200">
                            Norm. Skor: <strong>{hypo.normalized_score.toFixed(3)}</strong>
                          </span>
                          <span className="text-slate-500">
                            Log-Skor: {hypo.score.toFixed(2)}
                          </span>
                          <span className="text-slate-400">
                            {hypo.length} token
                          </span>
                        </div>
                      </div>

                      <p className="text-sm font-medium text-slate-800 leading-relaxed bg-slate-50 p-3 rounded-lg border border-slate-100">
                        {hypo.text}
                      </p>

                      <div className="flex items-center justify-end space-x-2 pt-1">
                        <button
                          type="button"
                          onClick={() => setPrompt(hypo.text)}
                          className="px-3 py-1 bg-slate-100 hover:bg-slate-200 text-slate-700 rounded text-xs font-medium transition"
                        >
                          İstem Olarak Kullan ⇲
                        </button>
                        <button
                          type="button"
                          onClick={() => handleCopyText(hypo.text, `beam-${hypo.hypothesis_id}`)}
                          className="px-3 py-1 bg-indigo-50 hover:bg-indigo-100 text-indigo-700 rounded text-xs font-medium transition flex items-center space-x-1"
                        >
                          {copiedId === `beam-${hypo.hypothesis_id}` ? (
                            <>
                              <Check className="w-3.5 h-3.5 text-emerald-600" />
                              <span>Kopyalandı</span>
                            </>
                          ) : (
                            <>
                              <Copy className="w-3.5 h-3.5" />
                              <span>Kopyala</span>
                            </>
                          )}
                        </button>
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="bg-white rounded-xl p-10 border border-slate-200 text-center shadow-sm">
                    <GitBranch className="w-10 h-10 text-slate-300 mx-auto mb-2" />
                    <p className="text-sm font-bold text-slate-800">Işın Araması Yapılmadı</p>
                    <p className="text-xs text-slate-500 mt-1 max-w-md mx-auto">
                      Beam search algoritması, her adımda en olası K hipotez yolunu paralel dallandırarak greedy (açgözlü) aramaya göre daha tutarlı ve zengin cümleler inşa eder.
                    </p>
                  </div>
                )}
              </div>
            )}

            {/* MODE 3: LOGITS & PROBABILITY DISTRIBUTION INSPECTOR */}
            {mode === 'logits' && (
              <div className="bg-white rounded-xl p-5 border border-slate-200 shadow-sm space-y-5">
                <div className="flex items-center justify-between border-b border-slate-100 pb-3">
                  <div>
                    <h3 className="text-sm font-bold text-slate-900 uppercase tracking-wider flex items-center space-x-2">
                      <BarChart3 className="w-4 h-4 text-cyan-600" />
                      <span>Sonraki Token Olasılık Dağılımı (Next-Token Softmax)</span>
                    </h3>
                    <p className="text-xs text-slate-500 mt-0.5">
                      Modelin "{prompt.slice(-30)}" ardına tahmin ettiği aday tokenlar:
                    </p>
                  </div>
                  {logitsResults && (
                    <span className="text-xs text-cyan-700 bg-cyan-50 px-2.5 py-1 rounded border border-cyan-200 font-mono">
                      T={logitsResults.temperature.toFixed(1)}
                    </span>
                  )}
                </div>

                {logitsResults?.candidates && logitsResults.candidates.length > 0 ? (
                  <div className="space-y-2.5">
                    {logitsResults.candidates.map((cand) => (
                      <div
                        key={cand.rank}
                        className="p-3 rounded-lg border border-slate-100 hover:border-cyan-300 bg-slate-50/50 hover:bg-cyan-50/30 transition space-y-1.5 group"
                      >
                        <div className="flex items-center justify-between text-xs">
                          <div className="flex items-center space-x-2">
                            <span className="w-5 h-5 rounded-full bg-slate-200 text-slate-700 flex items-center justify-center font-mono font-bold text-[11px]">
                              {cand.rank}
                            </span>
                            <span className="font-mono font-bold text-slate-900 bg-white px-2 py-0.5 rounded border border-slate-200 text-sm">
                              {cand.token_text.startsWith(' ') ? `␣${cand.token_text.trim()}` : cand.token_text}
                            </span>
                            <span className="text-[11px] text-slate-400 font-mono">
                              (id: {cand.token_id})
                            </span>
                          </div>

                          <div className="flex items-center space-x-3">
                            <span className="font-mono text-xs text-slate-500">
                              logit: {cand.raw_logit.toFixed(2)}
                            </span>
                            <span className="font-mono font-bold text-cyan-700 text-xs">
                              %{cand.percentage.toFixed(2)}
                            </span>
                            <button
                              type="button"
                              onClick={() => handleAppendTokenAndAdvance(cand)}
                              className="px-2.5 py-1 bg-cyan-600 hover:bg-cyan-700 text-white rounded text-xs font-semibold shadow transition flex items-center space-x-1"
                              title="Bu tokenı seçip metne ekle ve bir sonraki adımı incele"
                            >
                              <span>Seç &amp; İlerle</span>
                              <ChevronRight className="w-3.5 h-3.5" />
                            </button>
                          </div>
                        </div>

                        {/* Visual Progress Bar */}
                        <div className="w-full bg-slate-200 rounded-full h-2 overflow-hidden">
                          <div
                            className="bg-gradient-to-r from-cyan-500 to-indigo-600 h-2 rounded-full transition-all duration-500"
                            style={{ width: `${Math.max(cand.percentage, 2)}%` }}
                          />
                        </div>
                      </div>
                    ))}

                    <div className="p-3 bg-cyan-50/60 rounded-lg border border-cyan-100 text-xs text-cyan-800 flex items-start space-x-2">
                      <Sparkles className="w-4 h-4 text-cyan-600 shrink-0 mt-0.5" />
                      <span>
                        <strong>Adım Adım İnteraktif Decoding:</strong> Bir token seçtiğinizde, model o tokenı cümlenin sonuna ekler ve hemen ardından gelecek yeni adayları hesaplar.
                      </span>
                    </div>
                  </div>
                ) : (
                  <div className="text-center py-10">
                    <BarChart3 className="w-10 h-10 text-slate-300 mx-auto mb-2" />
                    <p className="text-sm font-bold text-slate-800">Dağılım Hesaplanıyor veya İstem Boş</p>
                    <button
                      type="button"
                      onClick={() => handleInspectLogits()}
                      className="mt-3 px-4 py-2 bg-cyan-600 hover:bg-cyan-700 text-white rounded-lg text-xs font-semibold shadow transition"
                    >
                      Olasılıkları Şimdi Hesapla
                    </button>
                  </div>
                )}
              </div>
            )}

            {/* Çıkarım Geçmişi (Tüm Modlar İçin Ortak Görünüm) */}
            <div className="space-y-4 pt-2">
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-bold text-slate-900 uppercase tracking-wider flex items-center space-x-2">
                  <MessageSquare className="w-4 h-4 text-indigo-600" />
                  <span>Üretim Kayıtları &amp; Karşılaştırma ({history.length})</span>
                </h3>
                {history.length > 0 && (
                  <button
                    type="button"
                    onClick={() => setHistory([])}
                    className="text-xs text-slate-400 hover:text-rose-600 transition"
                  >
                    Geçmişi Temizle
                  </button>
                )}
              </div>

              {history.length > 0 ? (
                history.map((item) => (
                  <div
                    key={item.id}
                    className="bg-white rounded-xl p-5 border border-slate-200 shadow-sm space-y-3"
                  >
                    <div className="flex items-center justify-between text-xs text-slate-400 border-b border-slate-100 pb-2">
                      <div className="flex items-center space-x-2">
                        <span className="font-semibold text-slate-700">{item.model_name}</span>
                        <span className="px-2 py-0.5 bg-indigo-50 text-indigo-700 rounded font-mono text-[11px]">
                          {item.mode}
                        </span>
                      </div>

                      <div className="flex items-center space-x-3">
                        <span className="flex items-center space-x-1">
                          <Zap className="w-3.5 h-3.5 text-amber-500" />
                          <span>{item.tokens_generated} token</span>
                        </span>
                        <span className="flex items-center space-x-1">
                          <Clock className="w-3.5 h-3.5 text-slate-400" />
                          <span>{item.time_ms.toFixed(0)} ms</span>
                        </span>
                        <span>{item.timestamp}</span>
                        <button
                          type="button"
                          onClick={() => handleCopyText(item.generated_text, item.id)}
                          className="p-1 hover:text-indigo-600 transition"
                          title="Metni Kopyala"
                        >
                          {copiedId === item.id ? (
                            <Check className="w-3.5 h-3.5 text-emerald-600" />
                          ) : (
                            <Copy className="w-3.5 h-3.5" />
                          )}
                        </button>
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
                <div className="bg-white rounded-xl p-8 border border-slate-200 text-center shadow-sm">
                  <Sparkles className="w-8 h-8 text-slate-300 mx-auto mb-2" />
                  <p className="text-sm font-bold text-slate-700">Henüz Kaydedilmiş Çıktı Yok</p>
                  <p className="text-xs text-slate-400 mt-0.5">
                    Seçtiğiniz mod ile metin ürettiğinizde sonuçlar burada anında listelenecektir.
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
