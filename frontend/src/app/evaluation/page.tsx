'use client';

import React, { useState, useEffect, useMemo } from 'react';
import Link from 'next/link';
import {
  BarChart2,
  Trophy,
  Sliders,
  Play,
  RefreshCw,
  CheckCircle2,
  AlertCircle,
  Info,
  Trash2,
  Search,
  Sparkles,
  Layers,
  ArrowRight,
  Database,
  HelpCircle,
  ExternalLink,
  ChevronRight,
  ChevronDown,
  ChevronUp,
  TrendingUp,
  Activity,
  Cpu,
  Compass,
  Eye,
  BookOpen,
  Check,
} from 'lucide-react';
import {
  api,
  TextMetricInspectResponse,
  BenchmarkResultResponse,
  ModelComparisonResponse,
  AvailableBenchmarksResponse,
  RadarComparisonResponse,
  BenchmarkSampleQuestion,
} from '@/lib/api';

// Preset sample pairs for Tab 1: BLEU & ROUGE N-gram Inspector
const TEXT_PRESETS = [
  {
    title: 'Tam Eşleşme (Exact Match)',
    desc: 'Birebir aynı metin; BLEU & ROUGE maksimum skora (100) ve BP = 1.0 değerine ulaşır.',
    candidate: 'Derin öğrenme modelleri büyük veri kümeleri üzerinde optimize edilir.',
    reference: 'Derin öğrenme modelleri büyük veri kümeleri üzerinde optimize edilir.',
    maxN: 4,
  },
  {
    title: 'Kısmi Eşleşme (Paraphrase & Reorder)',
    desc: 'Kelime dizilimleri farklı, ortak anahtar kelimeler ve unigram/bigram örtüşmeleri içerir.',
    candidate: 'Yapay zeka modelleri metinleri yüksek başarıyla özetleyebilir ve analiz edebilir.',
    reference: 'Modern yapay zeka sistemleri metin özetleme ve anlama görevlerinde yüksek performans gösterir.',
    maxN: 3,
  },
  {
    title: 'Brevity Penalty Cezası (Kısa Çıktı)',
    desc: 'Aday metin referanstan çok kısa olduğunda Brevity Penalty (BP) devreye girerek skoru düşürür.',
    candidate: 'Büyük dil modelleri verimli çalışır.',
    reference: 'Modern büyük dil modelleri binlerce GPU kümesi üzerinde dağıtık olarak oldukça verimli şekilde eğitilir.',
    maxN: 4,
  },
  {
    title: 'Farklı Alan & Düşük Örtüşme',
    desc: 'Anlamsal veya sözcüksel benzerliği olmayan iki farklı metin kıyaslaması.',
    candidate: 'Güneş sistemi içerisindeki gezegenler eliptik yörüngelerde dönerler.',
    reference: 'Yazılım mühendisliğinde mikroservis mimarileri ölçeklenebilirlik sağlar.',
    maxN: 2,
  },
];

// Radar Chart 5-color palette for multi-model comparison
const RADAR_PALETTE = [
  { stroke: '#4f46e5', fill: 'rgba(79, 70, 229, 0.22)', dot: '#4338ca', badge: 'bg-indigo-100 text-indigo-800 border-indigo-200', text: 'text-indigo-600' },
  { stroke: '#059669', fill: 'rgba(5, 150, 105, 0.22)', dot: '#047857', badge: 'bg-emerald-100 text-emerald-800 border-emerald-200', text: 'text-emerald-600' },
  { stroke: '#d97706', fill: 'rgba(217, 119, 6, 0.22)', dot: '#b45309', badge: 'bg-amber-100 text-amber-800 border-amber-200', text: 'text-amber-600' },
  { stroke: '#e11d48', fill: 'rgba(225, 29, 72, 0.22)', dot: '#be123c', badge: 'bg-rose-100 text-rose-800 border-rose-200', text: 'text-rose-600' },
  { stroke: '#8b5cf6', fill: 'rgba(139, 92, 246, 0.22)', dot: '#6d28d9', badge: 'bg-purple-100 text-purple-800 border-purple-200', text: 'text-purple-600' },
];

export default function EvaluationLabPage() {
  const [activeTab, setActiveTab] = useState<'inspector' | 'runner' | 'arena' | 'history'>('inspector');

  // Available Benchmarks & Models metadata
  const [availableBenchmarks, setAvailableBenchmarks] = useState<AvailableBenchmarksResponse['benchmarks']>({});
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [loadingMetadata, setLoadingMetadata] = useState(false);

  // --------------------------------------------------------------------------
  // Tab 1: BLEU & ROUGE N-gram Inspector State
  // --------------------------------------------------------------------------
  const [candidateText, setCandidateText] = useState(TEXT_PRESETS[1].candidate);
  const [referenceText, setReferenceText] = useState(TEXT_PRESETS[1].reference);
  const [maxN, setMaxN] = useState<number>(3);
  const [inspectLoading, setInspectLoading] = useState<boolean>(false);
  const [inspectData, setInspectData] = useState<TextMetricInspectResponse | null>(null);
  const [inspectError, setInspectError] = useState<string | null>(null);

  // --------------------------------------------------------------------------
  // Tab 2: Model Benchmark Runner State
  // --------------------------------------------------------------------------
  const [runModelName, setRunModelName] = useState<string>('nano-gpt-v1');
  const [runBenchmarkName, setRunBenchmarkName] = useState<string>('perplexity');
  const [runDatasetPath, setRunDatasetPath] = useState<string>('');
  const [runMaxSamples, setRunMaxSamples] = useState<number>(50);
  const [runBatchSize, setRunBatchSize] = useState<number>(8);
  const [runLoading, setRunLoading] = useState<boolean>(false);
  const [runResult, setRunResult] = useState<BenchmarkResultResponse | null>(null);
  const [runError, setRunError] = useState<string | null>(null);

  // --------------------------------------------------------------------------
  // Tab 2 extension: Sample Questions & CoT inspector
  // --------------------------------------------------------------------------
  const [sampleQuestions, setSampleQuestions] = useState<BenchmarkSampleQuestion[]>([]);
  const [loadingQuestions, setLoadingQuestions] = useState<boolean>(false);
  const [showQuestionPool, setShowQuestionPool] = useState<boolean>(false);

  // --------------------------------------------------------------------------
  // Tab 3: Model Comparison Arena & Radar State
  // --------------------------------------------------------------------------
  const [arenaSubMode, setArenaSubMode] = useState<'radar' | 'head_to_head'>('radar');
  const [radarSelectedModels, setRadarSelectedModels] = useState<string[]>([]);
  const [radarLoading, setRadarLoading] = useState<boolean>(false);
  const [radarData, setRadarData] = useState<RadarComparisonResponse | null>(null);
  const [radarError, setRadarError] = useState<string | null>(null);
  const [visibleRadarModels, setVisibleRadarModels] = useState<Record<string, boolean>>({});

  const [compareModelA, setCompareModelA] = useState<string>('nano-gpt-v1');
  const [compareModelB, setCompareModelB] = useState<string>('nano-gpt-v2');
  const [compareBenchmark, setCompareBenchmark] = useState<string>('perplexity');
  const [compareLoading, setCompareLoading] = useState<boolean>(false);
  const [compareResult, setCompareResult] = useState<ModelComparisonResponse | null>(null);
  const [compareError, setCompareError] = useState<string | null>(null);

  // --------------------------------------------------------------------------
  // Tab 4: Leaderboard & History State
  // --------------------------------------------------------------------------
  const [historyResults, setHistoryResults] = useState<BenchmarkResultResponse[]>([]);
  const [historyLoading, setHistoryLoading] = useState<boolean>(false);
  const [historyFilterModel, setHistoryFilterModel] = useState<string>('');
  const [historyFilterBenchmark, setHistoryFilterBenchmark] = useState<string>('');
  const [selectedHistoryItem, setSelectedHistoryItem] = useState<BenchmarkResultResponse | null>(null);
  const [deleteConfirmId, setDeleteConfirmId] = useState<string | null>(null);

  // Load benchmarks and registered models on mount
  useEffect(() => {
    async function loadMeta() {
      setLoadingMetadata(true);
      try {
        const benchResp = await api.evaluation.getBenchmarks();
        if (benchResp?.benchmarks) {
          setAvailableBenchmarks(benchResp.benchmarks);
        }
      } catch (err) {
        console.warn('Could not load benchmarks:', err);
      }

      const urlModel = typeof window !== 'undefined' ? new URLSearchParams(window.location.search).get('model') : null;
      try {
        const modelsResp = await api.models.list();
        if (modelsResp && modelsResp.length > 0) {
          const names = modelsResp.map((m) => m.model_name);
          const uniqueNames = Array.from(new Set(names));
          if (urlModel && !uniqueNames.includes(urlModel)) {
            uniqueNames.unshift(urlModel);
          }
          setAvailableModels(uniqueNames);
          const initialRadar = uniqueNames.slice(0, Math.min(3, uniqueNames.length));
          setRadarSelectedModels(initialRadar);
          if (urlModel) {
            setRunModelName(urlModel);
            setCompareModelA(urlModel);
            setCompareModelB(uniqueNames.find((n) => n !== urlModel) || urlModel);
            setActiveTab('runner');
          } else if (uniqueNames.length >= 1) {
            setRunModelName(uniqueNames[0]);
            setCompareModelA(uniqueNames[0]);
            setCompareModelB(uniqueNames[1] || uniqueNames[0]);
          }
        } else {
          const defaultList = urlModel
            ? [urlModel, 'nano-gpt-v1', 'turkish-gpt-small']
            : ['nano-gpt-v1', 'turkish-gpt-small', 'transformer-base'];
          setAvailableModels(defaultList);
          setRadarSelectedModels(defaultList.slice(0, 3));
          if (urlModel) {
            setRunModelName(urlModel);
            setCompareModelA(urlModel);
            setActiveTab('runner');
          }
        }
      } catch {
        const defaultList = urlModel
          ? [urlModel, 'nano-gpt-v1', 'turkish-gpt-small']
          : ['nano-gpt-v1', 'turkish-gpt-small', 'transformer-base'];
        setAvailableModels(defaultList);
        setRadarSelectedModels(defaultList.slice(0, 3));
        if (urlModel) {
          setRunModelName(urlModel);
          setCompareModelA(urlModel);
          setActiveTab('runner');
        }
      } finally {
        setLoadingMetadata(false);
      }
    }
    loadMeta();
  }, []);

  // Run initial inspectText on first render
  useEffect(() => {
    handleInspect();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Fetch History when switching to history tab
  useEffect(() => {
    if (activeTab === 'history') {
      fetchHistory();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeTab, historyFilterModel, historyFilterBenchmark]);

  // Auto-run Radar when opening Arena tab if not yet loaded
  useEffect(() => {
    if (activeTab === 'arena' && arenaSubMode === 'radar' && !radarData && !radarLoading && radarSelectedModels.length > 0) {
      handleRadarComparison(radarSelectedModels);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeTab, arenaSubMode, radarSelectedModels]);

  // Tab 1 Handler
  const handleInspect = async () => {
    if (!candidateText.trim() || !referenceText.trim()) {
      setInspectError('Lütfen hem aday hem de referans metin giriniz.');
      return;
    }
    setInspectLoading(true);
    setInspectError(null);
    try {
      const data = await api.evaluation.inspectText({
        candidate: candidateText,
        reference: referenceText,
        max_n: maxN,
      });
      setInspectData(data);
    } catch (err: any) {
      setInspectError(err?.response?.data?.detail || 'Metin analizi sırasında hata oluştu.');
    } finally {
      setInspectLoading(false);
    }
  };

  // Tab 2 Handler: Run Benchmark
  const handleRunBenchmark = async () => {
    setRunLoading(true);
    setRunError(null);
    setRunResult(null);
    try {
      const res = await api.evaluation.runBenchmark({
        model_name: runModelName,
        benchmark_name: runBenchmarkName,
        dataset_path: runDatasetPath.trim() ? runDatasetPath.trim() : undefined,
        max_samples: runMaxSamples,
        batch_size: runBatchSize,
      });
      setRunResult(res);
    } catch (err: any) {
      setRunError(err?.response?.data?.detail || 'Benchmark çalıştırılırken hata oluştu.');
    } finally {
      setRunLoading(false);
    }
  };

  // Tab 2: Fetch Sample Questions
  const handleFetchSampleQuestions = async (benchName?: string) => {
    const targetBench = benchName || runBenchmarkName;
    setLoadingQuestions(true);
    try {
      const q = await api.evaluation.getSampleQuestions(targetBench);
      setSampleQuestions(q);
      setShowQuestionPool(true);
    } catch (err) {
      console.warn('Could not fetch sample questions', err);
    } finally {
      setLoadingQuestions(false);
    }
  };

  // Tab 3: Radar Comparison Handler
  const handleRadarComparison = async (modelsToCompare?: string[]) => {
    const list = modelsToCompare || radarSelectedModels;
    if (!list || list.length < 1) {
      setRadarError('Lütfen radar karşılaştırması için en az bir model seçiniz.');
      return;
    }
    setRadarLoading(true);
    setRadarError(null);
    try {
      const res = await api.evaluation.getRadarComparison({
        model_names: list,
      });
      setRadarData(res);
      const vis: Record<string, boolean> = {};
      res.models.forEach((m) => {
        vis[m.model_name] = true;
      });
      setVisibleRadarModels(vis);
    } catch (err: any) {
      setRadarError(err?.response?.data?.detail || 'Radar analizi sırasında hata oluştu.');
    } finally {
      setRadarLoading(false);
    }
  };

  const toggleRadarModel = (modelName: string) => {
    setRadarSelectedModels((prev) => {
      if (prev.includes(modelName)) {
        if (prev.length <= 1) return prev;
        return prev.filter((m) => m !== modelName);
      } else {
        if (prev.length >= 5) return prev;
        return [...prev, modelName];
      }
    });
  };

  const toggleRadarVisibility = (modelName: string) => {
    setVisibleRadarModels((prev) => ({
      ...prev,
      [modelName]: !prev[modelName],
    }));
  };

  // Tab 3 Handler: Compare Models
  const handleCompare = async () => {
    if (compareModelA === compareModelB) {
      setCompareError('Karşılaştırmak için lütfen iki farklı model seçiniz.');
      return;
    }
    setCompareLoading(true);
    setCompareError(null);
    setCompareResult(null);
    try {
      const res = await api.evaluation.compareModels({
        model_names: [compareModelA, compareModelB],
        benchmark_name: compareBenchmark,
      });
      setCompareResult(res);
    } catch (err: any) {
      setCompareError(err?.response?.data?.detail || 'Model karşılaştırması sırasında hata oluştu.');
    } finally {
      setCompareLoading(false);
    }
  };

  // Tab 4 Handler: Fetch & Delete
  const fetchHistory = async () => {
    setHistoryLoading(true);
    try {
      const res = await api.evaluation.getResults({
        model_name: historyFilterModel.trim() ? historyFilterModel.trim() : undefined,
        benchmark_name: historyFilterBenchmark.trim() ? historyFilterBenchmark.trim() : undefined,
        limit: 50,
      });
      setHistoryResults(res || []);
    } catch (err) {
      console.error('History fetch error:', err);
    } finally {
      setHistoryLoading(false);
    }
  };

  const handleDeleteHistory = async (id: string) => {
    try {
      await api.evaluation.deleteResult(id);
      setHistoryResults((prev) => prev.filter((item) => item.benchmark_id !== id));
      if (selectedHistoryItem?.benchmark_id === id) {
        setSelectedHistoryItem(null);
      }
      setDeleteConfirmId(null);
    } catch (err) {
      console.error('Delete error:', err);
    }
  };

  // Matched words set for fast highlighting in Tab 1
  const matchedTokensSet = useMemo(() => {
    if (!inspectData?.matched_unigrams) return new Set<string>();
    return new Set(inspectData.matched_unigrams.map((w) => w.toLowerCase()));
  }, [inspectData]);

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 pb-16">
      {/* Top Banner & Header */}
      <div className="bg-gradient-to-r from-slate-900 via-indigo-950 to-slate-900 text-white border-b border-indigo-900/50 shadow-md">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
            <div>
              <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-500/20 text-indigo-300 text-xs font-semibold tracking-wide uppercase mb-3 border border-indigo-400/30">
                <Sparkles className="w-3.5 h-3.5" />
                Bölüm 31 & 32 • LLM Değerlendirme & Benchmark
              </div>
              <h1 className="text-3xl font-extrabold tracking-tight sm:text-4xl text-white">
                Model Evaluation & Benchmark Lab
              </h1>
              <p className="mt-2 text-base text-slate-300 max-w-3xl leading-relaxed">
                Dil modellerinin başarımını ölçün, BLEU & ROUGE n-gram örtüşmelerini görsel olarak inceleyin,
                Perplexity ve Doğruluk metrikleriyle modelleri kıyaslayın.
              </p>
            </div>

            <div className="flex items-center gap-3">
              <Link
                href="/transformer-lab"
                className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-slate-800/80 hover:bg-slate-700/80 text-slate-200 text-sm font-medium transition border border-slate-700/60 shadow-sm"
              >
                <Layers className="w-4 h-4 text-indigo-400" />
                Transformer Lab
              </Link>
              <Link
                href="/models"
                className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white text-sm font-semibold transition shadow-md shadow-indigo-600/30"
              >
                <Cpu className="w-4 h-4" />
                Modeller
              </Link>
            </div>
          </div>

          {/* Navigation Tabs */}
          <div className="flex space-x-1 mt-8 overflow-x-auto border-b border-slate-800 pb-1 scrollbar-none">
            <button
              id="tab-inspector"
              onClick={() => setActiveTab('inspector')}
              className={`flex items-center gap-2 px-4 py-2.5 text-sm font-medium rounded-t-xl transition-colors whitespace-nowrap ${
                activeTab === 'inspector'
                  ? 'bg-slate-50 text-indigo-900 shadow-sm font-semibold'
                  : 'text-slate-300 hover:text-white hover:bg-white/5'
              }`}
            >
              <Search className="w-4 h-4 text-indigo-600" />
              BLEU & ROUGE N-Gram Inspector
            </button>
            <button
              id="tab-runner"
              onClick={() => setActiveTab('runner')}
              className={`flex items-center gap-2 px-4 py-2.5 text-sm font-medium rounded-t-xl transition-colors whitespace-nowrap ${
                activeTab === 'runner'
                  ? 'bg-slate-50 text-indigo-900 shadow-sm font-semibold'
                  : 'text-slate-300 hover:text-white hover:bg-white/5'
              }`}
            >
              <Play className="w-4 h-4 text-emerald-600" />
              Benchmark Runner
            </button>
            <button
              id="tab-arena"
              onClick={() => setActiveTab('arena')}
              className={`flex items-center gap-2 px-4 py-2.5 text-sm font-medium rounded-t-xl transition-colors whitespace-nowrap ${
                activeTab === 'arena'
                  ? 'bg-slate-50 text-indigo-900 shadow-sm font-semibold'
                  : 'text-slate-300 hover:text-white hover:bg-white/5'
              }`}
            >
              <Trophy className="w-4 h-4 text-amber-500" />
              Model Comparison Arena
            </button>
            <button
              id="tab-history"
              onClick={() => setActiveTab('history')}
              className={`flex items-center gap-2 px-4 py-2.5 text-sm font-medium rounded-t-xl transition-colors whitespace-nowrap ${
                activeTab === 'history'
                  ? 'bg-slate-50 text-indigo-900 shadow-sm font-semibold'
                  : 'text-slate-300 hover:text-white hover:bg-white/5'
              }`}
            >
              <BarChart2 className="w-4 h-4 text-blue-500" />
              Leaderboard & Geçmiş
            </button>
          </div>
        </div>
      </div>

      {/* Main Content Area */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 mt-6">
        {/* ==================================================================== */}
        {/* TAB 1: BLEU & ROUGE N-GRAM INSPECTOR                                  */}
        {/* ==================================================================== */}
        {activeTab === 'inspector' && (
          <div className="space-y-6">
            {/* Presets Bar */}
            <div className="bg-white p-4 rounded-2xl shadow-sm border border-slate-200">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-semibold uppercase tracking-wider text-slate-500 flex items-center gap-1.5">
                  <Sparkles className="w-3.5 h-3.5 text-indigo-500" />
                  Hazır Örnekler (Presets)
                </span>
                <span className="text-xs text-slate-400">Örnek bir senaryo seçip anında n-gram analizi yapın</span>
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-2.5">
                {TEXT_PRESETS.map((preset, idx) => (
                  <button
                    key={idx}
                    onClick={() => {
                      setCandidateText(preset.candidate);
                      setReferenceText(preset.reference);
                      setMaxN(preset.maxN);
                    }}
                    className="text-left p-3 rounded-xl border border-slate-200 hover:border-indigo-400 hover:bg-indigo-50/40 transition group"
                  >
                    <div className="font-semibold text-xs text-slate-800 group-hover:text-indigo-700">
                      {preset.title}
                    </div>
                    <div className="text-[11px] text-slate-500 line-clamp-2 mt-1 leading-snug">
                      {preset.desc}
                    </div>
                  </button>
                ))}
              </div>
            </div>

            {/* Inputs: Candidate vs Reference */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Candidate Box */}
              <div className="bg-white p-5 rounded-2xl shadow-sm border border-slate-200 flex flex-col">
                <div className="flex items-center justify-between mb-2">
                  <label htmlFor="candidate-input" className="text-sm font-semibold text-slate-800 flex items-center gap-2">
                    <span className="w-2.5 h-2.5 rounded-full bg-blue-500"></span>
                    Model Çıktısı (Candidate Text)
                  </label>
                  <span className="text-xs text-slate-400 font-mono">
                    {candidateText.split(/\s+/).filter(Boolean).length} kelime
                  </span>
                </div>
                <textarea
                  id="candidate-input"
                  rows={4}
                  value={candidateText}
                  onChange={(e) => setCandidateText(e.target.value)}
                  placeholder="Değerlendirilecek model çıktısı metnini girin..."
                  className="w-full p-3 rounded-xl border border-slate-200 focus:outline-none focus:ring-2 focus:ring-blue-500/30 focus:border-blue-500 text-sm leading-relaxed transition bg-slate-50/50"
                />
              </div>

              {/* Reference Box */}
              <div className="bg-white p-5 rounded-2xl shadow-sm border border-slate-200 flex flex-col">
                <div className="flex items-center justify-between mb-2">
                  <label htmlFor="reference-input" className="text-sm font-semibold text-slate-800 flex items-center gap-2">
                    <span className="w-2.5 h-2.5 rounded-full bg-emerald-500"></span>
                    Hedef Referans (Ground Truth)
                  </label>
                  <span className="text-xs text-slate-400 font-mono">
                    {referenceText.split(/\s+/).filter(Boolean).length} kelime
                  </span>
                </div>
                <textarea
                  id="reference-input"
                  rows={4}
                  value={referenceText}
                  onChange={(e) => setReferenceText(e.target.value)}
                  placeholder="Karşılaştırılacak gerçek insan/altın referans metni girin..."
                  className="w-full p-3 rounded-xl border border-slate-200 focus:outline-none focus:ring-2 focus:ring-emerald-500/30 focus:border-emerald-500 text-sm leading-relaxed transition bg-slate-50/50"
                />
              </div>
            </div>

            {/* Action Bar */}
            <div className="bg-white p-4 rounded-2xl shadow-sm border border-slate-200 flex flex-col sm:flex-row items-center justify-between gap-4">
              <div className="flex items-center gap-4 w-full sm:w-auto">
                <div className="flex items-center gap-2">
                  <Sliders className="w-4 h-4 text-slate-500" />
                  <span className="text-xs font-semibold text-slate-700">Maks N-Gram:</span>
                </div>
                <div className="flex rounded-lg border border-slate-200 overflow-hidden bg-slate-50">
                  {[1, 2, 3, 4].map((n) => (
                    <button
                      key={n}
                      onClick={() => setMaxN(n)}
                      className={`px-3 py-1.5 text-xs font-semibold transition ${
                        maxN === n ? 'bg-indigo-600 text-white' : 'text-slate-600 hover:bg-slate-200/60'
                      }`}
                    >
                      {n}-gram
                    </button>
                  ))}
                </div>
              </div>

              <button
                id="btn-inspect-analyze"
                onClick={handleInspect}
                disabled={inspectLoading}
                className="w-full sm:w-auto inline-flex items-center justify-center gap-2 px-6 py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white font-semibold text-sm transition shadow-md shadow-indigo-600/20 disabled:opacity-50"
              >
                {inspectLoading ? (
                  <>
                    <RefreshCw className="w-4 h-4 animate-spin" />
                    Analiz Ediliyor...
                  </>
                ) : (
                  <>
                    <Activity className="w-4 h-4" />
                    Metinleri Analiz Et
                  </>
                )}
              </button>
            </div>

            {/* Inspection Errors */}
            {inspectError && (
              <div className="p-4 rounded-xl bg-red-50 border border-red-200 text-red-700 flex items-center gap-3 text-sm">
                <AlertCircle className="w-5 h-5 flex-shrink-0" />
                <span>{inspectError}</span>
              </div>
            )}

            {/* Results Grid */}
            {inspectData && (
              <div className="space-y-6">
                {/* Metric Summary Cards */}
                <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                  {/* Brevity Penalty */}
                  <div className="bg-white p-5 rounded-2xl shadow-sm border border-slate-200 relative overflow-hidden">
                    <div className="flex items-center justify-between text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">
                      <span>Brevity Penalty (BP)</span>
                      <span className="text-slate-400">c / r = {(inspectData.candidate_len / Math.max(1, inspectData.reference_len)).toFixed(2)}</span>
                    </div>
                    <div className="text-2xl font-bold text-slate-900">
                      {inspectData.brevity_penalty.toFixed(3)}
                    </div>
                    <div className="w-full bg-slate-100 rounded-full h-2 mt-3 overflow-hidden">
                      <div
                        className={`h-full rounded-full transition-all duration-500 ${
                          inspectData.brevity_penalty >= 0.9
                            ? 'bg-emerald-500'
                            : inspectData.brevity_penalty >= 0.7
                            ? 'bg-amber-500'
                            : 'bg-red-500'
                        }`}
                        style={{ width: `${inspectData.brevity_penalty * 100}%` }}
                      />
                    </div>
                    <p className="text-[11px] text-slate-400 mt-2">
                      {inspectData.brevity_penalty === 1.0
                        ? 'Ceza yok (uzunluk yeterli)'
                        : 'Kısa çıktı sebebiyle orantısal ceza uygulandı'}
                    </p>
                  </div>

                  {/* Primary BLEU Score */}
                  <div className="bg-white p-5 rounded-2xl shadow-sm border border-slate-200">
                    <div className="flex items-center justify-between text-xs font-semibold text-indigo-600 uppercase tracking-wider mb-2">
                      <span>Kümülatif BLEU</span>
                      <span className="text-xs bg-indigo-50 text-indigo-700 px-2 py-0.5 rounded font-mono">
                        max {maxN}-gram
                      </span>
                    </div>
                    <div className="text-2xl font-bold text-indigo-950">
                      {(inspectData.bleu[`bleu-${maxN}`] ?? inspectData.bleu['bleu-1'] ?? 0).toFixed(1)}
                      <span className="text-sm font-normal text-slate-500 ml-1">/ 100</span>
                    </div>
                    <div className="w-full bg-slate-100 rounded-full h-2 mt-3 overflow-hidden">
                      <div
                        className="h-full bg-indigo-600 rounded-full transition-all duration-500"
                        style={{
                          width: `${Math.min(
                            100,
                            inspectData.bleu[`bleu-${maxN}`] ?? inspectData.bleu['bleu-1'] ?? 0
                          )}%`,
                        }}
                      />
                    </div>
                    <p className="text-[11px] text-slate-400 mt-2">Hassasiyet (Precision) bazlı örtüşme</p>
                  </div>

                  {/* ROUGE-1 & ROUGE-2 */}
                  <div className="bg-white p-5 rounded-2xl shadow-sm border border-slate-200">
                    <div className="text-xs font-semibold text-emerald-600 uppercase tracking-wider mb-2">
                      ROUGE-1 & ROUGE-2
                    </div>
                    <div className="flex items-baseline gap-3">
                      <div>
                        <span className="text-xl font-bold text-emerald-950">
                          {(inspectData.rouge['rouge-1'] ?? 0).toFixed(1)}
                        </span>
                        <span className="text-[11px] text-slate-400 ml-1">R-1</span>
                      </div>
                      <span className="text-slate-300">|</span>
                      <div>
                        <span className="text-xl font-bold text-emerald-950">
                          {(inspectData.rouge['rouge-2'] ?? 0).toFixed(1)}
                        </span>
                        <span className="text-[11px] text-slate-400 ml-1">R-2</span>
                      </div>
                    </div>
                    <div className="w-full bg-slate-100 rounded-full h-2 mt-3 overflow-hidden">
                      <div
                        className="h-full bg-emerald-500 rounded-full transition-all duration-500"
                        style={{ width: `${Math.min(100, inspectData.rouge['rouge-1'] ?? 0)}%` }}
                      />
                    </div>
                    <p className="text-[11px] text-slate-400 mt-2">Geri çağırma (Recall & F1) örtüşmesi</p>
                  </div>

                  {/* ROUGE-L */}
                  <div className="bg-white p-5 rounded-2xl shadow-sm border border-slate-200">
                    <div className="text-xs font-semibold text-purple-600 uppercase tracking-wider mb-2">
                      ROUGE-L (LCS F1)
                    </div>
                    <div className="text-2xl font-bold text-purple-950">
                      {(inspectData.rouge['rouge-l'] ?? 0).toFixed(1)}
                      <span className="text-sm font-normal text-slate-500 ml-1">/ 100</span>
                    </div>
                    <div className="w-full bg-slate-100 rounded-full h-2 mt-3 overflow-hidden">
                      <div
                        className="h-full bg-purple-600 rounded-full transition-all duration-500"
                        style={{ width: `${Math.min(100, inspectData.rouge['rouge-l'] ?? 0)}%` }}
                      />
                    </div>
                    <p className="text-[11px] text-slate-400 mt-2">En uzun ortak alt dizi (LCS) F1 skoru</p>
                  </div>
                </div>

                {/* Token Highlight Comparison View */}
                <div className="bg-white p-5 rounded-2xl shadow-sm border border-slate-200">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
                      <Sparkles className="w-4 h-4 text-indigo-500" />
                      Sözcük (Token) Eşleşme Vurgulayıcı
                    </h3>
                    <div className="flex items-center gap-3 text-xs">
                      <span className="inline-flex items-center gap-1.5 text-slate-600">
                        <span className="w-3 h-3 rounded bg-emerald-100 border border-emerald-400"></span>
                        Örtüşen Sözcükler
                      </span>
                      <span className="inline-flex items-center gap-1.5 text-slate-600">
                        <span className="w-3 h-3 rounded bg-slate-100 border border-slate-300"></span>
                        Eşleşmeyen
                      </span>
                    </div>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {/* Candidate Tokens */}
                    <div className="p-4 rounded-xl bg-slate-50 border border-slate-200">
                      <div className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-2 flex items-center justify-between">
                        <span>Aday Metin Tokenleri ({inspectData.candidate_len})</span>
                      </div>
                      <div className="flex flex-wrap gap-1.5">
                        {inspectData.candidate_tokens.map((token, idx) => {
                          const isMatch = matchedTokensSet.has(token.toLowerCase());
                          return (
                            <span
                              key={idx}
                              className={`px-2 py-1 rounded text-xs font-mono transition ${
                                isMatch
                                  ? 'bg-emerald-100 text-emerald-800 border border-emerald-300 font-semibold shadow-2xs'
                                  : 'bg-white text-slate-600 border border-slate-200'
                              }`}
                            >
                              {token}
                            </span>
                          );
                        })}
                      </div>
                    </div>

                    {/* Reference Tokens */}
                    <div className="p-4 rounded-xl bg-slate-50 border border-slate-200">
                      <div className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-2 flex items-center justify-between">
                        <span>Referans Tokenleri ({inspectData.reference_len})</span>
                      </div>
                      <div className="flex flex-wrap gap-1.5">
                        {inspectData.reference_tokens.map((token, idx) => {
                          const isMatch = matchedTokensSet.has(token.toLowerCase());
                          return (
                            <span
                              key={idx}
                              className={`px-2 py-1 rounded text-xs font-mono transition ${
                                isMatch
                                  ? 'bg-emerald-100 text-emerald-800 border border-emerald-300 font-semibold shadow-2xs'
                                  : 'bg-white text-slate-600 border border-slate-200'
                              }`}
                            >
                              {token}
                            </span>
                          );
                        })}
                      </div>
                    </div>
                  </div>
                </div>

                {/* N-Gram Breakdown Cards */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  {/* Unigrams */}
                  <div className="bg-white p-4 rounded-2xl shadow-sm border border-slate-200">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-xs font-bold text-slate-700 uppercase">
                        Örtüşen 1-Gramlar ({inspectData.matched_unigrams.length})
                      </span>
                    </div>
                    {inspectData.matched_unigrams.length > 0 ? (
                      <div className="flex flex-wrap gap-1.5 max-h-36 overflow-y-auto">
                        {inspectData.matched_unigrams.map((ug, i) => (
                          <span
                            key={i}
                            className="px-2 py-0.5 rounded-md bg-indigo-50 text-indigo-700 border border-indigo-200 text-xs font-mono"
                          >
                            {ug}
                          </span>
                        ))}
                      </div>
                    ) : (
                      <span className="text-xs text-slate-400 italic">Eşleşen unigram bulunamadı</span>
                    )}
                  </div>

                  {/* Bigrams */}
                  <div className="bg-white p-4 rounded-2xl shadow-sm border border-slate-200">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-xs font-bold text-slate-700 uppercase">
                        Örtüşen 2-Gramlar ({inspectData.matched_bigrams.length})
                      </span>
                    </div>
                    {inspectData.matched_bigrams.length > 0 ? (
                      <div className="flex flex-wrap gap-1.5 max-h-36 overflow-y-auto">
                        {inspectData.matched_bigrams.map((bg, i) => (
                          <span
                            key={i}
                            className="px-2 py-0.5 rounded-md bg-blue-50 text-blue-700 border border-blue-200 text-xs font-mono"
                          >
                            {bg}
                          </span>
                        ))}
                      </div>
                    ) : (
                      <span className="text-xs text-slate-400 italic">Eşleşen 2-gram bulunamadı</span>
                    )}
                  </div>

                  {/* Trigrams */}
                  <div className="bg-white p-4 rounded-2xl shadow-sm border border-slate-200">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-xs font-bold text-slate-700 uppercase">
                        Örtüşen 3-Gramlar ({inspectData.matched_trigrams.length})
                      </span>
                    </div>
                    {inspectData.matched_trigrams.length > 0 ? (
                      <div className="flex flex-wrap gap-1.5 max-h-36 overflow-y-auto">
                        {inspectData.matched_trigrams.map((tg, i) => (
                          <span
                            key={i}
                            className="px-2 py-0.5 rounded-md bg-purple-50 text-purple-700 border border-purple-200 text-xs font-mono"
                          >
                            {tg}
                          </span>
                        ))}
                      </div>
                    ) : (
                      <span className="text-xs text-slate-400 italic">Eşleşen 3-gram bulunamadı</span>
                    )}
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* ==================================================================== */}
        {/* TAB 2: MODEL BENCHMARK RUNNER                                         */}
        {/* ==================================================================== */}
        {activeTab === 'runner' && (
          <div className="space-y-6">
            <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200">
              <div className="border-b border-slate-100 pb-4 mb-5">
                <h2 className="text-lg font-bold text-slate-900 flex items-center gap-2">
                  <Play className="w-5 h-5 text-emerald-600" />
                  Otomatik Benchmark Testi Çalıştır
                </h2>
                <p className="text-xs text-slate-500 mt-1">
                  Kayıtlı modelinizi seçin ve test verisi üzerinde Perplexity, BLEU, ROUGE veya Accuracy metriğini ölçerek veritabanına kalıcı olarak kaydedin.
                </p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {/* Model Selector */}
                <div>
                  <label htmlFor="select-model" className="block text-xs font-semibold text-slate-700 uppercase tracking-wider mb-2">
                    Test Edilecek Model
                  </label>
                  <select
                    id="select-model"
                    value={runModelName}
                    onChange={(e) => setRunModelName(e.target.value)}
                    className="w-full p-2.5 rounded-xl border border-slate-200 bg-slate-50 text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500/30 font-medium"
                  >
                    {availableModels.map((m, i) => (
                      <option key={i} value={m}>
                        {m}
                      </option>
                    ))}
                  </select>
                </div>

                {/* Benchmark Selector */}
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <label htmlFor="select-benchmark" className="text-xs font-semibold text-slate-700 uppercase tracking-wider">
                      Benchmark Türü
                    </label>
                    {(runBenchmarkName === 'gsm8k_cot' || runBenchmarkName === 'turkish_knowledge') && (
                      <button
                        type="button"
                        onClick={() => {
                          if (!showQuestionPool) {
                            handleFetchSampleQuestions(runBenchmarkName);
                          } else {
                            setShowQuestionPool(false);
                          }
                        }}
                        className="text-[11px] font-semibold text-emerald-700 hover:text-emerald-900 flex items-center gap-1 transition"
                      >
                        <Eye className="w-3 h-3" />
                        <span>{showQuestionPool ? 'Gizle' : 'Soruları Gör'}</span>
                      </button>
                    )}
                  </div>
                  <select
                    id="select-benchmark"
                    value={runBenchmarkName}
                    onChange={(e) => {
                      const newB = e.target.value;
                      setRunBenchmarkName(newB);
                      if (showQuestionPool) {
                        handleFetchSampleQuestions(newB);
                      }
                    }}
                    className="w-full p-2.5 rounded-xl border border-slate-200 bg-slate-50 text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500/30 font-medium"
                  >
                    <option value="perplexity">Perplexity (Belirsizlik / Şaşkınlık)</option>
                    <option value="bleu">BLEU Score (Çeviri / Üretim Kalitesi)</option>
                    <option value="rouge">ROUGE Score (Özetleme Kalitesi)</option>
                    <option value="gsm8k_cot">📐 Çok Adımlı Matematik (GSM8K CoT)</option>
                    <option value="turkish_knowledge">🇹🇷 Türkçe Bilgi &amp; Doğruluk Testi</option>
                    <option value="accuracy">Accuracy (Sınıflandırma Doğruluğu)</option>
                  </select>
                </div>

                {/* Dataset Path */}
                <div>
                  <label htmlFor="input-dataset" className="block text-xs font-semibold text-slate-700 uppercase tracking-wider mb-2">
                    Dataset Yolu (Opsiyonel)
                  </label>
                  <input
                    id="input-dataset"
                    type="text"
                    value={runDatasetPath}
                    onChange={(e) => setRunDatasetPath(e.target.value)}
                    placeholder="datasets/test_data.jsonl"
                    className="w-full p-2.5 rounded-xl border border-slate-200 bg-slate-50 text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500/30"
                  />
                </div>

                {/* Max Samples Slider */}
                <div>
                  <div className="flex justify-between items-center mb-2">
                    <label htmlFor="slider-samples" className="text-xs font-semibold text-slate-700 uppercase tracking-wider">
                      Maksimum Örnek Sayısı
                    </label>
                    <span className="text-xs font-mono font-bold text-emerald-700">{runMaxSamples}</span>
                  </div>
                  <input
                    id="slider-samples"
                    type="range"
                    min="10"
                    max="200"
                    step="10"
                    value={runMaxSamples}
                    onChange={(e) => setRunMaxSamples(Number(e.target.value))}
                    className="w-full accent-emerald-600"
                  />
                </div>

                {/* Batch Size Slider */}
                <div>
                  <div className="flex justify-between items-center mb-2">
                    <label htmlFor="slider-batch" className="text-xs font-semibold text-slate-700 uppercase tracking-wider">
                      Batch Size
                    </label>
                    <span className="text-xs font-mono font-bold text-emerald-700">{runBatchSize}</span>
                  </div>
                  <input
                    id="slider-batch"
                    type="range"
                    min="2"
                    max="32"
                    step="2"
                    value={runBatchSize}
                    onChange={(e) => setRunBatchSize(Number(e.target.value))}
                    className="w-full accent-emerald-600"
                  />
                </div>

                {/* Run Button */}
                <div className="flex items-end">
                  <button
                    id="btn-run-benchmark"
                    onClick={handleRunBenchmark}
                    disabled={runLoading}
                    className="w-full inline-flex items-center justify-center gap-2 p-2.5 rounded-xl bg-emerald-600 hover:bg-emerald-500 text-white font-semibold text-sm transition shadow-md shadow-emerald-600/20 disabled:opacity-50"
                  >
                    {runLoading ? (
                      <>
                        <RefreshCw className="w-4 h-4 animate-spin" />
                        Değerlendiriliyor...
                      </>
                    ) : (
                      <>
                        <Play className="w-4 h-4" />
                        Benchmark Başlat
                      </>
                    )}
                  </button>
                </div>
              </div>

              {/* Soru Havuzu Önizleme Paneli */}
              {showQuestionPool && (
                <div className="mt-5 pt-4 border-t border-slate-100 space-y-3">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <BookOpen className="w-4 h-4 text-emerald-600" />
                      <span className="text-xs font-bold text-slate-800 uppercase tracking-wider">
                        {runBenchmarkName === 'gsm8k_cot'
                          ? 'GSM8K Chain-of-Thought Soru Havuzu'
                          : 'Türkçe Olgusal Bilgi Soru Havuzu'}{' '}
                        ({sampleQuestions.length} Soru)
                      </span>
                    </div>
                    <button
                      type="button"
                      onClick={() => setShowQuestionPool(false)}
                      className="text-xs text-slate-400 hover:text-slate-600 transition"
                    >
                      Kapat ✕
                    </button>
                  </div>
                  {loadingQuestions ? (
                    <div className="py-6 text-center text-xs text-slate-500 flex items-center justify-center gap-2">
                      <RefreshCw className="w-4 h-4 animate-spin text-emerald-600" />
                      <span>Sorular yükleniyor...</span>
                    </div>
                  ) : (
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3 max-h-72 overflow-y-auto pr-1">
                      {sampleQuestions.map((q) => (
                        <div
                          key={q.id}
                          className="p-3 bg-slate-50 rounded-xl border border-slate-200 text-xs space-y-1.5 hover:border-emerald-300 transition"
                        >
                          <div className="flex items-center justify-between font-mono text-[10px] text-slate-500">
                            <span className="font-bold text-slate-700">{q.id}</span>
                            <span className="px-2 py-0.5 rounded bg-white border border-slate-200 text-slate-600">
                              {q.domain}
                            </span>
                          </div>
                          <p className="font-semibold text-slate-800 leading-snug">{q.input}</p>
                          <div className="bg-white p-2 rounded-lg border border-slate-100 font-mono text-[11px] text-slate-600 leading-relaxed whitespace-pre-wrap">
                            <span className="text-[10px] uppercase font-bold text-slate-400 block mb-0.5">
                              Beklenen Çözüm &amp; Cevap:
                            </span>
                            {q.target}
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* Run Error */}
            {runError && (
              <div className="p-4 rounded-xl bg-red-50 border border-red-200 text-red-700 flex items-center gap-3 text-sm">
                <AlertCircle className="w-5 h-5 flex-shrink-0" />
                <span>{runError}</span>
              </div>
            )}

            {/* Run Success Banner & Result Details */}
            {runResult && (
              <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200 space-y-5">
                <div className="flex items-center justify-between border-b border-slate-100 pb-4">
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-xl bg-emerald-100 border border-emerald-300 flex items-center justify-center text-emerald-700">
                      <CheckCircle2 className="w-6 h-6" />
                    </div>
                    <div>
                      <h3 className="font-bold text-slate-900 text-base">Benchmark Başarıyla Tamamlandı</h3>
                      <p className="text-xs text-slate-400 font-mono">ID: {runResult.benchmark_id}</p>
                    </div>
                  </div>
                  <span className="px-3 py-1 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200 text-xs font-semibold">
                    Kalıcı DB Kaydı Oluşturuldu
                  </span>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                  <div className="p-4 rounded-xl bg-slate-50 border border-slate-200">
                    <span className="text-xs text-slate-500 block uppercase tracking-wider">Model Adı</span>
                    <span className="text-base font-bold text-slate-900 font-mono mt-1 block">
                      {runResult.model_name}
                    </span>
                  </div>
                  <div className="p-4 rounded-xl bg-slate-50 border border-slate-200">
                    <span className="text-xs text-slate-500 block uppercase tracking-wider">Benchmark</span>
                    <span className="text-base font-bold text-slate-900 capitalize mt-1 block">
                      {runResult.benchmark_name}
                    </span>
                  </div>
                  <div className="p-4 rounded-xl bg-emerald-50/60 border border-emerald-200">
                    <span className="text-xs text-emerald-700 block uppercase tracking-wider font-semibold">
                      Genel Skor
                    </span>
                    <span className="text-2xl font-black text-emerald-950 mt-0.5 block">
                      {runResult.score.toFixed(2)}
                    </span>
                  </div>
                  <div className="p-4 rounded-xl bg-slate-50 border border-slate-200">
                    <span className="text-xs text-slate-500 block uppercase tracking-wider">Değerlendirilen Örnek</span>
                    <span className="text-base font-bold text-slate-900 mt-1 block">
                      {runResult.samples_evaluated} örnek
                    </span>
                  </div>
                </div>

                {/* Soru Bazlı Akıl Yürütme ve Sonuç İnceleyici */}
                {runResult.metrics?.details && Array.isArray(runResult.metrics.details) && (
                  <div className="space-y-3 pt-2">
                    <div className="flex items-center justify-between border-b border-slate-100 pb-2">
                      <h4 className="text-sm font-bold text-slate-900 flex items-center gap-2">
                        <Sparkles className="w-4 h-4 text-emerald-600" />
                        <span>Soru Bazlı Akıl Yürütme &amp; Eşleşme Analizi ({runResult.metrics.details.length} Soru)</span>
                      </h4>
                      <span className="text-xs text-emerald-700 bg-emerald-50 px-2.5 py-1 rounded-full border border-emerald-200 font-semibold font-mono">
                        Doğruluk: %{runResult.metrics.accuracy !== undefined ? runResult.metrics.accuracy : runResult.score.toFixed(1)}
                      </span>
                    </div>

                    <div className="space-y-3 max-h-96 overflow-y-auto pr-1">
                      {runResult.metrics.details.map((item: any, idx: number) => (
                        <div
                          key={item.id || idx}
                          className={`p-4 rounded-xl border text-xs space-y-2.5 transition ${
                            item.is_correct
                              ? 'bg-emerald-50/40 border-emerald-200'
                              : 'bg-rose-50/30 border-rose-200'
                          }`}
                        >
                          <div className="flex items-center justify-between">
                            <div className="flex items-center gap-2">
                              <span className="font-mono font-bold text-slate-700">{item.id || `#${idx + 1}`}</span>
                              <span className="px-2 py-0.5 rounded bg-white border border-slate-200 text-[10px] text-slate-600 font-mono">
                                {item.domain || 'genel'}
                              </span>
                            </div>
                            <div className="flex items-center gap-2">
                              {item.steps_count !== undefined && (
                                <span className="px-2 py-0.5 rounded bg-white border border-slate-200 font-mono text-[10px] text-slate-600">
                                  {item.steps_count} Adım
                                </span>
                              )}
                              <span
                                className={`px-2.5 py-0.5 rounded-full font-bold text-[11px] flex items-center gap-1 ${
                                  item.is_correct
                                    ? 'bg-emerald-100 text-emerald-800 border border-emerald-300'
                                    : 'bg-rose-100 text-rose-800 border border-rose-300'
                                }`}
                              >
                                {item.is_correct ? '✓ Doğru' : '✗ Eşleşmedi'}
                              </span>
                            </div>
                          </div>

                          <p className="font-semibold text-slate-900 leading-snug">{item.input}</p>

                          <div className="grid grid-cols-1 md:grid-cols-2 gap-3 pt-1">
                            <div className="p-2.5 bg-white rounded-lg border border-slate-200">
                              <span className="text-[10px] uppercase font-bold text-slate-400 block mb-1">
                                Referans / Hedef Akıl Yürütme:
                              </span>
                              <p className="font-mono text-[11px] text-slate-700 whitespace-pre-wrap leading-relaxed">
                                {item.target_cot || item.target}
                              </p>
                            </div>
                            <div className="p-2.5 bg-white rounded-lg border border-slate-200">
                              <div className="flex items-center justify-between mb-1">
                                <span className="text-[10px] uppercase font-bold text-slate-400">
                                  Model Çıktısı:
                                </span>
                                {item.predicted_answer !== undefined && (
                                  <span className="text-[10px] font-mono font-bold text-indigo-700 bg-indigo-50 px-1.5 py-0.5 rounded">
                                    Tahmin: {item.predicted_answer} (Hedef: {item.target_answer})
                                  </span>
                                )}
                              </div>
                              <p className="font-mono text-[11px] text-slate-900 whitespace-pre-wrap leading-relaxed">
                                {item.model_output}
                              </p>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Metrics detail JSON */}
                <div className="bg-slate-950 p-4 rounded-xl text-slate-200 font-mono text-xs overflow-x-auto">
                  <span className="text-slate-400 block mb-2 font-sans font-semibold text-xs">
                    Metrik Çıktıları (Raw Metrics JSON)
                  </span>
                  <pre>{JSON.stringify(runResult.metrics, null, 2)}</pre>
                </div>
              </div>
            )}
          </div>
        )}

        {/* ==================================================================== */}
        {/* TAB 3: MODEL COMPARISON ARENA                                         */}
        {/* ==================================================================== */}
        {activeTab === 'arena' && (
          <div className="space-y-6">
            {/* Arena Sub-Mode Switcher */}
            <div className="bg-white p-2 rounded-2xl shadow-sm border border-slate-200 flex flex-wrap items-center justify-between gap-3">
              <div className="flex items-center gap-1.5 p-1 bg-slate-100 rounded-xl">
                <button
                  id="subtab-radar"
                  onClick={() => setArenaSubMode('radar')}
                  className={`flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-bold transition-all ${
                    arenaSubMode === 'radar'
                      ? 'bg-white text-indigo-900 shadow-sm'
                      : 'text-slate-600 hover:text-slate-900'
                  }`}
                >
                  <Compass className="w-4 h-4 text-indigo-600" />
                  5-Eksenli Model Radar Analizi (Spider Chart)
                </button>
                <button
                  id="subtab-head-to-head"
                  onClick={() => setArenaSubMode('head_to_head')}
                  className={`flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-bold transition-all ${
                    arenaSubMode === 'head_to_head'
                      ? 'bg-white text-indigo-900 shadow-sm'
                      : 'text-slate-600 hover:text-slate-900'
                  }`}
                >
                  <Trophy className="w-4 h-4 text-amber-500" />
                  İkili Karşılaştırma (Head-to-Head)
                </button>
              </div>

              <div className="text-xs text-slate-500 px-3">
                {arenaSubMode === 'radar'
                  ? 'Çoklu model 5 temel boyutta (CoT, Bilgi, BLEU, ROUGE, PPL) eşzamanlı kıyaslanır.'
                  : 'İki modeli seçilen tek bir benchmark metriği üzerinde birebir yarıştırın.'}
              </div>
            </div>

            {/* SUB-MODE 1: 5-AXIS RADAR ANALYSIS */}
            {arenaSubMode === 'radar' && (
              <div className="space-y-6">
                {/* Model Selector Card */}
                <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200 space-y-4">
                  <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 border-b border-slate-100 pb-3">
                    <div>
                      <h2 className="text-base font-bold text-slate-900 flex items-center gap-2">
                        <Compass className="w-5 h-5 text-indigo-600" />
                        Radar Analizi İçin Modelleri Seçin (2 - 5 Model)
                      </h2>
                      <p className="text-xs text-slate-500 mt-0.5">
                        Radar diyagramında karşılaştırılacak modelleri işaretleyin. Seçili: {radarSelectedModels.length} / 5
                      </p>
                    </div>

                    <button
                      id="btn-run-radar"
                      onClick={() => handleRadarComparison()}
                      disabled={radarLoading || radarSelectedModels.length < 1}
                      className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white font-semibold text-xs transition shadow-md shadow-indigo-600/20 disabled:opacity-50"
                    >
                      {radarLoading ? (
                        <>
                          <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                          Radar Hesaplanıyor...
                        </>
                      ) : (
                        <>
                          <Sparkles className="w-3.5 h-3.5" />
                          Radar Analizini Çalıştır
                        </>
                      )}
                    </button>
                  </div>

                  {/* Model Choice Chips */}
                  <div className="flex flex-wrap items-center gap-2">
                    {availableModels.map((m) => {
                      const isSelected = radarSelectedModels.includes(m);
                      const colorIdx = radarSelectedModels.indexOf(m);
                      const palette = colorIdx >= 0 ? RADAR_PALETTE[colorIdx % RADAR_PALETTE.length] : null;

                      return (
                        <button
                          key={m}
                          type="button"
                          onClick={() => toggleRadarModel(m)}
                          className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-xl text-xs font-semibold border transition-all ${
                            isSelected && palette
                              ? `${palette.badge} ring-2 ring-indigo-500/20 shadow-sm`
                              : 'bg-slate-50 text-slate-600 border-slate-200 hover:bg-slate-100'
                          }`}
                        >
                          <span
                            className={`w-3.5 h-3.5 rounded flex items-center justify-center border text-[10px] ${
                              isSelected
                                ? 'bg-indigo-600 text-white border-indigo-600'
                                : 'bg-white border-slate-300'
                            }`}
                          >
                            {isSelected && <Check className="w-2.5 h-2.5" />}
                          </span>
                          <span className="font-mono">{m}</span>
                        </button>
                      );
                    })}
                  </div>
                </div>

                {/* Radar Error */}
                {radarError && (
                  <div className="p-4 rounded-xl bg-red-50 border border-red-200 text-red-700 flex items-center gap-3 text-sm">
                    <AlertCircle className="w-5 h-5 flex-shrink-0" />
                    <span>{radarError}</span>
                  </div>
                )}

                {/* Radar Loading Indicator */}
                {radarLoading && (
                  <div className="p-12 bg-white rounded-2xl border border-slate-200 text-center space-y-3">
                    <RefreshCw className="w-8 h-8 text-indigo-600 animate-spin mx-auto" />
                    <h3 className="font-bold text-slate-800 text-sm">5 Boyutlu Model Başarımı Hesaplanıyor...</h3>
                    <p className="text-xs text-slate-400 max-w-md mx-auto">
                      GSM8K Akıl Yürütme, Türkçe Bilgi, BLEU Akıcılığı, ROUGE Özetleme ve Perplexity metrikleri modeller üzerinde analiz ediliyor.
                    </p>
                  </div>
                )}

                {/* Radar Results Section */}
                {radarData && !radarLoading && (
                  <div className="space-y-6">
                    {/* Overall Winner Hero Card */}
                    <div className="p-5 rounded-2xl bg-gradient-to-r from-indigo-500/10 via-purple-500/15 to-emerald-500/10 border border-indigo-200 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
                      <div className="flex items-center gap-4">
                        <div className="w-12 h-12 rounded-2xl bg-gradient-to-tr from-amber-500 to-indigo-600 text-white flex items-center justify-center text-2xl shadow-md">
                          🏆
                        </div>
                        <div>
                          <span className="text-[11px] font-bold text-indigo-900 uppercase tracking-wider block">
                            5-Eksenli Radar Şampiyonu (Overall Winner)
                          </span>
                          <h3 className="text-xl font-extrabold text-slate-900 font-mono">
                            {radarData.overall_winner}
                          </h3>
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="text-xs bg-indigo-50 text-indigo-900 font-semibold px-3 py-1.5 rounded-xl border border-indigo-200 font-mono">
                          En Yüksek Ağırlıklı Ortalama Performans
                        </span>
                      </div>
                    </div>

                    {/* Chart & Dimension Breakdown Grid */}
                    <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
                      {/* Left: SVG Radar Canvas */}
                      <div className="lg:col-span-7 bg-white p-6 rounded-2xl shadow-sm border border-slate-200 flex flex-col items-center justify-between">
                        <div className="w-full flex items-center justify-between border-b border-slate-100 pb-3 mb-2">
                          <h3 className="font-bold text-slate-900 text-sm flex items-center gap-2">
                            <Activity className="w-4 h-4 text-indigo-600" />
                            Model Spider / Radar Grafiği
                          </h3>
                          <span className="text-[11px] text-slate-400 font-medium">0 - 100 Normalize Ölçek</span>
                        </div>

                        {/* Interactive SVG Radar */}
                        <div className="relative w-full max-w-[480px] aspect-square flex items-center justify-center py-2">
                          <svg
                            viewBox="0 0 480 480"
                            className="w-full h-full overflow-visible drop-shadow-sm"
                          >
                            <defs>
                              {radarData.models.map((m, mIdx) => {
                                const pal = RADAR_PALETTE[mIdx % RADAR_PALETTE.length];
                                return (
                                  <radialGradient
                                    key={`rad-grad-${m.model_name}`}
                                    id={`radar-glow-${mIdx}`}
                                    cx="50%"
                                    cy="50%"
                                    r="50%"
                                  >
                                    <stop offset="0%" stopColor={pal.stroke} stopOpacity="0.4" />
                                    <stop offset="100%" stopColor={pal.stroke} stopOpacity="0.05" />
                                  </radialGradient>
                                );
                              })}
                            </defs>

                            {/* Concentric Pentagon Rings */}
                            {[0.2, 0.4, 0.6, 0.8, 1.0].map((lvl, ringIdx) => {
                              const ringPoints = Array.from({ length: 5 }, (_, i) => {
                                const angle = -Math.PI / 2 + i * ((2 * Math.PI) / 5);
                                const x = 240 + lvl * 155 * Math.cos(angle);
                                const y = 240 + lvl * 155 * Math.sin(angle);
                                return `${x.toFixed(1)},${y.toFixed(1)}`;
                              }).join(' ');

                              return (
                                <g key={`ring-${ringIdx}`}>
                                  <polygon
                                    points={ringPoints}
                                    fill={ringIdx % 2 === 0 ? 'rgba(241, 245, 249, 0.4)' : 'none'}
                                    stroke="#cbd5e1"
                                    strokeWidth="1"
                                    strokeDasharray={lvl < 1.0 ? '3 3' : 'none'}
                                  />
                                  {/* Level percentage label at top spoke */}
                                  <text
                                    x="244"
                                    y={240 - lvl * 155 + 11}
                                    fill="#94a3b8"
                                    fontSize="10"
                                    fontFamily="monospace"
                                    fontWeight="bold"
                                  >
                                    {(lvl * 100).toFixed(0)}%
                                  </text>
                                </g>
                              );
                            })}

                            {/* Spoke Axes & Dimension Labels */}
                            {radarData.dimensions.map((dimName, i) => {
                              const angle = -Math.PI / 2 + i * ((2 * Math.PI) / 5);
                              const xEnd = 240 + 155 * Math.cos(angle);
                              const yEnd = 240 + 155 * Math.sin(angle);

                              // Label coordinate
                              const labelR = 192;
                              const lx = 240 + labelR * Math.cos(angle);
                              const ly = 240 + labelR * Math.sin(angle);

                              let textAnchor = 'middle';
                              if (Math.cos(angle) > 0.25) textAnchor = 'start';
                              else if (Math.cos(angle) < -0.25) textAnchor = 'end';

                              return (
                                <g key={`axis-${i}`}>
                                  {/* Spoke Line */}
                                  <line
                                    x1="240"
                                    y1="240"
                                    x2={xEnd.toFixed(1)}
                                    y2={yEnd.toFixed(1)}
                                    stroke="#94a3b8"
                                    strokeWidth="1.2"
                                  />
                                  {/* Outer tick */}
                                  <circle
                                    cx={xEnd.toFixed(1)}
                                    cy={yEnd.toFixed(1)}
                                    r="2.5"
                                    fill="#64748b"
                                  />
                                  {/* Label text */}
                                  <text
                                    x={lx.toFixed(1)}
                                    y={ly.toFixed(1)}
                                    textAnchor={textAnchor as 'start' | 'end' | 'middle'}
                                    fill="#1e293b"
                                    fontSize="11"
                                    fontWeight="bold"
                                    className="select-none"
                                  >
                                    {dimName}
                                  </text>
                                </g>
                              );
                            })}

                            {/* Model Overlay Polygons */}
                            {radarData.models.map((m, mIdx) => {
                              if (visibleRadarModels[m.model_name] === false) return null;
                              const pal = RADAR_PALETTE[mIdx % RADAR_PALETTE.length];

                              const pts = m.dimensions.map((dim, i) => {
                                const angle = -Math.PI / 2 + i * ((2 * Math.PI) / 5);
                                const r = (Math.max(5, Math.min(100, dim.score)) / 100) * 155;
                                const x = 240 + r * Math.cos(angle);
                                const y = 240 + r * Math.sin(angle);
                                return { x, y, score: dim.score, name: dim.dimension_name || dim.name || '' };
                              });

                              const pointsStr = pts.map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' ');

                              return (
                                <g key={`model-poly-${m.model_name}`} className="transition-all duration-300">
                                  {/* Filled Polygon */}
                                  <polygon
                                    points={pointsStr}
                                    fill={pal.fill}
                                    stroke={pal.stroke}
                                    strokeWidth="2.5"
                                    className="transition-all hover:opacity-90"
                                  />

                                  {/* Dimension Vertices */}
                                  {pts.map((pt, pIdx) => (
                                    <g key={`pt-${m.model_name}-${pIdx}`}>
                                      <circle
                                        cx={pt.x.toFixed(1)}
                                        cy={pt.y.toFixed(1)}
                                        r="4"
                                        fill={pal.dot}
                                        stroke="#ffffff"
                                        strokeWidth="1.5"
                                        className="transition hover:r-6 cursor-pointer"
                                      >
                                        <title>{`${m.model_name}\n${pt.name}: ${pt.score.toFixed(1)}`}</title>
                                      </circle>
                                    </g>
                                  ))}
                                </g>
                              );
                            })}
                          </svg>
                        </div>

                        {/* Visibility Legend / Model Toggles */}
                        <div className="w-full pt-3 border-t border-slate-100 flex flex-wrap items-center justify-center gap-3">
                          {radarData.models.map((m, mIdx) => {
                            const pal = RADAR_PALETTE[mIdx % RADAR_PALETTE.length];
                            const isVis = visibleRadarModels[m.model_name] !== false;

                            return (
                              <button
                                key={`legend-${m.model_name}`}
                                type="button"
                                onClick={() => toggleRadarVisibility(m.model_name)}
                                className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-semibold border transition ${
                                  isVis
                                    ? 'bg-slate-50 border-slate-300 text-slate-800'
                                    : 'bg-slate-100 border-slate-200 text-slate-400 line-through'
                                }`}
                              >
                                <span
                                  className="w-3 h-3 rounded-full"
                                  style={{ backgroundColor: pal.stroke }}
                                />
                                <span className="font-mono">{m.model_name}</span>
                                <span className="text-[10px] text-slate-500 font-normal">
                                  ({(m.overall_average ?? m.overall_score ?? 0).toFixed(1)})
                                </span>
                              </button>
                            );
                          })}
                        </div>
                      </div>

                      {/* Right: Dimension Winners & Breakdown */}
                      <div className="lg:col-span-5 space-y-4">
                        <div className="bg-white p-5 rounded-2xl shadow-sm border border-slate-200 space-y-3">
                          <h3 className="font-bold text-slate-900 text-sm flex items-center gap-2 border-b border-slate-100 pb-2">
                            <Trophy className="w-4 h-4 text-amber-500" />
                            Boyut Bazlı Liderler &amp; Dağılım
                          </h3>

                          <div className="space-y-3">
                            {radarData.dimensions.map((dimName) => {
                              const dimWinner = radarData.winner_by_dimension[dimName];
                              return (
                                <div
                                  key={`dim-card-${dimName}`}
                                  className="p-3 rounded-xl bg-slate-50/70 border border-slate-200 text-xs space-y-1.5"
                                >
                                  <div className="flex items-center justify-between">
                                    <span className="font-bold text-slate-800">{dimName}</span>
                                    <span className="px-2 py-0.5 rounded-full bg-amber-100 text-amber-900 border border-amber-300 font-mono font-bold text-[10px]">
                                      ★ Lider: {dimWinner}
                                    </span>
                                  </div>

                                  {/* Progress bar for each model in this dimension */}
                                  <div className="space-y-1 pt-1">
                                    {radarData.models.map((m, mIdx) => {
                                      const dimScoreObj = m.dimensions.find(
                                        (d) => d.dimension_name === dimName || d.name === dimName
                                      );
                                      const scoreVal = dimScoreObj ? dimScoreObj.score : 0;
                                      const pal = RADAR_PALETTE[mIdx % RADAR_PALETTE.length];
                                      const isWinner = m.model_name === dimWinner;

                                      return (
                                        <div key={`dim-${dimName}-${m.model_name}`} className="space-y-0.5">
                                          <div className="flex items-center justify-between text-[11px]">
                                            <span className="font-mono text-slate-600 truncate max-w-[140px]">
                                              {m.model_name}
                                            </span>
                                            <span className={`font-mono font-bold ${isWinner ? 'text-amber-700' : 'text-slate-700'}`}>
                                              {scoreVal.toFixed(1)}
                                            </span>
                                          </div>
                                          <div className="w-full bg-slate-200 rounded-full h-1.5 overflow-hidden">
                                            <div
                                              className="h-1.5 rounded-full transition-all duration-500"
                                              style={{
                                                width: `${Math.max(3, Math.min(100, scoreVal))}%`,
                                                backgroundColor: pal.stroke,
                                              }}
                                            />
                                          </div>
                                        </div>
                                      );
                                    })}
                                  </div>
                                </div>
                              );
                            })}
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Bottom: Detailed Score Matrix Table */}
                    <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200 space-y-4">
                      <div className="flex items-center justify-between border-b border-slate-100 pb-3">
                        <h3 className="font-bold text-slate-900 text-sm flex items-center gap-2">
                          <BarChart2 className="w-4 h-4 text-indigo-600" />
                          Çok Boyutlu Başarım Matrisi (Detailed Score Table)
                        </h3>
                        <span className="text-xs text-slate-400 font-mono">Normalize Edilmiş Skorlar (0-100)</span>
                      </div>

                      <div className="overflow-x-auto">
                        <table className="w-full text-left text-xs border-collapse">
                          <thead>
                            <tr className="border-b border-slate-200 text-slate-500 bg-slate-50/50">
                              <th className="py-2.5 px-3 font-semibold">Model</th>
                              {radarData.dimensions.map((d) => (
                                <th key={d} className="py-2.5 px-3 font-semibold text-center">
                                  {d}
                                </th>
                              ))}
                              <th className="py-2.5 px-3 font-semibold text-right">Genel Ortalama</th>
                            </tr>
                          </thead>
                          <tbody className="divide-y divide-slate-100">
                            {radarData.models.map((m, mIdx) => {
                              const isOverallWinner = m.model_name === radarData.overall_winner;
                              const pal = RADAR_PALETTE[mIdx % RADAR_PALETTE.length];

                              return (
                                <tr
                                  key={m.model_name}
                                  className={`hover:bg-slate-50/80 transition ${
                                    isOverallWinner ? 'bg-amber-50/20 font-semibold' : ''
                                  }`}
                                >
                                  <td className="py-3 px-3 flex items-center gap-2">
                                    <span
                                      className="w-2.5 h-2.5 rounded-full"
                                      style={{ backgroundColor: pal.stroke }}
                                    />
                                    <span className="font-mono text-slate-900">{m.model_name}</span>
                                    {isOverallWinner && (
                                      <span className="px-1.5 py-0.5 rounded bg-amber-100 text-amber-800 text-[10px] font-bold border border-amber-300">
                                        Şampiyon
                                      </span>
                                    )}
                                  </td>
                                  {radarData.dimensions.map((dimName) => {
                                    const dObj = m.dimensions.find(
                                      (d) => d.dimension_name === dimName || d.name === dimName
                                    );
                                    const score = dObj ? dObj.score : 0;
                                    const isDimWinner = radarData.winner_by_dimension[dimName] === m.model_name;

                                    return (
                                      <td key={dimName} className="py-3 px-3 text-center">
                                        <span
                                          className={`font-mono px-2 py-0.5 rounded-md ${
                                            isDimWinner
                                              ? 'bg-amber-100 text-amber-900 font-bold border border-amber-300'
                                              : 'text-slate-700'
                                          }`}
                                        >
                                          {score.toFixed(1)}
                                        </span>
                                      </td>
                                    );
                                  })}
                                  <td className="py-3 px-3 text-right">
                                    <span className="font-mono text-sm font-black text-slate-900">
                                      {(m.overall_average ?? m.overall_score ?? 0).toFixed(1)}
                                    </span>
                                  </td>
                                </tr>
                              );
                            })}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* SUB-MODE 2: HEAD TO HEAD BENCHMARK BATTLE */}
            {arenaSubMode === 'head_to_head' && (
              <div className="space-y-6">
                <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200">
                  <div className="border-b border-slate-100 pb-4 mb-5">
                    <h2 className="text-lg font-bold text-slate-900 flex items-center gap-2">
                      <Trophy className="w-5 h-5 text-amber-500" />
                      Model Karşılaştırma Arenası (Arena Comparison)
                    </h2>
                    <p className="text-xs text-slate-500 mt-1">
                      İki farklı modeli aynı benchmark üzerinde yarıştırın. Sistem hangi modelin daha üstün performans gösterdiğini analiz eder.
                    </p>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                    {/* Model A */}
                    <div className="p-4 rounded-xl border border-blue-200 bg-blue-50/30">
                      <label htmlFor="arena-model-a" className="block text-xs font-bold text-blue-900 uppercase tracking-wider mb-2 flex items-center gap-1.5">
                        <span className="w-2.5 h-2.5 rounded-full bg-blue-500"></span>
                        Model A (Birinci Model)
                      </label>
                      <select
                        id="arena-model-a"
                        value={compareModelA}
                        onChange={(e) => setCompareModelA(e.target.value)}
                        className="w-full p-2.5 rounded-xl border border-slate-200 bg-white text-sm font-medium"
                      >
                        {availableModels.map((m, i) => (
                          <option key={i} value={m}>
                            {m}
                          </option>
                        ))}
                      </select>
                    </div>

                    {/* Model B */}
                    <div className="p-4 rounded-xl border border-purple-200 bg-purple-50/30">
                      <label htmlFor="arena-model-b" className="block text-xs font-bold text-purple-900 uppercase tracking-wider mb-2 flex items-center gap-1.5">
                        <span className="w-2.5 h-2.5 rounded-full bg-purple-500"></span>
                        Model B (İkinci Model)
                      </label>
                      <select
                        id="arena-model-b"
                        value={compareModelB}
                        onChange={(e) => setCompareModelB(e.target.value)}
                        className="w-full p-2.5 rounded-xl border border-slate-200 bg-white text-sm font-medium"
                      >
                        {availableModels.map((m, i) => (
                          <option key={i} value={m}>
                            {m}
                          </option>
                        ))}
                      </select>
                    </div>

                    {/* Benchmark */}
                    <div className="p-4 rounded-xl border border-slate-200 bg-slate-50/50">
                      <label htmlFor="arena-benchmark" className="block text-xs font-bold text-slate-700 uppercase tracking-wider mb-2">
                        Yarışma Benchmark Metriği
                      </label>
                      <select
                        id="arena-benchmark"
                        value={compareBenchmark}
                        onChange={(e) => setCompareBenchmark(e.target.value)}
                        className="w-full p-2.5 rounded-xl border border-slate-200 bg-white text-sm font-medium"
                      >
                        <option value="perplexity">Perplexity (Düşük = İyi)</option>
                        <option value="bleu">BLEU Score (Yüksek = İyi)</option>
                        <option value="rouge">ROUGE Score (Yüksek = İyi)</option>
                        <option value="accuracy">Accuracy (Yüksek = İyi)</option>
                        <option value="gsm8k_cot">GSM8K CoT Reasoning (Yüksek = İyi)</option>
                        <option value="turkish_knowledge">Türkçe Bilgi &amp; Doğruluk (Yüksek = İyi)</option>
                      </select>
                    </div>
                  </div>

                  <div className="mt-6 flex justify-end">
                    <button
                      id="btn-compare-models"
                      onClick={handleCompare}
                      disabled={compareLoading}
                      className="inline-flex items-center gap-2 px-6 py-2.5 rounded-xl bg-amber-600 hover:bg-amber-500 text-white font-semibold text-sm transition shadow-md shadow-amber-600/20 disabled:opacity-50"
                    >
                      {compareLoading ? (
                        <>
                          <RefreshCw className="w-4 h-4 animate-spin" />
                          Karşılaştırılıyor...
                        </>
                      ) : (
                        <>
                          <Trophy className="w-4 h-4" />
                          Arena Karşılaştırması Başlat
                        </>
                      )}
                    </button>
                  </div>
                </div>

                {/* Compare Error */}
                {compareError && (
                  <div className="p-4 rounded-xl bg-red-50 border border-red-200 text-red-700 flex items-center gap-3 text-sm">
                    <AlertCircle className="w-5 h-5 flex-shrink-0" />
                    <span>{compareError}</span>
                  </div>
                )}

                {/* Compare Result */}
                {compareResult && (
                  <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200 space-y-6">
                    {/* Winner Banner */}
                    <div className="p-5 rounded-2xl bg-gradient-to-r from-amber-500/10 via-amber-500/20 to-amber-500/10 border border-amber-300 flex items-center justify-between">
                      <div className="flex items-center gap-4">
                        <div className="w-12 h-12 rounded-2xl bg-amber-500 text-white flex items-center justify-center text-2xl shadow-md">
                          🏆
                        </div>
                        <div>
                          <span className="text-xs font-bold text-amber-800 uppercase tracking-wider">
                            Arena Kazananı (Winner)
                          </span>
                          <h3 className="text-xl font-extrabold text-slate-900 font-mono">
                            {compareResult.winner}
                          </h3>
                        </div>
                      </div>
                      <span className="text-xs bg-amber-100 text-amber-900 font-semibold px-3 py-1 rounded-full border border-amber-300">
                        {compareResult.benchmark_name.toUpperCase()} Metriğinde Üstün
                      </span>
                    </div>

                    {/* Head to Head Side by Side Cards */}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      {compareResult.comparisons.map((c, i) => {
                        const isWinner = c.model_name === compareResult.winner;
                        return (
                          <div
                            key={i}
                            className={`p-5 rounded-2xl border transition-all ${
                              isWinner
                                ? 'bg-amber-50/40 border-amber-300 shadow-md ring-2 ring-amber-400/30'
                                : 'bg-slate-50/60 border-slate-200'
                            }`}
                          >
                            <div className="flex items-center justify-between mb-3">
                              <span className="font-bold text-slate-900 font-mono text-sm">{c.model_name}</span>
                              {isWinner ? (
                                <span className="px-2.5 py-0.5 rounded-md bg-amber-500 text-white text-xs font-bold">
                                  KAZANAN
                                </span>
                              ) : (
                                <span className="px-2.5 py-0.5 rounded-md bg-slate-200 text-slate-600 text-xs font-medium">
                                  İkinci
                                </span>
                              )}
                            </div>
                            <div className="text-3xl font-black text-slate-900">
                              {c.score.toFixed(2)}
                            </div>
                            <span className="text-xs text-slate-500 block mt-1">
                              {compareResult.benchmark_name} skoru
                            </span>

                            <div className="mt-4 pt-3 border-t border-slate-200/80 text-xs text-slate-600 space-y-1 font-mono">
                              {Object.entries(c.metrics || {}).map(([key, val]) => (
                                <div key={key} className="flex justify-between">
                                  <span className="text-slate-500">{key}:</span>
                                  <span className="font-semibold">{String(val)}</span>
                                </div>
                              ))}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* ==================================================================== */}
        {/* TAB 4: LEADERBOARD & HISTORY                                          */}
        {/* ==================================================================== */}
        {activeTab === 'history' && (
          <div className="space-y-6">
            {/* Filter & Refresh Controls */}
            <div className="bg-white p-4 rounded-2xl shadow-sm border border-slate-200 flex flex-col md:flex-row items-center justify-between gap-4">
              <div className="flex flex-wrap items-center gap-3 w-full md:w-auto">
                <div className="relative">
                  <input
                    type="text"
                    placeholder="Modele göre filtrele..."
                    value={historyFilterModel}
                    onChange={(e) => setHistoryFilterModel(e.target.value)}
                    className="p-2 pl-8 rounded-xl border border-slate-200 text-xs bg-slate-50 focus:outline-none focus:ring-2 focus:ring-indigo-500/20 w-44"
                  />
                  <Search className="w-3.5 h-3.5 text-slate-400 absolute left-2.5 top-3" />
                </div>

                <div className="relative">
                  <input
                    type="text"
                    placeholder="Benchmark'a göre filtrele..."
                    value={historyFilterBenchmark}
                    onChange={(e) => setHistoryFilterBenchmark(e.target.value)}
                    className="p-2 pl-8 rounded-xl border border-slate-200 text-xs bg-slate-50 focus:outline-none focus:ring-2 focus:ring-indigo-500/20 w-48"
                  />
                  <Search className="w-3.5 h-3.5 text-slate-400 absolute left-2.5 top-3" />
                </div>
              </div>

              <button
                onClick={fetchHistory}
                disabled={historyLoading}
                className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-slate-100 hover:bg-slate-200 text-slate-700 text-xs font-semibold transition"
              >
                <RefreshCw className={`w-3.5 h-3.5 ${historyLoading ? 'animate-spin' : ''}`} />
                Yenile
              </button>
            </div>

            {/* History Table */}
            <div className="bg-white rounded-2xl shadow-sm border border-slate-200 overflow-hidden">
              <div className="p-4 border-b border-slate-100 flex items-center justify-between">
                <h3 className="font-bold text-slate-900 text-sm flex items-center gap-2">
                  <Database className="w-4 h-4 text-indigo-600" />
                  Kayıtlı Benchmark Sonuçları ({historyResults.length})
                </h3>
                <span className="text-xs text-slate-400">Veritabanından canlı listeleniyor</span>
              </div>

              {historyLoading ? (
                <div className="p-12 text-center text-slate-400 text-sm flex flex-col items-center gap-2">
                  <RefreshCw className="w-6 h-6 animate-spin text-indigo-600" />
                  Kayıtlar yükleniyor...
                </div>
              ) : historyResults.length === 0 ? (
                <div className="p-12 text-center text-slate-500 text-sm">
                  <BarChart2 className="w-10 h-10 text-slate-300 mx-auto mb-2" />
                  Henüz kaydedilmiş benchmark kaydı bulunamadı.
                  <p className="text-xs text-slate-400 mt-1">
                    &quot;Benchmark Runner&quot; sekmesine gidip bir model testi başlatabilirsiniz.
                  </p>
                </div>
              ) : (
                <div className="overflow-x-auto">
                  <table className="w-full text-left border-collapse text-xs">
                    <thead>
                      <tr className="bg-slate-50/80 text-slate-500 font-semibold border-b border-slate-200">
                        <th className="py-3 px-4">ID</th>
                        <th className="py-3 px-4">Model Adı</th>
                        <th className="py-3 px-4">Benchmark</th>
                        <th className="py-3 px-4">Skor</th>
                        <th className="py-3 px-4">Örnekler</th>
                        <th className="py-3 px-4">Tarih</th>
                        <th className="py-3 px-4 text-right">İşlemler</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100">
                      {historyResults.map((item) => (
                        <tr key={item.benchmark_id} className="hover:bg-slate-50/60 transition">
                          <td className="py-3 px-4 font-mono font-semibold text-slate-700">
                            {item.benchmark_id}
                          </td>
                          <td className="py-3 px-4 font-mono text-indigo-900 font-bold">
                            {item.model_name}
                          </td>
                          <td className="py-3 px-4 capitalize">
                            <span className="px-2 py-0.5 rounded bg-slate-100 text-slate-700 font-medium">
                              {item.benchmark_name}
                            </span>
                          </td>
                          <td className="py-3 px-4 font-black text-slate-900 text-sm">
                            {item.score.toFixed(2)}
                          </td>
                          <td className="py-3 px-4 text-slate-600">
                            {item.samples_evaluated}
                          </td>
                          <td className="py-3 px-4 text-slate-400">
                            {new Date(item.timestamp).toLocaleString('tr-TR')}
                          </td>
                          <td className="py-3 px-4 text-right space-x-2">
                            <button
                              onClick={() => setSelectedHistoryItem(item)}
                              className="px-2.5 py-1 rounded-md bg-indigo-50 hover:bg-indigo-100 text-indigo-700 font-medium transition"
                            >
                              Detay
                            </button>

                            {deleteConfirmId === item.benchmark_id ? (
                              <div className="inline-flex items-center gap-1">
                                <button
                                  onClick={() => handleDeleteHistory(item.benchmark_id)}
                                  className="px-2 py-1 rounded bg-red-600 text-white font-semibold hover:bg-red-700"
                                >
                                  Onayla
                                </button>
                                <button
                                  onClick={() => setDeleteConfirmId(null)}
                                  className="px-2 py-1 rounded bg-slate-200 text-slate-700"
                                >
                                  İptal
                                </button>
                              </div>
                            ) : (
                              <button
                                onClick={() => setDeleteConfirmId(item.benchmark_id)}
                                className="p-1 rounded hover:bg-red-50 text-slate-400 hover:text-red-600 transition"
                                title="Sil"
                              >
                                <Trash2 className="w-4 h-4" />
                              </button>
                            )}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>

            {/* Modal / Detail View for selected item */}
            {selectedHistoryItem && (
              <div className="fixed inset-0 z-50 bg-slate-950/60 backdrop-blur-xs flex items-center justify-center p-4">
                <div className="bg-white rounded-2xl shadow-xl border border-slate-200 max-w-lg w-full p-6 space-y-4">
                  <div className="flex items-center justify-between border-b border-slate-100 pb-3">
                    <h4 className="font-bold text-slate-900 text-sm">
                      Benchmark Detayı: {selectedHistoryItem.benchmark_id}
                    </h4>
                    <button
                      onClick={() => setSelectedHistoryItem(null)}
                      className="text-slate-400 hover:text-slate-700 text-sm font-bold"
                    >
                      ✕
                    </button>
                  </div>

                  <div className="space-y-2 text-xs">
                    <div className="flex justify-between py-1 border-b border-slate-50">
                      <span className="text-slate-500">Model:</span>
                      <span className="font-mono font-bold text-slate-800">{selectedHistoryItem.model_name}</span>
                    </div>
                    <div className="flex justify-between py-1 border-b border-slate-50">
                      <span className="text-slate-500">Benchmark:</span>
                      <span className="font-semibold text-slate-800">{selectedHistoryItem.benchmark_name}</span>
                    </div>
                    <div className="flex justify-between py-1 border-b border-slate-50">
                      <span className="text-slate-500">Skor:</span>
                      <span className="font-bold text-indigo-700 text-sm">{selectedHistoryItem.score.toFixed(2)}</span>
                    </div>
                    <div className="flex justify-between py-1 border-b border-slate-50">
                      <span className="text-slate-500">Örnek Sayısı:</span>
                      <span className="text-slate-800">{selectedHistoryItem.samples_evaluated}</span>
                    </div>
                    <div className="flex justify-between py-1 border-b border-slate-50">
                      <span className="text-slate-500">Tarih:</span>
                      <span className="text-slate-800">{new Date(selectedHistoryItem.timestamp).toLocaleString('tr-TR')}</span>
                    </div>
                  </div>

                  <div className="bg-slate-950 p-3 rounded-xl text-slate-200 font-mono text-[11px] overflow-x-auto max-h-48">
                    <pre>{JSON.stringify(selectedHistoryItem.metrics, null, 2)}</pre>
                  </div>

                  <div className="flex justify-end pt-2">
                    <button
                      onClick={() => setSelectedHistoryItem(null)}
                      className="px-4 py-2 rounded-xl bg-slate-100 hover:bg-slate-200 text-slate-700 text-xs font-semibold"
                    >
                      Kapat
                    </button>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
