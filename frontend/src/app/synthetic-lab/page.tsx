'use client';

import React, { useState, useEffect, useMemo } from 'react';
import Link from 'next/link';
import {
  Sparkles,
  Filter,
  CheckCircle2,
  AlertCircle,
  AlertTriangle,
  RefreshCw,
  ArrowRight,
  Database,
  Layers,
  Activity,
  Code,
  Sliders,
  Download,
  Info,
  ShieldCheck,
  Search,
  BookOpen,
  Cpu,
  Brain,
} from 'lucide-react';
import {
  api,
  SyntheticSample,
  SyntheticTemplatesResponse,
  FilterPipelineResponse,
  FilterThresholdsInput,
  IngestToDatasetResponse,
} from '@/lib/api';

// Fallback presets if offline
const FALLBACK_TEMPLATES: SyntheticTemplatesResponse = {
  paradigms: [
    {
      id: 'self_instruct',
      name: 'Self-Instruct (Alpaca Tarzı)',
      description: 'Görev tohumlarından çok yönlü talimat-yanıt çiftleri türetme.',
      icon: '⚡',
    },
    {
      id: 'chain_of_thought',
      name: 'Düşünce Zinciri (Chain-of-Thought)',
      description: 'Adım adım mantıksal akıl yürütme ve problem çözme çiftleri.',
      icon: '🧠',
    },
    {
      id: 'code_synthesis',
      name: 'Kod & Algoritma Sentezi',
      description: 'Programlama problemleri, kod blokları ve karmaşıklık analizleri.',
      icon: '💻',
    },
    {
      id: 'textbook_qa',
      name: 'Ders Kitabı Q&A (Cosmopedia Tarzı)',
      description: 'Didaktik, pedagojik açıklamalar ve derin kavramsal okuma metinleri.',
      icon: '📚',
    },
  ],
  domains: [
    { id: 'computer_science', name: 'Bilgisayar Bilimleri & Algoritmalar' },
    { id: 'mathematics', name: 'Matematik & Olasılık' },
    { id: 'natural_sciences', name: 'Doğa Bilimleri & Biyoloji' },
    { id: 'turkish_knowledge', name: 'Türk Dili & Genel Kültür' },
  ],
  complexities: ['basic', 'intermediate', 'advanced'],
  preset_profiles: {
    strict: {
      name: 'Katı Filtre (Yüksek Kalite SFT)',
      min_perplexity: 10.0,
      max_perplexity: 85.0,
      min_words: 25,
      max_words: 1000,
      max_repetition_ratio: 0.12,
      min_quality_score: 75.0,
      pii_action: 'reject',
      max_jaccard_similarity: 0.75,
    },
    balanced: {
      name: 'Dengeli Filtre (Varsayılan Ön Eğitim / SFT)',
      min_perplexity: 5.0,
      max_perplexity: 120.0,
      min_words: 15,
      max_words: 1500,
      max_repetition_ratio: 0.18,
      min_quality_score: 65.0,
      pii_action: 'mask',
      max_jaccard_similarity: 0.82,
    },
    lenient: {
      name: 'Esnek Filtre (Geniş Kapsamlı Veri Çeşitliliği)',
      min_perplexity: 3.0,
      max_perplexity: 160.0,
      min_words: 8,
      max_words: 3000,
      max_repetition_ratio: 0.28,
      min_quality_score: 50.0,
      pii_action: 'mask',
      max_jaccard_similarity: 0.9,
    },
  },
};

// Preset inspector examples
const INSPECTOR_EXAMPLES = [
  {
    title: 'Kusursuz Pedagojik Örnek (Yüksek Kalite)',
    instruction: 'İkili arama (Binary Search) algoritmasının mantığını ve zaman karmaşıklığını açıklayın.',
    response:
      'İkili arama (Binary Search), sıralı bir dizi üzerinde hedef elemanı bulmak için böl ve fethet (divide and conquer) prensibini uygular.\n\nÇalışma Adımları:\n1. Dizinin ortasındaki eleman seçilir.\n2. Aranan değer orta elemandan küçükse sol yarıya, büyükse sağ yarıya odaklanılır.\n3. Bu adım hedef bulunana veya aralık tükenene kadar tekrarlanır.\n\nKarmaşıklık:\n- Zaman: O(log n)\n- Alan: O(1)',
  },
  {
    title: 'Model Çökmesi & Aşırı Tekrar (Düşük Perplexity)',
    instruction: 'Yapay zeka modellerinin genel çalışma mantığı nedir?',
    response:
      'model model model model model model model model model model model model model model model model model model model model model model model model model model model model model model model model',
  },
  {
    title: 'PII Sızıntısı & Hassas Veri',
    instruction: 'Müşteri destek kaydı oluşturun.',
    response:
      'Müşteri Ahmet Yılmaz adına talep açıldı. TC Kimlik No: 10000000146, Telefon: 0555 123 45 67, E-posta: ahmet.yilmaz@sirket.com. Temsilcimiz en kısa sürede dönüş yapacaktır.',
  },
  {
    title: 'Halüsinasyon & Düşük Tutarlılık (Yüksek Perplexity)',
    instruction: 'Kuantum alan teorisi hakkında bilgi verin.',
    response:
      'asdfghjk zxcvbnm qwe rty uiop 1928374 xkcd qwerty zzz random gibberish phrase lorem ipsum qwertyuiop asdfghjkl zxcvbnm 987654321 @#$%^&*',
  },
];

export default function SyntheticLabPage() {
  const [activeTab, setActiveTab] = useState<'generator' | 'pipeline' | 'inspector'>('generator');

  // Templates & Metadata
  const [metadata, setMetadata] = useState<SyntheticTemplatesResponse>(FALLBACK_TEMPLATES);

  // Tab 1: Generator State
  const [genParadigm, setGenParadigm] = useState<string>('self_instruct');
  const [genDomain, setGenDomain] = useState<string>('computer_science');
  const [genComplexity, setGenComplexity] = useState<string>('intermediate');
  const [genCount, setGenCount] = useState<number>(5);
  const [genIncludeEdgeCases, setGenIncludeEdgeCases] = useState<boolean>(true);
  const [genLoading, setGenLoading] = useState<boolean>(false);
  const [generatedSamples, setGeneratedSamples] = useState<SyntheticSample[]>([]);

  // Tab 2: Filter Pipeline State
  const [selectedProfile, setSelectedProfile] = useState<'strict' | 'balanced' | 'lenient'>('balanced');
  const [filterThresholds, setFilterThresholds] = useState<FilterThresholdsInput>(
    FALLBACK_TEMPLATES.preset_profiles.balanced
  );
  const [filterLoading, setFilterLoading] = useState<boolean>(false);
  const [pipelineResult, setPipelineResult] = useState<FilterPipelineResponse | null>(null);
  const [viewFilterTab, setViewFilterTab] = useState<'passed' | 'rejected'>('passed');

  // Tab 3: Single Inspector State
  const [inspectInstr, setInspectInstr] = useState<string>(INSPECTOR_EXAMPLES[0].instruction);
  const [inspectResp, setInspectResp] = useState<string>(INSPECTOR_EXAMPLES[0].response);
  const [inspectLoading, setInspectLoading] = useState<boolean>(false);
  const [inspectResult, setInspectResult] = useState<SyntheticSample['metrics'] | null>(null);

  // Export State
  const [exportLoading, setExportLoading] = useState<boolean>(false);
  const [exportMessage, setExportMessage] = useState<string | null>(null);

  // Ingest State (Dataset Compiler Bridge)
  const [ingestLoading, setIngestLoading] = useState<boolean>(false);
  const [ingestResult, setIngestResult] = useState<IngestToDatasetResponse | null>(null);
  const [ingestError, setIngestError] = useState<string | null>(null);

  // Fetch templates metadata on mount
  useEffect(() => {
    let isMounted = true;
    const fetchTemplates = async () => {
      try {
        const data = await api.syntheticLab.getTemplates();
        if (isMounted && data) {
          setMetadata(data);
          if (data.preset_profiles[selectedProfile]) {
            setFilterThresholds(data.preset_profiles[selectedProfile]);
          }
        }
      } catch (err) {
        console.warn('Could not load templates metadata, using fallback.', err);
      }
    };
    fetchTemplates();
    return () => {
      isMounted = false;
    };
  }, []);

  // Run initial generation on mount
  const handleGenerate = async () => {
    setGenLoading(true);
    try {
      const res = await api.syntheticLab.generateSamples({
        paradigm: genParadigm,
        domain: genDomain,
        complexity: genComplexity,
        count: genCount,
        include_edge_cases: genIncludeEdgeCases,
      });
      setGeneratedSamples(res.samples);
      // Auto-run pipeline on the new batch
      runFilterPipeline(res.samples, filterThresholds);
    } catch (err) {
      console.error('Generation failed', err);
    } finally {
      setGenLoading(false);
    }
  };

  // Run pipeline filtering
  const runFilterPipeline = async (
    samplesToFilter: SyntheticSample[],
    thresholds: FilterThresholdsInput
  ) => {
    if (samplesToFilter.length === 0) return;
    setFilterLoading(true);
    try {
      const res = await api.syntheticLab.filterSamples({
        samples: samplesToFilter,
        thresholds,
      });
      setPipelineResult(res);
    } catch (err) {
      console.error('Filtering pipeline failed', err);
    } finally {
      setFilterLoading(false);
    }
  };

  // Run Single Sample Inspection
  const handleInspect = async () => {
    setInspectLoading(true);
    try {
      const res = await api.syntheticLab.scoreSample({
        instruction: inspectInstr,
        response: inspectResp,
      });
      setInspectResult(res);
    } catch (err) {
      console.error('Inspection failed', err);
    } finally {
      setInspectLoading(false);
    }
  };

  // Export filtered samples
  const handleExport = async () => {
    if (!pipelineResult || pipelineResult.passed_samples.length === 0) return;
    setExportLoading(true);
    setExportMessage(null);
    try {
      const res = await api.syntheticLab.exportDataset({
        samples: pipelineResult.passed_samples,
        dataset_name: `synthetic_${genParadigm}_${genDomain}`,
        export_format: 'jsonl',
      });
      setExportMessage(res.message);
    } catch (err) {
      console.error('Export failed', err);
      setExportMessage('Export işlemi sırasında hata oluştu.');
    } finally {
      setExportLoading(false);
    }
  };

  // Ingest filtered samples directly to Dataset Compiler
  const handleIngestToDataset = async () => {
    if (!pipelineResult || pipelineResult.passed_samples.length === 0) return;
    setIngestLoading(true);
    setIngestError(null);
    try {
      const res = await api.syntheticLab.ingestToDataset({
        samples: pipelineResult.passed_samples,
        dataset_name: `synthetic_${genParadigm}_${genDomain}`,
        domain: genDomain,
        paradigm: genParadigm,
        training_allowed: true,
      });
      setIngestResult(res);
    } catch (err: any) {
      console.error('Ingest failed', err);
      setIngestError(err.response?.data?.detail || 'Dataset Compiler havuzuna aktarım başarısız oldu.');
    } finally {
      setIngestLoading(false);
    }
  };

  // Change preset profile
  const handleProfileSelect = (profKey: 'strict' | 'balanced' | 'lenient') => {
    setSelectedProfile(profKey);
    const prof = metadata.preset_profiles[profKey];
    if (prof) {
      setFilterThresholds(prof);
      if (generatedSamples.length > 0) {
        runFilterPipeline(generatedSamples, prof);
      }
    }
  };

  // Auto-run on mount
  useEffect(() => {
    handleGenerate();
    handleInspect();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 pb-16">
      {/* Header */}
      <header className="border-b border-slate-200 bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
            <div>
              <div className="flex items-center gap-2 mb-1">
                <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold bg-emerald-100 text-emerald-800">
                  <Sparkles className="w-3.5 h-3.5 mr-1 text-emerald-600" /> SYNTHETIC DATA & QUALITY FILTERING
                </span>
                <span className="text-xs text-slate-500 font-mono">v1.2.0 • Data-Centric AI</span>
              </div>
              <h1 className="text-2xl sm:text-3xl font-bold tracking-tight text-slate-900">
                Sentetik Veri Üretimi, Perplexity & Kalite Filtreleme Laboratuvarı
              </h1>
              <p className="text-sm text-slate-600 mt-1 max-w-3xl">
                Yüksek kaliteli SFT ve ön eğitim için sentetik talimat verisi üretin, dil modeli akıcılık ve 
                perplexity sınırlarını belirleyin ve çok aşamalı huni (pipeline funnel) ile verileri filtreleyin.
              </p>
            </div>

            {/* Quick Action Badges */}
            <div className="flex items-center gap-2 self-start md:self-auto">
              <Link
                href="/dataset-compiler"
                className="inline-flex items-center px-3 py-1.5 rounded-lg border border-slate-300 text-xs font-medium text-slate-700 hover:bg-slate-50 transition-colors"
              >
                Dataset Compiler <ArrowRight className="w-3.5 h-3.5 ml-1" />
              </Link>
              <Link
                href="/journey"
                className="inline-flex items-center px-3 py-1.5 rounded-lg bg-indigo-600 text-xs font-medium text-white hover:bg-indigo-700 transition-colors shadow-sm"
              >
                Öğrenme Yolu <Sparkles className="w-3.5 h-3.5 ml-1" />
              </Link>
            </div>
          </div>

          {/* Navigation Tabs */}
          <div className="flex items-center space-x-1 mt-6 border-b border-slate-200">
            <button
              id="tab-generator-btn"
              onClick={() => setActiveTab('generator')}
              className={`flex items-center gap-2 px-4 py-2.5 text-sm font-medium border-b-2 transition-all ${
                activeTab === 'generator'
                  ? 'border-indigo-600 text-indigo-600 bg-indigo-50/50 rounded-t-lg'
                  : 'border-transparent text-slate-600 hover:text-slate-900 hover:border-slate-300'
              }`}
            >
              <Cpu className="w-4 h-4" />
              Sentetik Veri Üreticisi
            </button>
            <button
              id="tab-pipeline-btn"
              onClick={() => setActiveTab('pipeline')}
              className={`flex items-center gap-2 px-4 py-2.5 text-sm font-medium border-b-2 transition-all ${
                activeTab === 'pipeline'
                  ? 'border-indigo-600 text-indigo-600 bg-indigo-50/50 rounded-t-lg'
                  : 'border-transparent text-slate-600 hover:text-slate-900 hover:border-slate-300'
              }`}
            >
              <Filter className="w-4 h-4" />
              Kalite Filtreleme & Huni Analizi
            </button>
            <button
              id="tab-inspector-btn"
              onClick={() => setActiveTab('inspector')}
              className={`flex items-center gap-2 px-4 py-2.5 text-sm font-medium border-b-2 transition-all ${
                activeTab === 'inspector'
                  ? 'border-indigo-600 text-indigo-600 bg-indigo-50/50 rounded-t-lg'
                  : 'border-transparent text-slate-600 hover:text-slate-900 hover:border-slate-300'
              }`}
            >
              <Activity className="w-4 h-4" />
              Tekil Örnek Teşhisi & Skorlayıcı
            </button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 mt-6">
        {/* ================================================================= */}
        {/* TAB 1: SYNTHETIC DATA GENERATOR                                   */}
        {/* ================================================================= */}
        {activeTab === 'generator' && (
          <div className="space-y-6">
            {/* Control Panel Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Paradigms Selection */}
              <div className="bg-white rounded-xl p-5 border border-slate-200 shadow-sm space-y-4">
                <div className="flex items-center justify-between">
                  <h3 className="text-sm font-semibold text-slate-900 flex items-center gap-2">
                    <Brain className="w-4 h-4 text-indigo-600" /> Üretim Paradigması
                  </h3>
                </div>

                <div className="space-y-2">
                  {metadata.paradigms.map((p) => (
                    <div
                      key={p.id}
                      onClick={() => setGenParadigm(p.id)}
                      className={`p-3 rounded-lg border cursor-pointer transition-all ${
                        genParadigm === p.id
                          ? 'border-indigo-600 bg-indigo-50/60 shadow-2xs'
                          : 'border-slate-200 hover:bg-slate-50'
                      }`}
                    >
                      <div className="flex items-center justify-between">
                        <span className="text-xs font-bold text-slate-900 flex items-center gap-1.5">
                          <span>{p.icon}</span> {p.name}
                        </span>
                        {genParadigm === p.id && (
                          <span className="w-2 h-2 rounded-full bg-indigo-600"></span>
                        )}
                      </div>
                      <p className="text-[11px] text-slate-500 mt-1 leading-relaxed">{p.description}</p>
                    </div>
                  ))}
                </div>
              </div>

              {/* Domain & Complexity Controls */}
              <div className="bg-white rounded-xl p-5 border border-slate-200 shadow-sm space-y-4">
                <div className="flex items-center justify-between">
                  <h3 className="text-sm font-semibold text-slate-900 flex items-center gap-2">
                    <Sliders className="w-4 h-4 text-emerald-600" /> Alan & Zorluk Ayarları
                  </h3>
                </div>

                <div className="space-y-3">
                  <div>
                    <label className="block text-xs font-medium text-slate-700 mb-1">
                      Hedef Bilgi Alanı (Domain)
                    </label>
                    <select
                      id="domain-select"
                      value={genDomain}
                      onChange={(e) => setGenDomain(e.target.value)}
                      className="w-full bg-slate-50 border border-slate-300 rounded-lg px-3 py-2 text-sm text-slate-900 focus:outline-none focus:ring-2 focus:ring-emerald-500"
                    >
                      {metadata.domains.map((d) => (
                        <option key={d.id} value={d.id}>
                          {d.name}
                        </option>
                      ))}
                    </select>
                  </div>

                  <div>
                    <label className="block text-xs font-medium text-slate-700 mb-1">
                      Zorluk Düzeyi
                    </label>
                    <div className="grid grid-cols-3 gap-2">
                      {metadata.complexities.map((comp) => (
                        <button
                          key={comp}
                          type="button"
                          onClick={() => setGenComplexity(comp)}
                          className={`py-1.5 px-2 rounded-lg text-xs font-medium border capitalize text-center transition-all ${
                            genComplexity === comp
                              ? 'bg-emerald-50 border-emerald-600 text-emerald-700 font-bold'
                              : 'border-slate-200 text-slate-600 hover:bg-slate-50'
                          }`}
                        >
                          {comp}
                        </button>
                      ))}
                    </div>
                  </div>

                  <div>
                    <div className="flex justify-between text-xs text-slate-600 mb-1">
                      <span>Örnek Sayısı:</span>
                      <span className="font-bold text-slate-900 font-mono">{genCount} adet</span>
                    </div>
                    <input
                      type="range"
                      min={1}
                      max={20}
                      step={1}
                      value={genCount}
                      onChange={(e) => setGenCount(Number(e.target.value))}
                      className="w-full accent-emerald-600"
                    />
                  </div>

                  <div className="pt-1">
                    <label className="flex items-center gap-2 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={genIncludeEdgeCases}
                        onChange={(e) => setGenIncludeEdgeCases(e.target.checked)}
                        className="w-4 h-4 rounded text-emerald-600 accent-emerald-600"
                      />
                      <span className="text-xs text-slate-700 font-medium">
                        Filtre testi için kasıtlı kusurlu uç örnekler ekle
                      </span>
                    </label>
                    <p className="text-[11px] text-slate-500 ml-6 mt-0.5">
                      Aşırı tekrar, PII sızıntısı ve yüksek perplexity içeren örnekler üreterek filtreleme hunisini sınar.
                    </p>
                  </div>
                </div>
              </div>

              {/* Action & Stats Card */}
              <div className="bg-white rounded-xl p-5 border border-slate-200 shadow-sm flex flex-col justify-between">
                <div>
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-sm font-semibold text-slate-900 flex items-center gap-2">
                      <Activity className="w-4 h-4 text-indigo-600" /> Üretim Özeti & Durum
                    </h3>
                  </div>

                  <div className="space-y-3">
                    <div className="grid grid-cols-2 gap-2 text-center">
                      <div className="bg-slate-50 p-3 rounded-lg border border-slate-100">
                        <p className="text-[11px] text-slate-500 font-medium">Havuzdaki Örnek</p>
                        <p className="text-xl font-bold text-slate-900 font-mono">
                          {generatedSamples.length} <span className="text-xs">adet</span>
                        </p>
                      </div>
                      <div className="bg-slate-50 p-3 rounded-lg border border-slate-100">
                        <p className="text-[11px] text-slate-500 font-medium">Ortalama Kalite</p>
                        <p className="text-xl font-bold text-indigo-600 font-mono">
                          {generatedSamples.length > 0
                            ? (
                                generatedSamples.reduce(
                                  (acc, s) => acc + (s.metrics.composite_score || 0),
                                  0
                                ) / generatedSamples.length
                              ).toFixed(1)
                            : 0}
                          /100
                        </p>
                      </div>
                    </div>

                    <div className="p-3 bg-slate-50 rounded-lg border border-slate-200 text-xs text-slate-600 space-y-1">
                      <p className="font-semibold text-slate-900">Pedagojik Not:</p>
                      <p className="text-[11px] leading-relaxed">
                        Sentetik veriler üretildikten sonra doğrudan eğitime sokulmamalıdır. 
                        <strong>Perplexity Thresholding</strong> ve <strong>Repetition Filtering</strong> aşamalarından geçirilerek model çökmesi (model collapse) engellenmelidir.
                      </p>
                    </div>
                  </div>
                </div>

                <button
                  id="generate-synthetic-btn"
                  onClick={handleGenerate}
                  disabled={genLoading}
                  className="w-full mt-4 flex items-center justify-center gap-2 py-2.5 px-4 rounded-lg bg-indigo-600 text-white text-xs font-semibold hover:bg-indigo-700 transition-colors shadow-sm disabled:opacity-50"
                >
                  <RefreshCw className={`w-3.5 h-3.5 ${genLoading ? 'animate-spin' : ''}`} />
                  {genLoading ? 'Veri Üretiliyor...' : 'Sentetik Veri Üret'}
                </button>
              </div>
            </div>

            {/* Generated Samples List */}
            <div className="bg-white rounded-xl p-6 border border-slate-200 shadow-sm space-y-4">
              <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
                <div>
                  <h3 className="text-base font-bold text-slate-900 flex items-center gap-2">
                    <Database className="w-5 h-5 text-indigo-600" /> Üretilen Sentetik Havuz ({generatedSamples.length} Örnek)
                  </h3>
                  <p className="text-xs text-slate-500">
                    Üretilen talimat-yanıt çiftlerini ve kalite puanlama skorlarını inceleyin.
                  </p>
                </div>
                <button
                  onClick={() => setActiveTab('pipeline')}
                  className="inline-flex items-center gap-1.5 text-xs font-semibold text-indigo-600 hover:text-indigo-800"
                >
                  Filtreleme Hunisine Git <ArrowRight className="w-3.5 h-3.5" />
                </button>
              </div>

              <div className="space-y-3">
                {generatedSamples.map((sample) => {
                  const isAccept = sample.metrics.verdict === 'ACCEPT';
                  const isRevision = sample.metrics.verdict === 'NEEDS_REVISION';

                  return (
                    <div
                      key={sample.id}
                      className="border border-slate-200 rounded-xl p-4 bg-slate-50/50 hover:bg-white transition-all space-y-3"
                    >
                      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
                        <div className="flex items-center gap-2">
                          <span className="font-mono text-xs font-bold text-slate-800 bg-slate-200/70 px-2 py-0.5 rounded">
                            {sample.id}
                          </span>
                          <span className="text-[11px] bg-slate-100 text-slate-600 px-2 py-0.5 rounded capitalize">
                            {sample.domain.replace('_', ' ')}
                          </span>
                          <span className="text-[11px] bg-indigo-50 text-indigo-700 px-2 py-0.5 rounded capitalize">
                            {sample.paradigm.replace('_', ' ')}
                          </span>
                        </div>

                        <div className="flex items-center gap-2">
                          <span
                            className={`text-xs px-2.5 py-0.5 rounded-full font-bold uppercase ${
                              isAccept
                                ? 'bg-emerald-100 text-emerald-800 border border-emerald-300'
                                : isRevision
                                ? 'bg-amber-100 text-amber-800 border border-amber-300'
                                : 'bg-rose-100 text-rose-800 border border-rose-300'
                            }`}
                          >
                            Skor: {sample.metrics.composite_score?.toFixed(1)} ({sample.metrics.verdict})
                          </span>
                          <span className="text-xs text-slate-500 font-mono">
                            PPL: {sample.metrics.perplexity?.toFixed(1)}
                          </span>
                        </div>
                      </div>

                      <div className="space-y-1.5 text-xs">
                        <div className="font-semibold text-slate-900 bg-white p-2.5 rounded-lg border border-slate-200">
                          <span className="text-indigo-600 font-bold mr-1">Talimat:</span> {sample.instruction}
                        </div>
                        {sample.input_context && (
                          <div className="text-slate-600 italic bg-slate-100/70 p-2 rounded-lg text-[11px]">
                            <span className="font-semibold not-italic">Bağlam:</span> {sample.input_context}
                          </div>
                        )}
                        <div className="text-slate-700 bg-white p-3 rounded-lg border border-slate-200 whitespace-pre-wrap leading-relaxed font-sans">
                          {sample.response}
                        </div>
                      </div>

                      {/* Diagnostic pills */}
                      <div className="flex flex-wrap items-center gap-3 pt-1 text-[11px] text-slate-500 border-t border-slate-200/60">
                        <span>Kelime: <strong className="text-slate-700">{sample.metrics.word_count}</strong></span>
                        <span>2-Gram Tekrar: <strong className="text-slate-700">%{(sample.metrics.repetition_ratio_2g * 100).toFixed(1)}</strong></span>
                        <span>Çeşitlilik (TTR): <strong className="text-slate-700">{sample.metrics.lexical_diversity_ttr?.toFixed(2)}</strong></span>
                        <span>PII: <strong className={sample.metrics.pii_count > 0 ? 'text-rose-600 font-bold' : 'text-emerald-600 font-semibold'}>{sample.metrics.pii_count} adet</strong></span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        )}

        {/* ================================================================= */}
        {/* TAB 2: FILTER PIPELINE & FUNNEL ANALYTICS                          */}
        {/* ================================================================= */}
        {activeTab === 'pipeline' && (
          <div className="space-y-6">
            {/* Threshold Controls Card */}
            <div className="bg-white rounded-xl p-5 border border-slate-200 shadow-sm space-y-4">
              <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
                <div>
                  <h3 className="text-base font-bold text-slate-900 flex items-center gap-2">
                    <Filter className="w-5 h-5 text-indigo-600" /> Çok Aşamalı Kalite Filtreleme Parametreleri
                  </h3>
                  <p className="text-xs text-slate-500">
                    Perplexity, uzunluk, n-gram döngüleri ve PII eşiklerini ayarlayarak eleme hunisini yönetin.
                  </p>
                </div>

                {/* Preset Profile Selector */}
                <div className="flex items-center gap-1.5 bg-slate-100 p-1 rounded-lg text-xs font-medium">
                  {(['strict', 'balanced', 'lenient'] as const).map((profKey) => (
                    <button
                      key={profKey}
                      onClick={() => handleProfileSelect(profKey)}
                      className={`px-3 py-1 rounded-md capitalize transition-all ${
                        selectedProfile === profKey
                          ? 'bg-white shadow text-indigo-600 font-bold'
                          : 'text-slate-600 hover:text-slate-900'
                      }`}
                    >
                      {profKey === 'strict' ? 'Katı' : profKey === 'balanced' ? 'Dengeli' : 'Esnek'}
                    </button>
                  ))}
                </div>
              </div>

              {/* Sliders Grid */}
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 pt-2">
                <div>
                  <div className="flex justify-between text-xs text-slate-600 mb-1">
                    <span>Min Perplexity:</span>
                    <span className="font-bold text-indigo-600 font-mono">{filterThresholds.min_perplexity}</span>
                  </div>
                  <input
                    type="range"
                    min={1.0}
                    max={20.0}
                    step={0.5}
                    value={filterThresholds.min_perplexity}
                    onChange={(e) =>
                      setFilterThresholds({ ...filterThresholds, min_perplexity: Number(e.target.value) })
                    }
                    className="w-full accent-indigo-600"
                  />
                  <p className="text-[10px] text-slate-400 mt-0.5">Altındaki modeller çökme/aşırı tekrardır.</p>
                </div>

                <div>
                  <div className="flex justify-between text-xs text-slate-600 mb-1">
                    <span>Max Perplexity:</span>
                    <span className="font-bold text-indigo-600 font-mono">{filterThresholds.max_perplexity}</span>
                  </div>
                  <input
                    type="range"
                    min={40.0}
                    max={250.0}
                    step={5.0}
                    value={filterThresholds.max_perplexity}
                    onChange={(e) =>
                      setFilterThresholds({ ...filterThresholds, max_perplexity: Number(e.target.value) })
                    }
                    className="w-full accent-indigo-600"
                  />
                  <p className="text-[10px] text-slate-400 mt-0.5">Üstündeki metinler halüsinasyon/tutarsızlıktır.</p>
                </div>

                <div>
                  <div className="flex justify-between text-xs text-slate-600 mb-1">
                    <span>Min Kalite Puanı:</span>
                    <span className="font-bold text-emerald-600 font-mono">{filterThresholds.min_quality_score}/100</span>
                  </div>
                  <input
                    type="range"
                    min={30.0}
                    max={90.0}
                    step={2.0}
                    value={filterThresholds.min_quality_score}
                    onChange={(e) =>
                      setFilterThresholds({ ...filterThresholds, min_quality_score: Number(e.target.value) })
                    }
                    className="w-full accent-emerald-600"
                  />
                  <p className="text-[10px] text-slate-400 mt-0.5">Kompozit rubrik puanı eşiği.</p>
                </div>

                <div>
                  <div className="flex justify-between text-xs text-slate-600 mb-1">
                    <span>Max 2-Gram Tekrar:</span>
                    <span className="font-bold text-amber-600 font-mono">
                      %{(filterThresholds.max_repetition_ratio * 100).toFixed(0)}
                    </span>
                  </div>
                  <input
                    type="range"
                    min={0.05}
                    max={0.4}
                    step={0.01}
                    value={filterThresholds.max_repetition_ratio}
                    onChange={(e) =>
                      setFilterThresholds({
                        ...filterThresholds,
                        max_repetition_ratio: Number(e.target.value),
                      })
                    }
                    className="w-full accent-amber-600"
                  />
                  <p className="text-[10px] text-slate-400 mt-0.5">Döngüye giren metinleri eler.</p>
                </div>
              </div>

              {/* Second Row Controls */}
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 pt-2 border-t border-slate-100">
                <div>
                  <label className="block text-xs font-medium text-slate-700 mb-1">
                    PII & KVKK Politikası
                  </label>
                  <select
                    value={filterThresholds.pii_action}
                    onChange={(e) =>
                      setFilterThresholds({ ...filterThresholds, pii_action: e.target.value })
                    }
                    className="w-full bg-slate-50 border border-slate-300 rounded-lg px-3 py-1.5 text-xs text-slate-900 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                  >
                    <option value="reject">Reddet (PII içeren örneği tamamen ele)</option>
                    <option value="mask">Maskele (TC ve telefonları yıldızla sansürle)</option>
                    <option value="allow">İzin Ver (Dikkate alma)</option>
                  </select>
                </div>

                <div>
                  <div className="flex justify-between text-xs text-slate-600 mb-1">
                    <span>Min Kelime Sayısı:</span>
                    <span className="font-bold text-slate-900 font-mono">{filterThresholds.min_words}</span>
                  </div>
                  <input
                    type="range"
                    min={5}
                    max={50}
                    step={1}
                    value={filterThresholds.min_words}
                    onChange={(e) =>
                      setFilterThresholds({ ...filterThresholds, min_words: Number(e.target.value) })
                    }
                    className="w-full accent-indigo-600"
                  />
                </div>

                <div className="flex items-end">
                  <button
                    id="apply-filter-btn"
                    onClick={() => runFilterPipeline(generatedSamples, filterThresholds)}
                    disabled={filterLoading || generatedSamples.length === 0}
                    className="w-full flex items-center justify-center gap-2 py-2 px-4 rounded-lg bg-slate-900 text-white text-xs font-semibold hover:bg-slate-800 transition-colors shadow-sm disabled:opacity-50"
                  >
                    <RefreshCw className={`w-3.5 h-3.5 ${filterLoading ? 'animate-spin' : ''}`} />
                    Filtreleri Havuza Uygula
                  </button>
                </div>
              </div>
            </div>

            {/* Funnel Metrics & Stage Visualizer */}
            {pipelineResult && (
              <div className="bg-white rounded-xl p-6 border border-slate-200 shadow-sm space-y-6">
                <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
                  <div>
                    <h3 className="text-base font-bold text-slate-900 flex items-center gap-2">
                      <Layers className="w-5 h-5 text-indigo-600" /> Çok Aşamalı Eleme Hunisi (Pipeline Funnel)
                    </h3>
                    <p className="text-xs text-slate-500">
                      Her bir eleme aşamasında elenen ve kalan örnek sayıları.
                    </p>
                  </div>

                  <div className="flex items-center gap-3">
                    <div className="text-right">
                      <p className="text-[11px] text-slate-500">Nihai Verim (Yield)</p>
                      <p className="text-lg font-bold text-emerald-600 font-mono">
                        %{pipelineResult.final_yield_pct}
                      </p>
                    </div>
                    <div className="text-right border-l pl-3 border-slate-200">
                      <p className="text-[11px] text-slate-500">Kalite İyileşmesi</p>
                      <p className="text-lg font-bold text-indigo-600 font-mono">
                        {pipelineResult.avg_initial_quality} ➔ {pipelineResult.avg_final_quality}
                      </p>
                    </div>
                  </div>
                </div>

                {/* Visual Funnel Bars */}
                <div className="space-y-3">
                  {pipelineResult.funnel_stages.map((stage, idx) => {
                    const widthPct = Math.max(8, stage.retention_rate);
                    return (
                      <div key={idx} className="space-y-1">
                        <div className="flex justify-between text-xs font-medium text-slate-700">
                          <span className="flex items-center gap-2">
                            <span className="w-4 h-4 rounded-full bg-slate-100 text-slate-600 flex items-center justify-center text-[10px] font-bold">
                              {idx + 1}
                            </span>
                            {stage.stage_name}
                          </span>
                          <span className="font-mono text-slate-600">
                            {stage.passed_count} / {stage.input_count} kaldı ({stage.retention_rate}%)
                          </span>
                        </div>
                        <div className="w-full bg-slate-100 h-3 rounded-full overflow-hidden flex">
                          <div
                            style={{ width: `${widthPct}%` }}
                            className="bg-indigo-600 h-full rounded-full transition-all"
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>

                {/* Filtered Dataset Section */}
                <div className="border-t border-slate-100 pt-5 space-y-4">
                  <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
                    <div className="flex items-center gap-1 bg-slate-100 p-1 rounded-lg text-xs font-medium">
                      <button
                        onClick={() => setViewFilterTab('passed')}
                        className={`px-3 py-1 rounded-md transition-all ${
                          viewFilterTab === 'passed'
                            ? 'bg-white shadow text-emerald-700 font-bold'
                            : 'text-slate-600 hover:text-slate-900'
                        }`}
                      >
                        Kabul Edilenler ({pipelineResult.passed_samples.length})
                      </button>
                      <button
                        onClick={() => setViewFilterTab('rejected')}
                        className={`px-3 py-1 rounded-md transition-all ${
                          viewFilterTab === 'rejected'
                            ? 'bg-white shadow text-rose-700 font-bold'
                            : 'text-slate-600 hover:text-slate-900'
                        }`}
                      >
                        Elenen Örnekler ({pipelineResult.rejected_samples.length})
                      </button>
                    </div>

                    {/* Export & Ingest Actions */}
                    <div className="flex flex-wrap items-center gap-2">
                      <button
                        id="export-synthetic-btn"
                        onClick={handleExport}
                        disabled={exportLoading || pipelineResult.passed_samples.length === 0}
                        className="inline-flex items-center gap-1.5 px-3.5 py-2 rounded-lg bg-emerald-600 text-white text-xs font-semibold hover:bg-emerald-700 transition-colors shadow-sm disabled:opacity-50"
                      >
                        <Download className="w-3.5 h-3.5" />
                        {exportLoading ? 'Dışa Aktarılıyor...' : 'Küratörlü JSONL İndir'}
                      </button>

                      <button
                        id="ingest-synthetic-btn"
                        onClick={handleIngestToDataset}
                        disabled={ingestLoading || pipelineResult.passed_samples.length === 0}
                        className="inline-flex items-center gap-1.5 px-3.5 py-2 rounded-lg bg-indigo-600 text-white text-xs font-semibold hover:bg-indigo-700 transition-colors shadow-sm disabled:opacity-50"
                      >
                        <Database className="w-3.5 h-3.5" />
                        {ingestLoading ? 'Aktarılıyor...' : "Dataset Compiler'a Aktar"}
                      </button>
                    </div>
                  </div>

                  {exportMessage && (
                    <div className="p-3 bg-emerald-50 border border-emerald-200 rounded-lg text-xs text-emerald-800 flex items-center gap-2">
                      <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0" />
                      <span>{exportMessage}</span>
                    </div>
                  )}

                  {ingestError && (
                    <div className="p-3 bg-rose-50 border border-rose-200 rounded-lg text-xs text-rose-800 flex items-center gap-2">
                      <AlertTriangle className="w-4 h-4 text-rose-600 shrink-0" />
                      <span>{ingestError}</span>
                    </div>
                  )}

                  {ingestResult && (
                    <div className="p-4 bg-indigo-50 border border-indigo-200 rounded-xl text-xs space-y-2.5">
                      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
                        <div className="flex items-center gap-2">
                          <CheckCircle2 className="w-4 h-4 text-indigo-600 shrink-0" />
                          <span className="font-bold text-indigo-950">{ingestResult.message}</span>
                        </div>
                        <span className="font-mono bg-indigo-100 text-indigo-800 px-2 py-0.5 rounded font-semibold text-[11px] self-start sm:self-auto">
                          File ID: {ingestResult.file_id}
                        </span>
                      </div>
                      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 pt-2 border-t border-indigo-100">
                        <span className="text-indigo-800">
                          {ingestResult.document_count} doküman derleme ve LLM eğitimi için kullanıma hazır (Ort. Kalite Skoru: %{ingestResult.avg_quality_score}).
                        </span>
                        <Link
                          id="goto-compiler-btn"
                          href={`/dataset-compiler?file_id=${ingestResult.file_id}&dataset_name=${ingestResult.dataset_name}`}
                          className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-indigo-600 text-white font-semibold hover:bg-indigo-700 transition-colors shadow-sm text-xs self-start sm:self-auto shrink-0"
                        >
                          Dataset Compiler&apos;da Derle <ArrowRight className="w-3.5 h-3.5" />
                        </Link>
                      </div>
                    </div>
                  )}

                  {/* Sample display */}
                  <div className="space-y-3">
                    {viewFilterTab === 'passed' ? (
                      pipelineResult.passed_samples.map((s) => (
                        <div
                          key={s.id}
                          className="p-4 rounded-xl border border-emerald-200 bg-emerald-50/20 space-y-2 text-xs"
                        >
                          <div className="flex justify-between items-center">
                            <span className="font-mono font-bold text-slate-800">{s.id}</span>
                            <span className="px-2 py-0.5 rounded-full bg-emerald-100 text-emerald-800 font-bold">
                              Kalite Skoru: {s.metrics.composite_score}
                            </span>
                          </div>
                          <p className="font-semibold text-slate-900">{s.instruction}</p>
                          <p className="text-slate-700 whitespace-pre-wrap leading-relaxed">{s.response}</p>
                        </div>
                      ))
                    ) : (
                      pipelineResult.rejected_samples.map((s) => (
                        <div
                          key={s.id}
                          className="p-4 rounded-xl border border-rose-200 bg-rose-50/20 space-y-2 text-xs"
                        >
                          <div className="flex justify-between items-center">
                            <span className="font-mono font-bold text-slate-800">{s.id}</span>
                            <span className="px-2 py-0.5 rounded-full bg-rose-100 text-rose-800 font-bold">
                              {s.rejection_stage || 'Elendi'}
                            </span>
                          </div>
                          <div className="p-2 bg-rose-100/60 rounded text-[11px] text-rose-900 font-medium">
                            Sebep: {s.rejection_reason}
                          </div>
                          <p className="font-semibold text-slate-900">{s.instruction}</p>
                          <p className="text-slate-600 line-clamp-3 leading-relaxed">{s.response}</p>
                        </div>
                      ))
                    )}
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* ================================================================= */}
        {/* TAB 3: SINGLE SAMPLE INSPECTOR & DIAGNOSTIC SCORER                */}
        {/* ================================================================= */}
        {activeTab === 'inspector' && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Input Area */}
              <div className="bg-white rounded-xl p-5 border border-slate-200 shadow-sm space-y-4">
                <div className="flex items-center justify-between">
                  <h3 className="text-sm font-semibold text-slate-900 flex items-center gap-2">
                    <Code className="w-4 h-4 text-indigo-600" /> Özel Örnek Girdisi
                  </h3>
                  <span className="text-xs text-slate-500">Canlı Puanlayıcı</span>
                </div>

                {/* Preset Scenarios */}
                <div>
                  <label className="block text-xs font-medium text-slate-700 mb-1">
                    Örnek Senaryo Yükle
                  </label>
                  <div className="grid grid-cols-2 gap-1.5">
                    {INSPECTOR_EXAMPLES.map((ex, idx) => (
                      <button
                        key={idx}
                        type="button"
                        onClick={() => {
                          setInspectInstr(ex.instruction);
                          setInspectResp(ex.response);
                        }}
                        className="text-left p-2 rounded-lg text-[11px] font-medium border border-slate-200 hover:bg-slate-50 text-slate-700 truncate"
                      >
                        {ex.title}
                      </button>
                    ))}
                  </div>
                </div>

                <div>
                  <label className="block text-xs font-medium text-slate-700 mb-1">
                    Talimat / Soru (Instruction)
                  </label>
                  <textarea
                    id="inspect-instruction-input"
                    rows={2}
                    value={inspectInstr}
                    onChange={(e) => setInspectInstr(e.target.value)}
                    className="w-full bg-slate-50 border border-slate-300 rounded-lg p-2.5 text-xs text-slate-900 focus:outline-none focus:ring-2 focus:ring-indigo-500 font-sans"
                    placeholder="Talimat metnini yazın..."
                  />
                </div>

                <div>
                  <label className="block text-xs font-medium text-slate-700 mb-1">
                    Model Yanıtı (Response)
                  </label>
                  <textarea
                    id="inspect-response-input"
                    rows={6}
                    value={inspectResp}
                    onChange={(e) => setInspectResp(e.target.value)}
                    className="w-full bg-slate-50 border border-slate-300 rounded-lg p-2.5 text-xs text-slate-900 focus:outline-none focus:ring-2 focus:ring-indigo-500 font-sans"
                    placeholder="Değerlendirilecek yanıt metnini yazın..."
                  />
                </div>

                <button
                  id="run-inspect-btn"
                  onClick={handleInspect}
                  disabled={inspectLoading || !inspectResp.trim()}
                  className="w-full flex items-center justify-center gap-2 py-2.5 px-4 rounded-lg bg-indigo-600 text-white text-xs font-semibold hover:bg-indigo-700 transition-colors shadow-sm disabled:opacity-50"
                >
                  <RefreshCw className={`w-3.5 h-3.5 ${inspectLoading ? 'animate-spin' : ''}`} />
                  {inspectLoading ? 'Puanlanıyor...' : 'Metni Teşhis Et & Puanla'}
                </button>
              </div>

              {/* Diagnostic Output Card */}
              <div className="bg-white rounded-xl p-5 border border-slate-200 shadow-sm space-y-4">
                <div className="flex items-center justify-between">
                  <h3 className="text-sm font-semibold text-slate-900 flex items-center gap-2">
                    <Activity className="w-4 h-4 text-emerald-600" /> Detaylı Kalite Teşhisi
                  </h3>
                  {inspectResult && (
                    <span
                      className={`text-xs px-2.5 py-0.5 rounded-full font-bold uppercase ${
                        inspectResult.verdict === 'ACCEPT'
                          ? 'bg-emerald-100 text-emerald-800 border border-emerald-300'
                          : inspectResult.verdict === 'NEEDS_REVISION'
                          ? 'bg-amber-100 text-amber-800 border border-amber-300'
                          : 'bg-rose-100 text-rose-800 border border-rose-300'
                      }`}
                    >
                      {inspectResult.verdict}
                    </span>
                  )}
                </div>

                {inspectResult ? (
                  <div className="space-y-4">
                    {/* Score summary */}
                    <div className="text-center bg-slate-50 p-4 rounded-xl border border-slate-200">
                      <p className="text-xs text-slate-500 font-medium">Toplam Kalite Skoru</p>
                      <p className="text-4xl font-extrabold text-slate-900 font-mono mt-1">
                        {inspectResult.composite_score}{' '}
                        <span className="text-sm font-semibold text-slate-500">/ 100</span>
                      </p>
                      <div className="w-full bg-slate-200 h-2.5 rounded-full mt-3 overflow-hidden">
                        <div
                          style={{ width: `${inspectResult.composite_score}%` }}
                          className={`h-full transition-all ${
                            inspectResult.composite_score >= 70
                              ? 'bg-emerald-500'
                              : inspectResult.composite_score >= 50
                              ? 'bg-amber-500'
                              : 'bg-rose-500'
                          }`}
                        />
                      </div>
                    </div>

                    {/* Subscore meters */}
                    <div className="grid grid-cols-2 sm:grid-cols-3 gap-2.5 text-xs">
                      <div className="bg-slate-50 p-2.5 rounded-lg border border-slate-100">
                        <span className="text-[11px] text-slate-500">Perplexity (Akıcılık):</span>
                        <p className="font-bold text-slate-900 font-mono text-sm">
                          {inspectResult.perplexity?.toFixed(1)}
                        </p>
                      </div>
                      <div className="bg-slate-50 p-2.5 rounded-lg border border-slate-100">
                        <span className="text-[11px] text-slate-500">2-Gram Tekrar:</span>
                        <p className="font-bold text-slate-900 font-mono text-sm">
                          %{((inspectResult.repetition_ratio_2g || 0) * 100).toFixed(1)}
                        </p>
                      </div>
                      <div className="bg-slate-50 p-2.5 rounded-lg border border-slate-100">
                        <span className="text-[11px] text-slate-500">Çeşitlilik (TTR):</span>
                        <p className="font-bold text-slate-900 font-mono text-sm">
                          {inspectResult.lexical_diversity_ttr?.toFixed(2)}
                        </p>
                      </div>
                      <div className="bg-slate-50 p-2.5 rounded-lg border border-slate-100">
                        <span className="text-[11px] text-slate-500">Kelime Sayısı:</span>
                        <p className="font-bold text-slate-900 font-mono text-sm">
                          {inspectResult.word_count}
                        </p>
                      </div>
                      <div className="bg-slate-50 p-2.5 rounded-lg border border-slate-100">
                        <span className="text-[11px] text-slate-500">PII Varlığı:</span>
                        <p
                          className={`font-bold font-mono text-sm ${
                            inspectResult.pii_count > 0 ? 'text-rose-600' : 'text-emerald-600'
                          }`}
                        >
                          {inspectResult.pii_count} adet
                        </p>
                      </div>
                      <div className="bg-slate-50 p-2.5 rounded-lg border border-slate-100">
                        <span className="text-[11px] text-slate-500">Biçimlendirme:</span>
                        <p className="font-bold text-slate-900 font-mono text-sm">
                          {inspectResult.subscores?.formatting}/100
                        </p>
                      </div>
                    </div>

                    {/* Verdict Reasons */}
                    {inspectResult.verdict_reasons && inspectResult.verdict_reasons.length > 0 && (
                      <div className="p-3 bg-amber-50 rounded-lg border border-amber-200 text-xs text-amber-900 space-y-1">
                        <p className="font-semibold flex items-center gap-1.5">
                          <AlertTriangle className="w-3.5 h-3.5 text-amber-600" /> İyileştirme / Eleme Gerekçeleri:
                        </p>
                        <ul className="list-disc list-inside space-y-0.5 text-[11px] text-amber-800">
                          {inspectResult.verdict_reasons.map((r, idx) => (
                            <li key={idx}>{r}</li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="py-12 text-center text-slate-400 text-xs">
                    Metni değerlendirmek için butona tıklayın...
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
