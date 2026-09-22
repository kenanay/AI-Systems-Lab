'use client';

import React, { useState, useEffect } from 'react';
import Link from 'next/link';
import {
  Layers,
  Search,
  MessageSquare,
  Sparkles,
  Sliders,
  CheckCircle2,
  BookOpen,
  ArrowRight,
  RefreshCw,
  Cpu,
  Zap,
  Info,
  ExternalLink,
  ChevronDown,
  ChevronUp,
  FileText,
  ShieldCheck,
  Flame,
  HelpCircle
} from 'lucide-react';
import {
  ragApi,
  ChunkItem,
  SearchResultItem,
  CitationItem,
  CollectionInfo
} from '@/lib/api';

const SAMPLE_PRESETS = [
  {
    name: '🤖 Yapay Zeka & Transformer',
    text: `Transformer mimarisi, 2017 yılında 'Attention Is All You Need' makalesiyle duyurulmuştur. Geleneksel yinelemeli sinir ağları (RNN) ve LSTM yapılarının aksine ardışık hesaplama darboğazını ortadan kaldırarak tam paralel matris işlemlerine olanak sağlar.

Mimarinin temel taşı olan Multi-Head Attention mekanizması, Query (Q), Key (K) ve Value (V) projeksiyonları üzerinden çalışır. Dikkat ağırlıkları Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V formülü ile hesaplanır. Her bir attention başlığı cümlenin farklı semantik ilişkilerine (özne-yüklem uyumu, zamir gönderimi, zamansal bağlam) odaklanır.

Autoregressive üretim sırasında çıkarım hızını optimize etmek amacıyla KV Cache tekniği kullanılır. Her yeni token üretiminde geçmiş adımların Key ve Value tensörleri yeniden hesaplanmaz, GPU belleğinde saklanır. Bu sayede çıkarım süresi O(N^2) karmaşıklıktan O(N) seviyesine indirgenir.`
  },
  {
    name: '⚖️ Hukuk & KVKK Mevzuatı',
    text: `6698 sayılı Kişisel Verilerin Korunması Kanunu (KVKK), kişisel verilerin işlenmesinde başta özel hayatın gizliliği olmak üzere kişilerin temel hak ve özgürlüklerini korumak amacıyla yürürlüğe girmiştir.

Kanun kapsamında 'Kişisel Veri', kimliği belirli veya belirlenebilir gerçek kişiye ilişkin her türlü bilgiyi ifade eder. TC kimlik numarası, ad-soyad, telefon, e-posta, araç plakası ve biyometrik veriler bu kapsamdadır.

Veri sorumlusu, kişisel verilerin hukuka aykırı olarak işlenmesini ve erişilmesini önlemek, muhafazasını sağlamak amacıyla uygun güvenlik düzeyini temin etmeye yönelik gerekli her türlü teknik ve idari tedbiri almak zorundadır. Yapay zeka sistemlerinde kullanılan eğitim verilerinde yer alan PII (Personally Identifiable Information) unsurlarının anonimleştirilmesi veya maskelenmesi yasal bir yükümlülüktür.`
  },
  {
    name: '🧬 Biyoinformatik & Genetik',
    text: `Biyoinformatik, biyolojik verilerin bilgisayar bilimi, istatistik ve yapay zeka algoritmalarıyla analiz edilmesini konu alan disiplinlerarası bir alandır.

DNA dizileme teknolojilerinin gelişmesiyle birlikte insan genomundaki 3 milyar baz çifti dijital ortama aktarılmıştır. Biyolojik sekans analizinde FASTA ve FASTQ formatları temel standartlardır.

Protein yapılarının tahmininde AlphaFold ve ESM-Fold gibi derin öğrenme mimarileri, amino asit diziliminden 3 boyutlu atomik koordinatları yüksek doğrulukla modelleyebilmektedir. İlaç keşfi süreçlerinde hedef protein-ligand bağlanma afinitesini hesaplamak için moleküler yerleştirme (docking) ve graf sinir ağları (GNN) yaygın olarak kullanılmaktadır.`
  }
];

const PRESET_QUESTIONS = [
  'Transformer mimarisinde Attention formülü nedir?',
  'KV Cache çıkarım hızını neden ve nasıl artırır?',
  'LoRA düşük rankli ince ayarı bellek tasarrufunu nasıl sağlar?',
  'KVKK kapsamında veri sorumlusunun PII yükümlülükleri nelerdir?',
  'BPE tokenizasyonunun Türkçe için avantajı nedir?'
];

const CHUNK_COLORS = [
  'bg-blue-50 border-blue-200 text-blue-900',
  'bg-purple-50 border-purple-200 text-purple-900',
  'bg-emerald-50 border-emerald-200 text-emerald-900',
  'bg-amber-50 border-amber-200 text-amber-900',
  'bg-indigo-50 border-indigo-200 text-indigo-900',
  'bg-rose-50 border-rose-200 text-rose-900',
];

export default function RAGLabPage() {
  const [activeTab, setActiveTab] = useState<'chunking' | 'retrieval' | 'qa'>('qa');

  // --- Chunking State ---
  const [rawText, setRawText] = useState(SAMPLE_PRESETS[0].text);
  const [chunkStrategy, setChunkStrategy] = useState<'recursive' | 'sentence' | 'fixed'>('recursive');
  const [chunkSize, setChunkSize] = useState<number>(350);
  const [chunkOverlap, setChunkOverlap] = useState<number>(70);
  const [chunks, setChunks] = useState<ChunkItem[]>([]);
  const [isChunking, setIsChunking] = useState<boolean>(false);
  const [selectedChunkIdx, setSelectedChunkIdx] = useState<number | null>(null);

  // --- Retrieval State ---
  const [searchQuery, setSearchQuery] = useState('Attention formülü ve KV Cache');
  const [searchMode, setSearchMode] = useState<'hybrid' | 'dense' | 'sparse'>('hybrid');
  const [topK, setTopK] = useState<number>(4);
  const [alpha, setAlpha] = useState<number>(0.5);
  const [searchResults, setSearchResults] = useState<SearchResultItem[]>([]);
  const [searchLatency, setSearchLatency] = useState<number | null>(null);
  const [isSearching, setIsSearching] = useState<boolean>(false);
  const [collections, setCollections] = useState<CollectionInfo[]>([]);

  // --- Q&A State ---
  const [qaQuestion, setQaQuestion] = useState('Transformer mimarisinde Attention formülü nedir ve KV Cache nasıl çalışır?');
  const [qaAnswer, setQaAnswer] = useState<string>('');
  const [qaCitations, setQaCitations] = useState<CitationItem[]>([]);
  const [qaRetrievedChunks, setQaRetrievedChunks] = useState<SearchResultItem[]>([]);
  const [qaStats, setQaStats] = useState<Record<string, any>>({});
  const [qaPromptUsed, setQaPromptUsed] = useState<string>('');
  const [showPromptDetails, setShowPromptDetails] = useState<boolean>(false);
  const [isAnswering, setIsAnswering] = useState<boolean>(false);

  // Load Collections
  useEffect(() => {
    const fetchCollections = async () => {
      try {
        const data = await ragApi.collections();
        setCollections(data);
      } catch (err) {
        console.error('Koleksiyonlar yüklenemedi:', err);
      }
    };
    fetchCollections();
  }, []);

  // Run initial chunking
  useEffect(() => {
    handleRunChunking();
  }, []);

  // Handlers
  const handleRunChunking = async () => {
    if (!rawText.trim()) return;
    setIsChunking(true);
    try {
      const res = await ragApi.chunk({
        text: rawText,
        strategy: chunkStrategy,
        chunk_size: chunkSize,
        chunk_overlap: chunkOverlap
      });
      setChunks(res.chunks);
    } catch (err) {
      console.error('Chunking hatası:', err);
    } finally {
      setIsChunking(false);
    }
  };

  const handleRunSearch = async () => {
    if (!searchQuery.trim()) return;
    setIsSearching(true);
    try {
      const res = await ragApi.search({
        query: searchQuery,
        collection_name: 'default',
        mode: searchMode,
        top_k: topK,
        alpha: alpha
      });
      setSearchResults(res.results);
      setSearchLatency(res.latency_ms);
    } catch (err) {
      console.error('Arama hatası:', err);
    } finally {
      setIsSearching(false);
    }
  };

  const handleRunQA = async () => {
    if (!qaQuestion.trim()) return;
    setIsAnswering(true);
    try {
      const res = await ragApi.query({
        question: qaQuestion,
        collection_name: 'default',
        retrieval_mode: 'hybrid',
        top_k: 3,
        alpha: 0.5
      });
      setQaAnswer(res.answer);
      setQaCitations(res.citations);
      setQaRetrievedChunks(res.retrieved_chunks);
      setQaStats(res.stats);
      setQaPromptUsed(res.prompt_used);
    } catch (err) {
      console.error('RAG Soru-Cevap hatası:', err);
    } finally {
      setIsAnswering(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 pb-16">
      {/* Top Banner */}
      <div className="bg-gradient-to-r from-slate-900 via-indigo-950 to-slate-900 text-white border-b border-indigo-900/50 py-10 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">
          <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-6">
            <div>
              <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-semibold bg-indigo-500/20 text-indigo-300 border border-indigo-500/30 mb-3">
                <Sparkles className="w-3.5 h-3.5 text-indigo-400" />
                RAG Pipeline v1.2 • Production Engine
              </div>
              <h1 className="text-3xl font-extrabold tracking-tight sm:text-4xl text-white">
                RAG & Retrieval Lab
              </h1>
              <p className="mt-2 text-slate-300 max-w-2xl text-sm leading-relaxed">
                Retrieval-Augmented Generation mimarisi: Çoklu stratejili metin parçalama (Chunking), 
                Yoğun (Dense Vektör) ve Seyrek (BM25) hibrit arama (RRF) ve kaynak doğrulamalı atıf (Citation) motoru.
              </p>
            </div>

            {/* Quick Stats Badges */}
            <div className="flex flex-wrap gap-3">
              <div className="bg-slate-800/80 backdrop-blur border border-slate-700/60 rounded-xl px-4 py-3 text-center min-w-[120px]">
                <div className="text-xs text-slate-400 font-medium">Koleksiyon</div>
                <div className="text-lg font-bold text-white mt-0.5">default</div>
                <div className="text-[10px] text-emerald-400 font-mono">● Aktif</div>
              </div>
              <div className="bg-slate-800/80 backdrop-blur border border-slate-700/60 rounded-xl px-4 py-3 text-center min-w-[120px]">
                <div className="text-xs text-slate-400 font-medium">Vektör Havuzu</div>
                <div className="text-lg font-bold text-indigo-300 mt-0.5">
                  {collections[0]?.vector_count || 20} Parça
                </div>
                <div className="text-[10px] text-slate-400 font-mono">d_model=64</div>
              </div>
              <div className="bg-slate-800/80 backdrop-blur border border-slate-700/60 rounded-xl px-4 py-3 text-center min-w-[120px]">
                <div className="text-xs text-slate-400 font-medium">Füzyon Algoritması</div>
                <div className="text-lg font-bold text-amber-300 mt-0.5">RRF (k=60)</div>
                <div className="text-[10px] text-indigo-300 font-mono">Dense + BM25</div>
              </div>
            </div>
          </div>

          {/* Navigation Tabs */}
          <div className="flex items-center gap-2 mt-8 border-b border-slate-800 pb-0">
            <button
              onClick={() => setActiveTab('qa')}
              className={`flex items-center gap-2 px-5 py-3 font-medium text-sm rounded-t-lg transition-all border-b-2 -mb-px ${
                activeTab === 'qa'
                  ? 'bg-slate-800/90 text-white border-indigo-400 shadow-sm'
                  : 'text-slate-400 hover:text-slate-200 border-transparent hover:border-slate-700'
              }`}
            >
              <MessageSquare className="w-4 h-4 text-indigo-400" />
              <span>Soru-Cevap & Atıf (Q&A Playground)</span>
              <span className="ml-1 px-1.5 py-0.5 text-[10px] rounded bg-indigo-500/30 text-indigo-200 font-mono">
                Grounding
              </span>
            </button>

            <button
              onClick={() => setActiveTab('retrieval')}
              className={`flex items-center gap-2 px-5 py-3 font-medium text-sm rounded-t-lg transition-all border-b-2 -mb-px ${
                activeTab === 'retrieval'
                  ? 'bg-slate-800/90 text-white border-indigo-400 shadow-sm'
                  : 'text-slate-400 hover:text-slate-200 border-transparent hover:border-slate-700'
              }`}
            >
              <Search className="w-4 h-4 text-amber-400" />
              <span>Hibrit Arama (Retrieval) Lab</span>
            </button>

            <button
              onClick={() => setActiveTab('chunking')}
              className={`flex items-center gap-2 px-5 py-3 font-medium text-sm rounded-t-lg transition-all border-b-2 -mb-px ${
                activeTab === 'chunking'
                  ? 'bg-slate-800/90 text-white border-indigo-400 shadow-sm'
                  : 'text-slate-400 hover:text-slate-200 border-transparent hover:border-slate-700'
              }`}
            >
              <Layers className="w-4 h-4 text-emerald-400" />
              <span>Parçalama (Chunking) Lab</span>
            </button>
          </div>
        </div>
      </div>

      {/* Content Area */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 mt-8">

        {/* ==================================================================== */}
        {/* TAB 1: RAG Q&A PLAYGROUND */}
        {/* ==================================================================== */}
        {activeTab === 'qa' && (
          <div className="space-y-8">
            <div className="bg-white rounded-2xl p-6 sm:p-8 shadow-sm border border-slate-200">
              <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-6 border-b border-slate-100">
                <div>
                  <h2 className="text-xl font-bold text-slate-900 flex items-center gap-2">
                    <MessageSquare className="w-5 h-5 text-indigo-600" />
                    Kaynak Doğrulamalı Soru-Cevap
                  </h2>
                  <p className="text-sm text-slate-500 mt-1">
                    Sorunuz yerel veri havuzunda aranır, en yüksek alaka düzeyine sahip bağlam parçaları derlenir ve üretilen yanıtta doğrulanabilir atıflar (<span className="font-mono text-indigo-600">[Kaynak 1]</span>) eklenir.
                  </p>
                </div>
                <div className="flex items-center gap-2">
                  <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-emerald-100 text-emerald-800">
                    <ShieldCheck className="w-3.5 h-3.5" />
                    Halüsinasyon Korumalı
                  </span>
                </div>
              </div>

              {/* Presets */}
              <div className="mt-6">
                <label className="text-xs font-semibold text-slate-500 tracking-wider uppercase">
                  Örnek Sorular
                </label>
                <div className="flex flex-wrap gap-2 mt-2">
                  {PRESET_QUESTIONS.map((q, idx) => (
                    <button
                      key={idx}
                      onClick={() => setQaQuestion(q)}
                      className="text-xs bg-slate-100 hover:bg-indigo-50 hover:text-indigo-700 text-slate-700 px-3 py-1.5 rounded-lg border border-slate-200 transition-colors"
                    >
                      {q}
                    </button>
                  ))}
                </div>
              </div>

              {/* Input Area */}
              <div className="mt-6">
                <label className="block text-sm font-medium text-slate-700 mb-2">
                  Sorunuzu Yazın
                </label>
                <div className="flex gap-3">
                  <input
                    type="text"
                    value={qaQuestion}
                    onChange={(e) => setQaQuestion(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && handleRunQA()}
                    placeholder="Örn: Transformer mimarisinde Scaled Dot-Product Attention formülü nedir?"
                    className="flex-1 px-4 py-3 bg-slate-50 border border-slate-200 rounded-xl focus:ring-2 focus:ring-indigo-500 focus:outline-none text-slate-900 font-medium placeholder-slate-400"
                  />
                  <button
                    onClick={handleRunQA}
                    disabled={isAnswering || !qaQuestion.trim()}
                    className="px-6 py-3 bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 text-white font-semibold rounded-xl transition-all shadow-sm flex items-center gap-2"
                  >
                    {isAnswering ? (
                      <>
                        <RefreshCw className="w-4 h-4 animate-spin" />
                        <span>Üretiliyor...</span>
                      </>
                    ) : (
                      <>
                        <Zap className="w-4 h-4" />
                        <span>Sorgula & Üret</span>
                      </>
                    )}
                  </button>
                </div>
              </div>

              {/* Generated Answer Display */}
              {qaAnswer && (
                <div className="mt-8 pt-8 border-t border-slate-100 space-y-6">
                  <div className="bg-gradient-to-br from-indigo-50/70 via-slate-50 to-purple-50/40 rounded-2xl p-6 border border-indigo-100">
                    <div className="flex items-center justify-between mb-4">
                      <div className="flex items-center gap-2">
                        <div className="p-1.5 bg-indigo-600 text-white rounded-lg">
                          <Sparkles className="w-4 h-4" />
                        </div>
                        <span className="font-bold text-slate-900">Grounded Model Yanıtı</span>
                        <span className="text-xs bg-indigo-100 text-indigo-800 px-2 py-0.5 rounded font-mono font-medium">
                          {qaStats.model_name || 'local-rag-grounded-generator'}
                        </span>
                      </div>
                      {qaStats.latency_ms && (
                        <span className="text-xs text-slate-500 font-mono">
                          ⚡ {qaStats.latency_ms} ms
                        </span>
                      )}
                    </div>

                    <div className="text-slate-800 text-base leading-relaxed whitespace-pre-line font-normal">
                      {qaAnswer}
                    </div>

                    {/* Citations list */}
                    {qaCitations.length > 0 && (
                      <div className="mt-6 pt-4 border-t border-indigo-200/50">
                        <div className="text-xs font-semibold text-indigo-900 uppercase tracking-wider mb-3">
                          Doğrulanan Kaynaklar & Atıflar ({qaCitations.length})
                        </div>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                          {qaCitations.map((cit) => (
                            <div
                              key={cit.citation_id}
                              className="bg-white/80 backdrop-blur rounded-xl p-3 border border-indigo-100/80 shadow-xs"
                            >
                              <div className="flex items-center justify-between gap-2 mb-1">
                                <span className="inline-flex items-center gap-1 font-bold text-xs text-indigo-600 bg-indigo-50 px-2 py-0.5 rounded">
                                  [Kaynak {cit.citation_id}]
                                </span>
                                <span className="text-[11px] font-semibold text-slate-700 truncate max-w-[180px]">
                                  {cit.source_title}
                                </span>
                                <span className="text-[10px] text-emerald-600 font-mono bg-emerald-50 px-1.5 py-0.5 rounded">
                                  Skor: {cit.score}
                                </span>
                              </div>
                              <p className="text-xs text-slate-600 italic line-clamp-2">
                                &quot;{cit.snippet}&quot;
                              </p>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Prompt Inspector Toggle */}
                  <div className="border border-slate-200 rounded-xl overflow-hidden">
                    <button
                      onClick={() => setShowPromptDetails(!showPromptDetails)}
                      className="w-full flex items-center justify-between px-4 py-3 bg-slate-50 hover:bg-slate-100 text-slate-700 text-xs font-semibold transition-colors"
                    >
                      <span className="flex items-center gap-2">
                        <Info className="w-4 h-4 text-indigo-600" />
                        Model Enjeksiyonunu İncele (Grounded Prompt & System Instruction)
                      </span>
                      {showPromptDetails ? (
                        <ChevronUp className="w-4 h-4" />
                      ) : (
                        <ChevronDown className="w-4 h-4" />
                      )}
                    </button>
                    {showPromptDetails && qaPromptUsed && (
                      <div className="p-4 bg-slate-900 text-slate-200 font-mono text-xs whitespace-pre-wrap leading-relaxed max-h-96 overflow-y-auto">
                        {qaPromptUsed}
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* ==================================================================== */}
        {/* TAB 2: HYBRID RETRIEVAL EXPLORER */}
        {/* ==================================================================== */}
        {activeTab === 'retrieval' && (
          <div className="space-y-6">
            <div className="bg-white rounded-2xl p-6 sm:p-8 shadow-sm border border-slate-200">
              <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-6 border-b border-slate-100">
                <div>
                  <h2 className="text-xl font-bold text-slate-900 flex items-center gap-2">
                    <Search className="w-5 h-5 text-amber-500" />
                    Hibrit Arama (Dense vs Sparse vs RRF)
                  </h2>
                  <p className="text-sm text-slate-500 mt-1">
                    Semantik Vektör (Dense) ve BM25 Leksikal (Sparse) arama sonuçlarının Reciprocal Rank Fusion ile nasıl birleştiğini inceleyin.
                  </p>
                </div>
                {searchLatency !== null && (
                  <div className="px-3 py-1 bg-amber-50 text-amber-800 rounded-full border border-amber-200 text-xs font-mono font-semibold">
                    ⚡ Arama Süresi: {searchLatency} ms
                  </div>
                )}
              </div>

              {/* Controls */}
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mt-6">
                <div className="lg:col-span-2 space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-slate-700 mb-1.5">
                      Arama Sorgusu
                    </label>
                    <div className="flex gap-2">
                      <input
                        type="text"
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        onKeyDown={(e) => e.key === 'Enter' && handleRunSearch()}
                        placeholder="Aramak istediğiniz terimleri girin..."
                        className="flex-1 px-4 py-2.5 bg-slate-50 border border-slate-200 rounded-xl focus:ring-2 focus:ring-indigo-500 text-sm"
                      />
                      <button
                        onClick={handleRunSearch}
                        disabled={isSearching}
                        className="px-5 py-2.5 bg-amber-500 hover:bg-amber-600 text-white font-semibold rounded-xl text-sm transition-all flex items-center gap-1.5"
                      >
                        {isSearching ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Search className="w-4 h-4" />}
                        <span>Ara</span>
                      </button>
                    </div>
                  </div>

                  {/* Mode Selector */}
                  <div>
                    <label className="block text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">
                      Arama Modu
                    </label>
                    <div className="grid grid-cols-3 gap-2">
                      <button
                        onClick={() => setSearchMode('hybrid')}
                        className={`p-3 rounded-xl border text-center transition-all ${
                          searchMode === 'hybrid'
                            ? 'bg-amber-50 border-amber-300 text-amber-900 font-semibold ring-2 ring-amber-400/30'
                            : 'bg-slate-50 border-slate-200 text-slate-600 hover:bg-slate-100'
                        }`}
                      >
                        <div className="text-xs font-bold">⚡ Hibrit (RRF)</div>
                        <div className="text-[11px] text-slate-500 mt-0.5">Vektör + BM25</div>
                      </button>

                      <button
                        onClick={() => setSearchMode('dense')}
                        className={`p-3 rounded-xl border text-center transition-all ${
                          searchMode === 'dense'
                            ? 'bg-indigo-50 border-indigo-300 text-indigo-900 font-semibold ring-2 ring-indigo-400/30'
                            : 'bg-slate-50 border-slate-200 text-slate-600 hover:bg-slate-100'
                        }`}
                      >
                        <div className="text-xs font-bold">🧠 Dense (Vektör)</div>
                        <div className="text-[11px] text-slate-500 mt-0.5">Cosine Similarity</div>
                      </button>

                      <button
                        onClick={() => setSearchMode('sparse')}
                        className={`p-3 rounded-xl border text-center transition-all ${
                          searchMode === 'sparse'
                            ? 'bg-emerald-50 border-emerald-300 text-emerald-900 font-semibold ring-2 ring-emerald-400/30'
                            : 'bg-slate-50 border-slate-200 text-slate-600 hover:bg-slate-100'
                        }`}
                      >
                        <div className="text-xs font-bold">🔍 Sparse (BM25)</div>
                        <div className="text-[11px] text-slate-500 mt-0.5">Ters İndeks & TF-IDF</div>
                      </button>
                    </div>
                  </div>
                </div>

                {/* Sliders */}
                <div className="bg-slate-50 rounded-xl p-4 border border-slate-200 space-y-4">
                  <div>
                    <div className="flex justify-between text-xs font-medium text-slate-700 mb-1">
                      <span>Top-K Sonuç</span>
                      <span className="font-mono text-indigo-600 font-bold">{topK}</span>
                    </div>
                    <input
                      type="range"
                      min={1}
                      max={10}
                      value={topK}
                      onChange={(e) => setTopK(Number(e.target.value))}
                      className="w-full h-1.5 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-indigo-600"
                    />
                  </div>

                  {searchMode === 'hybrid' && (
                    <div>
                      <div className="flex justify-between text-xs font-medium text-slate-700 mb-1">
                        <span>Dense / Sparse Ağırlığı (Alpha)</span>
                        <span className="font-mono text-amber-600 font-bold">{alpha}</span>
                      </div>
                      <input
                        type="range"
                        min={0.0}
                        max={1.0}
                        step={0.05}
                        value={alpha}
                        onChange={(e) => setAlpha(Number(e.target.value))}
                        className="w-full h-1.5 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-amber-600"
                      />
                      <div className="flex justify-between text-[10px] text-slate-400 mt-1">
                        <span>BM25 Ağırlıklı (0.0)</span>
                        <span>Vektör Ağırlıklı (1.0)</span>
                      </div>
                    </div>
                  )}

                  <div className="pt-2 border-t border-slate-200 text-xs text-slate-500">
                    <p>
                      <strong>İpucu:</strong> Hibrit arama, teknik kısaltmaları BM25 ile yakalarken kavramsal benzerlikleri dense vektörlerle eşleştirir.
                    </p>
                  </div>
                </div>
              </div>

              {/* Results List */}
              <div className="mt-8">
                <div className="text-sm font-bold text-slate-900 mb-4 flex items-center gap-2">
                  <span>Arama Sonuçları</span>
                  <span className="text-xs bg-slate-100 text-slate-600 px-2 py-0.5 rounded-full font-mono">
                    {searchResults.length} eşleşme
                  </span>
                </div>

                {searchResults.length === 0 ? (
                  <div className="p-8 text-center bg-slate-50 rounded-xl border border-slate-200 text-slate-400 text-sm">
                    Arama yapmak için yukarıdaki butona tıklayın veya arama sorgusunu değiştirin.
                  </div>
                ) : (
                  <div className="space-y-4">
                    {searchResults.map((res) => (
                      <div
                        key={res.chunk_id}
                        className="p-5 bg-white rounded-xl border border-slate-200 hover:border-indigo-300 transition-all shadow-xs"
                      >
                        <div className="flex items-center justify-between gap-3 mb-2">
                          <div className="flex items-center gap-2">
                            <span className="w-6 h-6 rounded-full bg-slate-900 text-white text-xs font-bold flex items-center justify-center">
                              #{res.rank}
                            </span>
                            <span className="font-semibold text-sm text-slate-800">
                              {res.metadata.title || res.chunk_id}
                            </span>
                          </div>

                          <div className="flex items-center gap-2">
                            {res.metadata.dense_rank && (
                              <span className="text-[11px] bg-indigo-50 text-indigo-700 px-2 py-0.5 rounded border border-indigo-100 font-mono">
                                Vektör: #{res.metadata.dense_rank}
                              </span>
                            )}
                            {res.metadata.sparse_rank && (
                              <span className="text-[11px] bg-emerald-50 text-emerald-700 px-2 py-0.5 rounded border border-emerald-100 font-mono">
                                BM25: #{res.metadata.sparse_rank}
                              </span>
                            )}
                            <span className="text-xs font-bold text-slate-900 font-mono bg-slate-100 px-2 py-0.5 rounded">
                              Skor: {res.score}
                            </span>
                          </div>
                        </div>

                        <p className="text-sm text-slate-700 leading-relaxed bg-slate-50/70 p-3 rounded-lg border border-slate-100">
                          {res.text}
                        </p>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* ==================================================================== */}
        {/* TAB 3: CHUNKING VISUALIZER */}
        {/* ==================================================================== */}
        {activeTab === 'chunking' && (
          <div className="space-y-6">
            <div className="bg-white rounded-2xl p-6 sm:p-8 shadow-sm border border-slate-200">
              <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-6 border-b border-slate-100">
                <div>
                  <h2 className="text-xl font-bold text-slate-900 flex items-center gap-2">
                    <Layers className="w-5 h-5 text-emerald-600" />
                    Etkileşimli Metin Parçalama (Chunking)
                  </h2>
                  <p className="text-sm text-slate-500 mt-1">
                    Metinlerin semantik bütünlüğünü koruyarak nasıl parçalandığını, örtüşmelerin (overlap) nasıl oluştuğunu renk kodlarıyla canlı izleyin.
                  </p>
                </div>

                <button
                  onClick={handleRunChunking}
                  disabled={isChunking}
                  className="px-5 py-2.5 bg-emerald-600 hover:bg-emerald-700 text-white font-semibold rounded-xl text-sm transition-all flex items-center gap-2"
                >
                  {isChunking ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Layers className="w-4 h-4" />}
                  <span>Yeniden Parçala</span>
                </button>
              </div>

              {/* Sample Preset Choosers */}
              <div className="mt-6">
                <label className="text-xs font-semibold text-slate-500 tracking-wider uppercase">
                  Örnek Metinler
                </label>
                <div className="flex flex-wrap gap-2 mt-2">
                  {SAMPLE_PRESETS.map((preset, idx) => (
                    <button
                      key={idx}
                      onClick={() => {
                        setRawText(preset.text);
                      }}
                      className="text-xs bg-slate-100 hover:bg-emerald-50 hover:text-emerald-800 text-slate-700 px-3 py-1.5 rounded-lg border border-slate-200 transition-colors"
                    >
                      {preset.name}
                    </button>
                  ))}
                </div>
              </div>

              {/* Text Input */}
              <div className="mt-4">
                <textarea
                  rows={6}
                  value={rawText}
                  onChange={(e) => setRawText(e.target.value)}
                  placeholder="Parçalamak istediğiniz metni buraya yapıştırın..."
                  className="w-full p-4 bg-slate-50 border border-slate-200 rounded-xl focus:ring-2 focus:ring-emerald-500 text-sm font-mono text-slate-800 leading-relaxed"
                />
              </div>

              {/* Controls */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-6 p-4 bg-slate-50 rounded-xl border border-slate-200">
                {/* Strategy */}
                <div>
                  <label className="block text-xs font-semibold text-slate-600 uppercase mb-2">
                    Strateji
                  </label>
                  <div className="grid grid-cols-3 gap-1">
                    {(['recursive', 'sentence', 'fixed'] as const).map((strat) => (
                      <button
                        key={strat}
                        onClick={() => setChunkStrategy(strat)}
                        className={`py-2 px-2 text-xs font-semibold rounded-lg capitalize border transition-all ${
                          chunkStrategy === strat
                            ? 'bg-emerald-600 text-white border-emerald-600'
                            : 'bg-white text-slate-700 border-slate-200 hover:bg-slate-100'
                        }`}
                      >
                        {strat}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Chunk Size */}
                <div>
                  <div className="flex justify-between text-xs font-semibold text-slate-700 mb-1">
                    <span>Parça Boyutu (Karakter)</span>
                    <span className="font-mono text-emerald-700">{chunkSize}</span>
                  </div>
                  <input
                    type="range"
                    min={100}
                    max={1200}
                    step={25}
                    value={chunkSize}
                    onChange={(e) => setChunkSize(Number(e.target.value))}
                    className="w-full h-1.5 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-emerald-600"
                  />
                </div>

                {/* Overlap */}
                <div>
                  <div className="flex justify-between text-xs font-semibold text-slate-700 mb-1">
                    <span>Örtüşme (Overlap)</span>
                    <span className="font-mono text-emerald-700">{chunkOverlap}</span>
                  </div>
                  <input
                    type="range"
                    min={0}
                    max={Math.min(300, chunkSize - 20)}
                    step={10}
                    value={chunkOverlap}
                    onChange={(e) => setChunkOverlap(Number(e.target.value))}
                    className="w-full h-1.5 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-emerald-600"
                  />
                </div>
              </div>

              {/* Chunks Metrics Bar */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-6">
                <div className="p-3 bg-white border border-slate-200 rounded-xl text-center">
                  <div className="text-xs text-slate-500">Parça Sayısı</div>
                  <div className="text-xl font-bold text-slate-900">{chunks.length}</div>
                </div>
                <div className="p-3 bg-white border border-slate-200 rounded-xl text-center">
                  <div className="text-xs text-slate-500">Toplam Karakter</div>
                  <div className="text-xl font-bold text-slate-900">{rawText.length}</div>
                </div>
                <div className="p-3 bg-white border border-slate-200 rounded-xl text-center">
                  <div className="text-xs text-slate-500">Tahmini Token</div>
                  <div className="text-xl font-bold text-emerald-600">
                    {chunks.reduce((acc, c) => acc + c.token_count, 0)}
                  </div>
                </div>
                <div className="p-3 bg-white border border-slate-200 rounded-xl text-center">
                  <div className="text-xs text-slate-500">Ortalama Boyut</div>
                  <div className="text-xl font-bold text-slate-900">
                    {chunks.length > 0
                      ? Math.round(chunks.reduce((acc, c) => acc + c.text.length, 0) / chunks.length)
                      : 0} ch
                  </div>
                </div>
              </div>

              {/* Segmented Visual View */}
              <div className="mt-8">
                <div className="text-sm font-bold text-slate-900 mb-3">
                  Renklendirilmiş Parça Dağılımı
                </div>
                <div className="flex flex-wrap gap-2 p-4 bg-slate-50 rounded-xl border border-slate-200">
                  {chunks.map((chunk, idx) => {
                    const colorClass = CHUNK_COLORS[idx % CHUNK_COLORS.length];
                    const isSelected = selectedChunkIdx === idx;
                    return (
                      <div
                        key={chunk.chunk_id}
                        onClick={() => setSelectedChunkIdx(isSelected ? null : idx)}
                        className={`p-3 rounded-lg border text-xs leading-relaxed cursor-pointer transition-all ${colorClass} ${
                          isSelected ? 'ring-2 ring-slate-900 shadow-md scale-[1.01]' : 'hover:opacity-90'
                        }`}
                      >
                        <div className="flex items-center justify-between font-bold text-[10px] mb-1.5 opacity-80 uppercase tracking-wider">
                          <span>Parça #{chunk.chunk_index + 1}</span>
                          <span>{chunk.token_count} token</span>
                        </div>
                        <div>{chunk.text}</div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
