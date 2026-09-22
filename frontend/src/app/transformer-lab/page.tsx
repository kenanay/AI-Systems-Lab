'use client';

import React, { useState, useEffect } from 'react';
import Link from 'next/link';
import {
  Layers,
  Cpu,
  Zap,
  Activity,
  ArrowRight,
  RefreshCw,
  Sliders,
  CheckCircle2,
  Info,
  Maximize2,
  GitBranch,
  TrendingUp,
  Table,
  HelpCircle,
  ChevronRight,
  Hash,
  Play,
  Box,
  Compass,
  Database,
  BarChart2
} from 'lucide-react';
import {
  transformerLabApi,
  SinusoidalResponse,
  RoPEResponse,
  ALiBiResponse,
  PEComparisonItem,
  KVCacheAnalysisResponse,
  BlockSimulateResponse,
  ModelParamsResponse,
  ArchitecturePreset,
} from '@/lib/api';

export default function TransformerLabPage() {
  const [activeTab, setActiveTab] = useState<'pe' | 'attention_variants' | 'block' | 'params'>('pe');

  // --------------------------------------------------------------------------
  // Tab 1: Positional Encoding State
  // --------------------------------------------------------------------------
  const [peSubTab, setPeSubTab] = useState<'rope' | 'sinusoidal' | 'alibi' | 'compare'>('rope');

  // RoPE state
  const [ropeDim, setRopeDim] = useState(16);
  const [ropeSeqLen, setRopeSeqLen] = useState(8);
  const [ropeBase, setRopeBase] = useState(10000.0);
  const [ropeData, setRopeData] = useState<RoPEResponse | null>(null);
  const [ropeLoading, setRopeLoading] = useState(false);

  // Sinusoidal state
  const [sinSeqLen, setSinSeqLen] = useState(16);
  const [sinDModel, setSinDModel] = useState(32);
  const [sinusoidalData, setSinusoidalData] = useState<SinusoidalResponse | null>(null);
  const [sinusoidalLoading, setSinusoidalLoading] = useState(false);

  // ALiBi state
  const [alibiHeads, setAlibiHeads] = useState(8);
  const [alibiSeqLen, setAlibiSeqLen] = useState(8);
  const [alibiData, setAlibiData] = useState<ALiBiResponse | null>(null);
  const [selectedAlibiHead, setSelectedAlibiHead] = useState(0);

  // PE Comparison table state
  const [peComparison, setPeComparison] = useState<PEComparisonItem[]>([]);

  // --------------------------------------------------------------------------
  // Tab 2: MHA vs GQA vs MQA State
  // --------------------------------------------------------------------------
  const [kvBatchSize, setKvBatchSize] = useState(2);
  const [kvSeqLen, setKvSeqLen] = useState(4096);
  const [kvQHeads, setKvQHeads] = useState(32);
  const [kvKVHeads, setKvKVHeads] = useState(8);
  const [kvHeadDim, setKvHeadDim] = useState(128);
  const [kvLayers, setKvLayers] = useState(32);
  const [kvDtype, setKvDtype] = useState('float16');
  const [kvData, setKvData] = useState<KVCacheAnalysisResponse | null>(null);
  const [kvLoading, setKvLoading] = useState(false);

  // --------------------------------------------------------------------------
  // Tab 3: Transformer Block State
  // --------------------------------------------------------------------------
  const [blockNormType, setBlockNormType] = useState('rmsnorm');
  const [blockFfnType, setBlockFfnType] = useState('swiglu');
  const [blockNormPlacement, setBlockNormPlacement] = useState('pre_ln');
  const [blockData, setBlockData] = useState<BlockSimulateResponse | null>(null);
  const [blockLoading, setBlockLoading] = useState(false);

  // --------------------------------------------------------------------------
  // Tab 4: Architecture Parameters State
  // --------------------------------------------------------------------------
  const [presets, setPresets] = useState<ArchitecturePreset[]>([]);
  const [selectedPreset, setSelectedPreset] = useState('LLaMA-3 8B');
  const [modelVocab, setModelVocab] = useState(128256);
  const [modelDModel, setModelDModel] = useState(4096);
  const [modelLayers, setModelLayers] = useState(32);
  const [modelHeads, setModelHeads] = useState(32);
  const [modelKVHeads, setModelKVHeads] = useState(8);
  const [modelDFF, setModelDFF] = useState(14336);
  const [modelFFNType, setModelFFNType] = useState('swiglu');
  const [modelTieEmbeddings, setModelTieEmbeddings] = useState(false);
  const [paramsData, setParamsData] = useState<ModelParamsResponse | null>(null);

  // --------------------------------------------------------------------------
  // Actions: Positional Encoding
  // --------------------------------------------------------------------------
  const fetchRoPE = async (dim = ropeDim, len = ropeSeqLen, base = ropeBase) => {
    setRopeLoading(true);
    try {
      const res = await transformerLabApi.getRoPE(dim, len, base);
      setRopeData(res);
    } catch (err) {
      console.error(err);
    } finally {
      setRopeLoading(false);
    }
  };

  const fetchSinusoidal = async (seq = sinSeqLen, d = sinDModel) => {
    setSinusoidalLoading(true);
    try {
      const res = await transformerLabApi.getSinusoidalPE(seq, d);
      setSinusoidalData(res);
    } catch (err) {
      console.error(err);
    } finally {
      setSinusoidalLoading(false);
    }
  };

  const fetchALiBi = async (heads = alibiHeads, seq = alibiSeqLen) => {
    try {
      const res = await transformerLabApi.getALiBi(heads, seq);
      setAlibiData(res);
      setSelectedAlibiHead(0);
    } catch (err) {
      console.error(err);
    }
  };

  const fetchPEComparison = async () => {
    try {
      const res = await transformerLabApi.comparePE();
      setPeComparison(res);
    } catch (err) {
      console.error(err);
    }
  };

  // --------------------------------------------------------------------------
  // Actions: Attention Variants & KV-Cache
  // --------------------------------------------------------------------------
  const runKvCacheAnalysis = async (
    b = kvBatchSize,
    seq = kvSeqLen,
    qh = kvQHeads,
    kvh = kvKVHeads,
    hdim = kvHeadDim,
    layers = kvLayers,
    dt = kvDtype
  ) => {
    setKvLoading(true);
    try {
      const res = await transformerLabApi.analyzeAttentionVariants({
        batch_size: b,
        seq_len: seq,
        num_query_heads: qh,
        num_kv_heads: kvh,
        head_dim: hdim,
        num_layers: layers,
        dtype: dt,
      });
      setKvData(res);
    } catch (err) {
      console.error(err);
    } finally {
      setKvLoading(false);
    }
  };

  // --------------------------------------------------------------------------
  // Actions: Transformer Block
  // --------------------------------------------------------------------------
  const runBlockSimulation = async (
    norm = blockNormType,
    ffn = blockFfnType,
    placement = blockNormPlacement
  ) => {
    setBlockLoading(true);
    try {
      const res = await transformerLabApi.simulateBlock({
        norm_type: norm,
        ffn_type: ffn,
        norm_placement: placement,
        batch_size: 1,
        seq_len: 4,
        d_model: 8,
        d_ff: 16,
      });
      setBlockData(res);
    } catch (err) {
      console.error(err);
    } finally {
      setBlockLoading(false);
    }
  };

  // --------------------------------------------------------------------------
  // Actions: Parameters
  // --------------------------------------------------------------------------
  const fetchPresetsAndCalculate = async () => {
    try {
      const pList = await transformerLabApi.getPresets();
      setPresets(pList);
      if (pList.length > 0) {
        applyPreset(pList[0]);
      }
    } catch (err) {
      console.error(err);
    }
  };

  const applyPreset = (p: ArchitecturePreset) => {
    setSelectedPreset(p.name);
    setModelVocab(p.vocab_size);
    setModelDModel(p.d_model);
    setModelLayers(p.n_layers);
    setModelHeads(p.n_heads);
    setModelKVHeads(p.n_kv_heads);
    setModelDFF(p.d_ff);
    setModelFFNType(p.ffn_type);
    setModelTieEmbeddings(p.tie_word_embeddings);
    runParamCalculation(
      p.vocab_size,
      p.d_model,
      p.n_layers,
      p.n_heads,
      p.n_kv_heads,
      p.d_ff,
      p.tie_word_embeddings,
      p.ffn_type
    );
  };

  const runParamCalculation = async (
    vocab = modelVocab,
    d = modelDModel,
    layers = modelLayers,
    heads = modelHeads,
    kvh = modelKVHeads,
    dff = modelDFF,
    tied = modelTieEmbeddings,
    ffn = modelFFNType
  ) => {
    try {
      const res = await transformerLabApi.calculateParams({
        vocab_size: vocab,
        d_model: d,
        n_layers: layers,
        n_heads: heads,
        n_kv_heads: kvh,
        d_ff: dff,
        tie_word_embeddings: tied,
        ffn_type: ffn,
      });
      setParamsData(res);
    } catch (err) {
      console.error(err);
    }
  };

  // Initial loads
  useEffect(() => {
    fetchRoPE(16, 8, 10000.0);
    fetchSinusoidal(16, 32);
    fetchALiBi(8, 8);
    fetchPEComparison();
    runKvCacheAnalysis(2, 4096, 32, 8, 128, 32, 'float16');
    runBlockSimulation('rmsnorm', 'swiglu', 'pre_ln');
    fetchPresetsAndCalculate();
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-50 via-white to-gray-50 text-gray-900 pb-16">
      {/* Top Header Banner */}
      <div className="border-b border-gray-200/80 bg-white/70 backdrop-blur-md sticky top-16 z-40">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
            <div>
              <div className="flex items-center space-x-2.5">
                <span className="text-2xl p-2 rounded-xl bg-violet-50 border border-violet-100 shadow-sm">
                  🏛️
                </span>
                <div>
                  <h1 className="text-xl sm:text-2xl font-bold tracking-tight text-gray-900">
                    Transformer Architecture Lab
                  </h1>
                  <p className="text-xs sm:text-sm text-gray-500">
                    Positional Encodings (RoPE, Sinusoidal, ALiBi), MHA vs GQA, Pre-LN vs Post-LN & SwiGLU Blok Simülatörü
                  </p>
                </div>
              </div>
            </div>

            {/* Tab Buttons */}
            <div className="flex items-center bg-gray-100/80 p-1 rounded-xl border border-gray-200 text-xs sm:text-sm font-medium overflow-x-auto">
              <button
                onClick={() => setActiveTab('pe')}
                className={`flex items-center space-x-1.5 px-3 py-2 rounded-lg transition-all ${
                  activeTab === 'pe'
                    ? 'bg-white text-violet-700 font-semibold shadow-sm border border-gray-200/50'
                    : 'text-gray-600 hover:text-gray-900'
                }`}
              >
                <Compass className="w-3.5 h-3.5" />
                <span>1. Positional Encoding</span>
              </button>

              <button
                onClick={() => setActiveTab('attention_variants')}
                className={`flex items-center space-x-1.5 px-3 py-2 rounded-lg transition-all ${
                  activeTab === 'attention_variants'
                    ? 'bg-white text-violet-700 font-semibold shadow-sm border border-gray-200/50'
                    : 'text-gray-600 hover:text-gray-900'
                }`}
              >
                <Database className="w-3.5 h-3.5" />
                <span>2. MHA vs GQA vs MQA</span>
              </button>

              <button
                onClick={() => setActiveTab('block')}
                className={`flex items-center space-x-1.5 px-3 py-2 rounded-lg transition-all ${
                  activeTab === 'block'
                    ? 'bg-white text-violet-700 font-semibold shadow-sm border border-gray-200/50'
                    : 'text-gray-600 hover:text-gray-900'
                }`}
              >
                <Box className="w-3.5 h-3.5" />
                <span>3. Transformer Bloğu</span>
              </button>

              <button
                onClick={() => setActiveTab('params')}
                className={`flex items-center space-x-1.5 px-3 py-2 rounded-lg transition-all ${
                  activeTab === 'params'
                    ? 'bg-white text-violet-700 font-semibold shadow-sm border border-gray-200/50'
                    : 'text-gray-600 hover:text-gray-900'
                }`}
              >
                <BarChart2 className="w-3.5 h-3.5" />
                <span>4. Model Parametreleri</span>
              </button>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 mt-6">
        {/* ================================================================== */}
        {/* TAB 1: POSITIONAL ENCODINGS (RoPE, Sinusoidal, ALiBi, Compare)      */}
        {/* ================================================================== */}
        {activeTab === 'pe' && (
          <div className="space-y-6">
            {/* Sub-nav for Positional Encoding variants */}
            <div className="flex items-center space-x-2 border-b border-gray-200 pb-3">
              <button
                onClick={() => setPeSubTab('rope')}
                className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                  peSubTab === 'rope'
                    ? 'bg-violet-600 text-white shadow-sm'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
              >
                🌀 RoPE (Rotary Position Embedding - LLaMA/Mistral)
              </button>

              <button
                onClick={() => setPeSubTab('sinusoidal')}
                className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                  peSubTab === 'sinusoidal'
                    ? 'bg-violet-600 text-white shadow-sm'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
              >
                🌊 Sinusoidal Encoding (Vaswani 2017)
              </button>

              <button
                onClick={() => setPeSubTab('alibi')}
                className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                  peSubTab === 'alibi'
                    ? 'bg-violet-600 text-white shadow-sm'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
              >
                📐 ALiBi (Linear Biases)
              </button>

              <button
                onClick={() => setPeSubTab('compare')}
                className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                  peSubTab === 'compare'
                    ? 'bg-violet-600 text-white shadow-sm'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
              >
                📊 Karşılaştırma Atlası
              </button>
            </div>

            {/* 1A: RoPE Interactive View */}
            {peSubTab === 'rope' && (
              <div className="space-y-6">
                <div className="bg-white rounded-2xl p-6 border border-gray-200/80 shadow-sm space-y-4">
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                    <div>
                      <h2 className="text-base font-semibold text-gray-900 flex items-center space-x-2">
                        <Compass className="w-4 h-4 text-violet-600" />
                        <span>RoPE: 2D Karmaşık Düzlemde Vektör Rotasyonu</span>
                      </h2>
                      <p className="text-xs text-gray-500 mt-0.5">
                        Query ve Key vektörleri pozisyon indeksi m ile çarpılan rotasyon matrisi R(m·θ) ile döndürülür.
                        (R_m q)ᵀ (R_n k) = qᵀ R_{'{n-m}'} k özelliği sayesinde iç çarpım yalnızca <strong>göreli mesafeye</strong> bağlı kalır.
                      </p>
                    </div>

                    <button
                      onClick={() => fetchRoPE()}
                      disabled={ropeLoading}
                      className="px-4 py-2 bg-violet-600 hover:bg-violet-700 text-white rounded-xl text-xs font-semibold shadow-sm transition-all flex items-center space-x-1.5 self-start sm:self-auto"
                    >
                      {ropeLoading ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <Play className="w-3.5 h-3.5" />}
                      <span>Hesapla</span>
                    </button>
                  </div>

                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                    <div>
                      <label className="text-xs font-semibold text-gray-500 block mb-1">
                        Başlık Boyutu (dim = {ropeDim})
                      </label>
                      <input
                        type="range"
                        min={8}
                        max={64}
                        step={8}
                        value={ropeDim}
                        onChange={(e) => {
                          const v = parseInt(e.target.value);
                          setRopeDim(v);
                          fetchRoPE(v, ropeSeqLen, ropeBase);
                        }}
                        className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-violet-600"
                      />
                    </div>

                    <div>
                      <label className="text-xs font-semibold text-gray-500 block mb-1">
                        Pozisyon Sayısı (m = {ropeSeqLen})
                      </label>
                      <input
                        type="range"
                        min={4}
                        max={16}
                        step={2}
                        value={ropeSeqLen}
                        onChange={(e) => {
                          const v = parseInt(e.target.value);
                          setRopeSeqLen(v);
                          fetchRoPE(ropeDim, v, ropeBase);
                        }}
                        className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-violet-600"
                      />
                    </div>

                    <div>
                      <label className="text-xs font-semibold text-gray-500 block mb-1">
                        Frekans Tabanı (θ Base = {ropeBase.toLocaleString()})
                      </label>
                      <select
                        value={ropeBase}
                        onChange={(e) => {
                          const v = parseFloat(e.target.value);
                          setRopeBase(v);
                          fetchRoPE(ropeDim, ropeSeqLen, v);
                        }}
                        className="w-full px-3 py-1.5 border rounded-xl text-xs bg-white font-medium"
                      >
                        <option value={10000}>10,000 (Orijinal RoPE / LLaMA-1/2)</option>
                        <option value={500000}>500,000 (LLaMA-3 128K Long Context)</option>
                        <option value={1000000}>1,000,000 (Extreme Long Context)</option>
                      </select>
                    </div>
                  </div>
                </div>

                {ropeData && (
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    {/* 2D Vector Rotation Circular Diagram */}
                    <div className="bg-white rounded-2xl border border-gray-200/80 p-6 shadow-sm space-y-3">
                      <div className="flex items-center justify-between">
                        <span className="text-xs font-semibold text-gray-700 uppercase tracking-wider">
                          2D Düzlemde Vektör Rotasyonu (m = 0..{ropeSeqLen - 1})
                        </span>
                        <div className="flex space-x-3 text-xs font-mono">
                          <span className="text-blue-600 flex items-center space-x-1">
                            <span className="w-2.5 h-2.5 rounded-full bg-blue-600 inline-block"></span>
                            <span>Query Vektörü</span>
                          </span>
                          <span className="text-amber-500 flex items-center space-x-1">
                            <span className="w-2.5 h-2.5 rounded-full bg-amber-500 inline-block"></span>
                            <span>Key Vektörü</span>
                          </span>
                        </div>
                      </div>

                      <div className="relative h-64 w-full bg-gray-950 rounded-xl flex items-center justify-center p-2">
                        <svg className="w-full h-full" viewBox="-120 -120 240 240">
                          {/* Unit circle and axes */}
                          <circle cx="0" cy="0" r="90" fill="none" stroke="#374151" strokeWidth="1" strokeDasharray="3 3" />
                          <circle cx="0" cy="0" r="45" fill="none" stroke="#1f2937" strokeWidth="1" />
                          <line x1="-100" y1="0" x2="100" y2="0" stroke="#374151" strokeWidth="1" />
                          <line x1="0" y1="-100" x2="0" y2="100" stroke="#374151" strokeWidth="1" />

                          {/* Rotated vectors for each position */}
                          {ropeData.positions_data.map((pos) => {
                            const qX = pos.q_rotated[0] * 85;
                            const qY = -pos.q_rotated[1] * 85; // Invert SVG y
                            const kX = pos.k_rotated[0] * 85;
                            const kY = -pos.k_rotated[1] * 85;

                            return (
                              <g key={pos.position}>
                                {/* Query Vector Line */}
                                <line x1="0" y1="0" x2={qX} y2={qY} stroke="#3b82f6" strokeWidth="2" strokeOpacity={0.8} />
                                <circle cx={qX} cy={qY} r="3.5" fill="#3b82f6" />
                                <text x={qX + 4} y={qY - 2} fill="#93c5fd" fontSize="9" fontFamily="monospace">
                                  q{pos.position} ({pos.angle_deg}°)
                                </text>

                                {/* Key Vector Line */}
                                <line x1="0" y1="0" x2={kX} y2={kY} stroke="#f59e0b" strokeWidth="1.5" strokeOpacity={0.7} strokeDasharray="2 2" />
                                <circle cx={kX} cy={kY} r="3" fill="#f59e0b" />
                              </g>
                            );
                          })}
                        </svg>
                      </div>

                      <p className="text-xs text-gray-500">
                        Her pozisyon ilerledikçe $q$ ve $k$ vektörleri saat yönünün tersine açılan açı ile döner.
                        Açısal fark sabit kaldığında iki vektör arasındaki benzerlik skoru da korunur.
                      </p>
                    </div>

                    {/* Relative Distance Dot Product Heatmap */}
                    <div className="bg-white rounded-2xl border border-gray-200/80 p-6 shadow-sm space-y-3">
                      <div className="flex items-center justify-between">
                        <span className="text-xs font-semibold text-gray-700 uppercase tracking-wider">
                          Göreli Mesafe Benzerlik Matrisi (q_m · k_n)
                        </span>
                        <span className="text-[11px] text-violet-600 font-mono font-medium">
                          Bantlı Toeplitz Yapısı
                        </span>
                      </div>

                      <div className="overflow-x-auto">
                        <table className="w-full text-center text-xs font-mono">
                          <thead>
                            <tr>
                              <th className="p-1 text-gray-400 font-sans text-[10px]">m \ n</th>
                              {ropeData.relative_dot_matrix.map((_, colIdx) => (
                                <th key={colIdx} className="p-1 text-gray-600 font-bold">
                                  pos {colIdx}
                                </th>
                              ))}
                            </tr>
                          </thead>
                          <tbody>
                            {ropeData.relative_dot_matrix.map((row, rIdx) => (
                              <tr key={rIdx}>
                                <td className="p-1 font-bold text-gray-600 text-left">pos {rIdx}</td>
                                {row.map((val, cIdx) => {
                                  // Intensity based on dot product [-1, 1]
                                  const intensity = Math.max(0, Math.min(1, (val + 1) / 2));
                                  return (
                                    <td
                                      key={cIdx}
                                      className="p-1.5 border border-gray-100 rounded text-[11px] font-semibold"
                                      style={{
                                        backgroundColor: `rgba(124, 58, 237, ${intensity * 0.35 + 0.05})`,
                                        color: intensity > 0.6 ? '#5b21b6' : '#1e1b4b',
                                      }}
                                    >
                                      {val.toFixed(2)}
                                    </td>
                                  );
                                })}
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>

                      <p className="text-xs text-gray-500">
                        Aynı köşegen (diagonal) üzerindeki değerler birbirine eşittir (Toeplitz özelliği).
                        Bu durum, mutlak pozisyonlar $m$ ve $n$ ne kadar büyük olursa olsun, modelin aradaki farkı $|m - n|$ tutarlı olarak algıladığını gösterir.
                      </p>
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* 1B: Sinusoidal Encoding View */}
            {peSubTab === 'sinusoidal' && (
              <div className="space-y-6">
                <div className="bg-white rounded-2xl p-6 border border-gray-200/80 shadow-sm space-y-4">
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                    <div>
                      <h2 className="text-base font-semibold text-gray-900 flex items-center space-x-2">
                        <TrendingUp className="w-4 h-4 text-violet-600" />
                        <span>Sinusoidal Pozisyonel Kodlama (Vaswani 2017)</span>
                      </h2>
                      <p className="text-xs text-gray-500 mt-0.5">
                        Farklı dalga boylarındaki sinüs ve kosinüs harmonik fonksiyonları kullanılarak oluşturulan sabit pozisyon matrisi.
                      </p>
                    </div>

                    <button
                      onClick={() => fetchSinusoidal()}
                      disabled={sinusoidalLoading}
                      className="px-4 py-2 bg-violet-600 hover:bg-violet-700 text-white rounded-xl text-xs font-semibold shadow-sm transition-all flex items-center space-x-1.5 self-start sm:self-auto"
                    >
                      {sinusoidalLoading ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <Play className="w-3.5 h-3.5" />}
                      <span>Yenile</span>
                    </button>
                  </div>

                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                    <div>
                      <label className="text-xs font-semibold text-gray-500 block mb-1">
                        Dizi Uzunluğu (seq_len = {sinSeqLen})
                      </label>
                      <input
                        type="range"
                        min={8}
                        max={32}
                        step={4}
                        value={sinSeqLen}
                        onChange={(e) => {
                          const v = parseInt(e.target.value);
                          setSinSeqLen(v);
                          fetchSinusoidal(v, sinDModel);
                        }}
                        className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-violet-600"
                      />
                    </div>

                    <div>
                      <label className="text-xs font-semibold text-gray-500 block mb-1">
                        Model Boyutu (d_model = {sinDModel})
                      </label>
                      <input
                        type="range"
                        min={16}
                        max={64}
                        step={8}
                        value={sinDModel}
                        onChange={(e) => {
                          const v = parseInt(e.target.value);
                          setSinDModel(v);
                          fetchSinusoidal(sinSeqLen, v);
                        }}
                        className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-violet-600"
                      />
                    </div>
                  </div>
                </div>

                {sinusoidalData && (
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    {/* Heatmap */}
                    <div className="bg-white rounded-2xl border border-gray-200/80 p-6 shadow-sm space-y-3">
                      <div className="flex items-center justify-between">
                        <span className="text-xs font-semibold text-gray-700 uppercase tracking-wider">
                          PE Isı Haritası ({sinusoidalData.seq_len} Pozisyon × {sinusoidalData.d_model} Boyut)
                        </span>
                      </div>

                      <div className="relative h-64 overflow-auto border border-gray-200 rounded-xl p-2 bg-gray-900">
                        <div className="flex flex-col space-y-1">
                          {sinusoidalData.pe_matrix.map((row, posIdx) => (
                            <div key={posIdx} className="flex items-center space-x-1">
                              <span className="text-[9px] font-mono text-gray-500 w-8 text-right pr-1">
                                #{posIdx}
                              </span>
                              {row.map((val, dimIdx) => {
                                // -1 to 1: blue to red
                                const normVal = (val + 1) / 2;
                                return (
                                  <div
                                    key={dimIdx}
                                    className="h-3 flex-1 rounded-sm"
                                    title={`Pos: ${posIdx}, Dim: ${dimIdx}, Val: ${val}`}
                                    style={{
                                      backgroundColor:
                                        val > 0
                                          ? `rgba(129, 140, 248, ${val})`
                                          : `rgba(244, 63, 94, ${Math.abs(val)})`,
                                    }}
                                  />
                                );
                              })}
                            </div>
                          ))}
                        </div>
                      </div>
                      <span className="text-[11px] text-gray-500 block">
                        Sol sütunlar (düşük indisler) yüksek frekansta dalgalanırken, sağ sütunlar (yüksek indisler) sabit kalır.
                      </span>
                    </div>

                    {/* Wave Channels Graph */}
                    <div className="bg-white rounded-2xl border border-gray-200/80 p-6 shadow-sm space-y-3">
                      <span className="text-xs font-semibold text-gray-700 uppercase tracking-wider block">
                        Frekans Dalgaları (Seçili Boyutlar)
                      </span>

                      <div className="h-64 bg-gray-950 p-4 rounded-xl text-white">
                        <svg className="w-full h-full" viewBox="0 0 400 160" preserveAspectRatio="none">
                          <line x1="0" y1="80" x2="400" y2="80" stroke="#374151" strokeWidth="1" strokeDasharray="3 3" />
                          {sinusoidalData.waves.map((w, wIdx) => {
                            const colors = ['#818cf8', '#34d399', '#fbbf24', '#f43f5e'];
                            const c = colors[wIdx % colors.length];
                            const pathStr = w.values.reduce((acc, val, idx) => {
                              const xCoord = (idx / (w.values.length - 1)) * 400;
                              const yCoord = 80 - val * 65;
                              return `${acc} ${idx === 0 ? 'M' : 'L'} ${xCoord} ${yCoord}`;
                            }, '');

                            return (
                              <g key={w.dimension_index}>
                                <path d={pathStr} fill="none" stroke={c} strokeWidth="2" />
                              </g>
                            );
                          })}
                        </svg>
                      </div>

                      <div className="flex flex-wrap gap-2 text-[11px] font-mono">
                        {sinusoidalData.waves.map((w, wIdx) => {
                          const colors = ['text-indigo-600', 'text-emerald-600', 'text-amber-600', 'text-rose-600'];
                          return (
                            <span key={w.dimension_index} className={`px-2 py-1 bg-gray-50 border rounded font-semibold ${colors[wIdx % colors.length]}`}>
                              Dim #{w.dimension_index} ({w.type.toUpperCase()})
                            </span>
                          );
                        })}
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* 1C: ALiBi View */}
            {peSubTab === 'alibi' && alibiData && (
              <div className="space-y-6">
                <div className="bg-white rounded-2xl p-6 border border-gray-200/80 shadow-sm space-y-4">
                  <h2 className="text-base font-semibold text-gray-900 flex items-center space-x-2">
                    <Sliders className="w-4 h-4 text-violet-600" />
                    <span>ALiBi: Attention with Linear Biases</span>
                  </h2>
                  <p className="text-xs text-gray-500">
                    ALiBi, mutlak veya göreli pozisyon embedding vektörleri eklemek yerine doğrudan softmax öncesi attention skorlarına $m_h \cdot |i - j|$ doğrusal ceza uygular.
                  </p>

                  <div className="flex items-center space-x-2 overflow-x-auto pb-2">
                    {alibiData.heads.map((h) => (
                      <button
                        key={h.head_index}
                        onClick={() => setSelectedAlibiHead(h.head_index)}
                        className={`px-3 py-1.5 rounded-xl text-xs font-semibold transition-all ${
                          selectedAlibiHead === h.head_index
                            ? 'bg-violet-600 text-white shadow-sm'
                            : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                        }`}
                      >
                        Başlık #{h.head_index} (Eğim: {h.slope})
                      </button>
                    ))}
                  </div>

                  {/* Selected Head Matrix */}
                  <div className="bg-gray-50 p-4 rounded-xl border border-gray-200">
                    <span className="text-xs font-semibold text-gray-700 block mb-2 font-mono">
                      Başlık #{selectedAlibiHead} için Attention Ceza Matrisi (Eğim m = {alibiData.heads[selectedAlibiHead]?.slope})
                    </span>
                    <div className="overflow-x-auto">
                      <table className="w-full text-center font-mono text-xs">
                        <thead>
                          <tr>
                            <th className="p-1 text-gray-400">i \ j</th>
                            {alibiData.heads[selectedAlibiHead]?.bias_matrix.map((_, colIdx) => (
                              <th key={colIdx} className="p-1 text-gray-600">pos {colIdx}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {alibiData.heads[selectedAlibiHead]?.bias_matrix.map((row, rIdx) => (
                            <tr key={rIdx}>
                              <td className="p-1 font-bold text-gray-600 text-left">pos {rIdx}</td>
                              {row.map((val, cIdx) => (
                                <td
                                  key={cIdx}
                                  className={`p-1.5 border border-gray-200 rounded ${
                                    val === 0 ? 'bg-emerald-50 text-emerald-800 font-bold' : 'bg-gray-100 text-gray-800'
                                  }`}
                                >
                                  {val.toFixed(2)}
                                </td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* 1D: Compare View */}
            {peSubTab === 'compare' && (
              <div className="bg-white rounded-2xl p-6 border border-gray-200/80 shadow-sm space-y-4">
                <h2 className="text-base font-semibold text-gray-900 flex items-center space-x-2">
                  <Table className="w-4 h-4 text-violet-600" />
                  <span>Pozisyonel Kodlama Yöntemleri Karşılaştırma Atlası</span>
                </h2>

                <div className="overflow-x-auto">
                  <table className="w-full text-left text-xs sm:text-sm">
                    <thead>
                      <tr className="border-b border-gray-200 text-gray-500 uppercase font-mono text-xs">
                        <th className="py-2.5 px-3">Yöntem</th>
                        <th className="py-2.5 px-3">Tür</th>
                        <th className="py-2.5 px-3">Modern LLM Kullanımı</th>
                        <th className="py-2.5 px-3">Ekstrapolasyon</th>
                        <th className="py-2.5 px-3">Parametre Maliyeti</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-100">
                      {peComparison.map((m) => (
                        <tr key={m.name} className="hover:bg-gray-50/70">
                          <td className="py-3 px-3 font-semibold text-violet-900">
                            <div>{m.name}</div>
                            <span className="text-[11px] text-gray-400 font-normal font-mono">{m.authors}</span>
                          </td>
                          <td className="py-3 px-3 text-gray-700">{m.type}</td>
                          <td className="py-3 px-3 font-mono text-xs font-semibold text-gray-800">{m.modern_usage}</td>
                          <td className="py-3 px-3 text-emerald-700 font-medium">{m.extrapolation}</td>
                          <td className="py-3 px-3 font-mono text-xs text-gray-600">{m.parameter_overhead}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        )}

        {/* ================================================================== */}
        {/* TAB 2: MHA vs GQA vs MQA (KV-CACHE OPTIMIZATION)                   */}
        {/* ================================================================== */}
        {activeTab === 'attention_variants' && (
          <div className="space-y-6">
            <div className="bg-white rounded-2xl p-6 border border-gray-200/80 shadow-sm space-y-4">
              <h2 className="text-base font-semibold text-gray-900 flex items-center space-x-2">
                <Database className="w-4 h-4 text-violet-600" />
                <span>Grouped-Query Attention (GQA) & KV-Cache Bellek Simülatörü</span>
              </h2>
              <p className="text-xs text-gray-500">
                Orijinal Multi-Head Attention (MHA) her Query başlığı için ayrı bir Key ve Value başlığı saklarken,
                GQA birden fazla Query başlığının ortak bir KV çiftini paylaşmasını sağlayarak GPU VRAM ve bellek bant genişliği darboğazını çözer.
              </p>

              {/* Sliders Grid */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                <div>
                  <label className="text-xs font-semibold text-gray-500 block mb-1">
                    Batch Size ({kvBatchSize})
                  </label>
                  <input
                    type="range"
                    min={1}
                    max={16}
                    value={kvBatchSize}
                    onChange={(e) => {
                      const v = parseInt(e.target.value);
                      setKvBatchSize(v);
                      runKvCacheAnalysis(v, kvSeqLen, kvQHeads, kvKVHeads, kvHeadDim, kvLayers, kvDtype);
                    }}
                    className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-violet-600"
                  />
                </div>

                <div>
                  <label className="text-xs font-semibold text-gray-500 block mb-1">
                    Bağlam Uzunluğu ({kvSeqLen.toLocaleString()} Token)
                  </label>
                  <input
                    type="range"
                    min={512}
                    max={16384}
                    step={512}
                    value={kvSeqLen}
                    onChange={(e) => {
                      const v = parseInt(e.target.value);
                      setKvSeqLen(v);
                      runKvCacheAnalysis(kvBatchSize, v, kvQHeads, kvKVHeads, kvHeadDim, kvLayers, kvDtype);
                    }}
                    className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-violet-600"
                  />
                </div>

                <div>
                  <label className="text-xs font-semibold text-gray-500 block mb-1">
                    KV Başlık Sayısı ({kvKVHeads} / {kvQHeads} Q Başlığı)
                  </label>
                  <select
                    value={kvKVHeads}
                    onChange={(e) => {
                      const v = parseInt(e.target.value);
                      setKvKVHeads(v);
                      runKvCacheAnalysis(kvBatchSize, kvSeqLen, kvQHeads, v, kvHeadDim, kvLayers, kvDtype);
                    }}
                    className="w-full px-3 py-1.5 border rounded-xl text-xs bg-white font-medium"
                  >
                    <option value={32}>32 (MHA - 1:1 Tam Eşleşme)</option>
                    <option value={8}>8 (GQA 4:1 - LLaMA-3 / Mistral)</option>
                    <option value={4}>4 (GQA 8:1 - Yüksek Tasarruf)</option>
                    <option value={1}>1 (MQA - Multi-Query Attention)</option>
                  </select>
                </div>

                <div>
                  <label className="text-xs font-semibold text-gray-500 block mb-1">
                    Veri Tipi (Dtype)
                  </label>
                  <select
                    value={kvDtype}
                    onChange={(e) => {
                      setKvDtype(e.target.value);
                      runKvCacheAnalysis(kvBatchSize, kvSeqLen, kvQHeads, kvKVHeads, kvHeadDim, kvLayers, e.target.value);
                    }}
                    className="w-full px-3 py-1.5 border rounded-xl text-xs bg-white font-medium"
                  >
                    <option value="float16">FP16 / BF16 (2 Bytes)</option>
                    <option value="int8">INT8 Quantized (1 Byte)</option>
                    <option value="float32">FP32 (4 Bytes)</option>
                  </select>
                </div>
              </div>
            </div>

            {kvData && (
              <div className="space-y-6">
                {/* Metric Summary Cards */}
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                  <div className="p-4 bg-white rounded-2xl border border-gray-200/80 shadow-sm">
                    <span className="text-xs text-gray-500 font-medium">Mimari Sınıfı</span>
                    <p className="text-base font-bold text-violet-700 mt-1 truncate">
                      {kvData.variant_type}
                    </p>
                    <span className="text-[11px] text-gray-400 mt-0.5 block">
                      {kvData.queries_per_kv_head} Query / 1 KV Başlığı
                    </span>
                  </div>

                  <div className="p-4 bg-white rounded-2xl border border-gray-200/80 shadow-sm">
                    <span className="text-xs text-gray-500 font-medium">KV-Cache Boyutu</span>
                    <p className="text-xl font-bold font-mono text-gray-900 mt-1">
                      {kvData.current_kv_cache.formatted}
                    </p>
                    <span className="text-[11px] text-gray-400 mt-0.5 block">
                      {kvData.num_layers} Katman Toplamı
                    </span>
                  </div>

                  <div className="p-4 bg-white rounded-2xl border border-gray-200/80 shadow-sm">
                    <span className="text-xs text-gray-500 font-medium">MHA&apos;ya Göre Tasarruf</span>
                    <p className="text-xl font-bold font-mono text-emerald-600 mt-1">
                      %{kvData.comparison.savings_vs_mha_pct} ({kvData.comparison.savings_multiplier})
                    </p>
                    <span className="text-[11px] text-gray-400 mt-0.5 block">
                      MHA: {kvData.comparison.mha_baseline_formatted}
                    </span>
                  </div>

                  <div className="p-4 bg-white rounded-2xl border border-gray-200/80 shadow-sm">
                    <span className="text-xs text-gray-500 font-medium">Adım Başı Bellek Okuma</span>
                    <p className="text-xl font-bold font-mono text-amber-600 mt-1">
                      {kvData.memory_bandwidth.read_per_step_formatted}
                    </p>
                    <span className="text-[11px] text-gray-400 mt-0.5 block">
                      {kvData.memory_bandwidth.theoretical_throughput_a100}
                    </span>
                  </div>
                </div>

                {/* Head Groups Visualization */}
                <div className="bg-white rounded-2xl border border-gray-200/80 p-6 shadow-sm space-y-4">
                  <h3 className="text-sm font-semibold text-gray-900 flex items-center space-x-2">
                    <GitBranch className="w-4 h-4 text-violet-600" />
                    <span>Query Başlıklarının KV Başlıklarına Eşlenmesi (Gruplama Şeması)</span>
                  </h3>

                  <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-3">
                    {kvData.head_groups.map((g) => (
                      <div key={g.kv_head_index} className="p-3.5 bg-violet-50/60 border border-violet-100 rounded-xl space-y-2">
                        <div className="flex items-center justify-between">
                          <span className="text-xs font-bold text-violet-900">
                            KV Başlığı #{g.kv_head_index}
                          </span>
                          <span className="px-1.5 py-0.5 bg-violet-600 text-white rounded text-[10px] font-mono">
                            {g.query_count} Q Paylaşımı
                          </span>
                        </div>

                        <div className="flex flex-wrap gap-1">
                          {g.shared_query_heads.map((qIdx) => (
                            <span key={qIdx} className="px-1.5 py-0.5 bg-white border border-violet-200 rounded text-[10px] font-mono text-gray-700">
                              Q#{qIdx}
                            </span>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* ================================================================== */}
        {/* TAB 3: TRANSFORMER BLOCK INSPECTOR                                 */}
        {/* ================================================================== */}
        {activeTab === 'block' && (
          <div className="space-y-6">
            <div className="bg-white rounded-2xl p-6 border border-gray-200/80 shadow-sm space-y-4">
              <h2 className="text-base font-semibold text-gray-900 flex items-center space-x-2">
                <Box className="w-4 h-4 text-violet-600" />
                <span>Transformer Blok Mimarisi & Alt Katman Akışı</span>
              </h2>
              <p className="text-xs text-gray-500">
                Pre-LN (modern stabil eğitim) vs Post-LN, RMSNorm (ortalama çıkarma olmadan %50 daha hızlı) ve SwiGLU FFN katmanlarının tensör boyutlarını inceleyin.
              </p>

              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                <div>
                  <label className="text-xs font-semibold text-gray-500 block mb-1">
                    Normalizasyon Konumu
                  </label>
                  <select
                    value={blockNormPlacement}
                    onChange={(e) => {
                      setBlockNormPlacement(e.target.value);
                      runBlockSimulation(blockNormType, blockFfnType, e.target.value);
                    }}
                    className="w-full px-3 py-2 border rounded-xl text-xs font-medium bg-white"
                  >
                    <option value="pre_ln">Pre-LN (Modern GPT & LLaMA Standardı)</option>
                    <option value="post_ln">Post-LN (Orijinal 2017 Transformer)</option>
                  </select>
                </div>

                <div>
                  <label className="text-xs font-semibold text-gray-500 block mb-1">
                    Normalizasyon Katmanı
                  </label>
                  <select
                    value={blockNormType}
                    onChange={(e) => {
                      setBlockNormType(e.target.value);
                      runBlockSimulation(e.target.value, blockFfnType, blockNormPlacement);
                    }}
                    className="w-full px-3 py-2 border rounded-xl text-xs font-medium bg-white"
                  >
                    <option value="rmsnorm">RMSNorm (Root Mean Square - LLaMA)</option>
                    <option value="layernorm">LayerNorm (Standard - Mean & Variance)</option>
                  </select>
                </div>

                <div>
                  <label className="text-xs font-semibold text-gray-500 block mb-1">
                    İleri Besleme Ağı (FFN)
                  </label>
                  <select
                    value={blockFfnType}
                    onChange={(e) => {
                      setBlockFfnType(e.target.value);
                      runBlockSimulation(blockNormType, e.target.value, blockNormPlacement);
                    }}
                    className="w-full px-3 py-2 border rounded-xl text-xs font-medium bg-white"
                  >
                    <option value="swiglu">SwiGLU (SiLU Kapılı - 3 Matris)</option>
                    <option value="standard_mlp">Standart MLP (GELU/ReLU - 2 Matris)</option>
                  </select>
                </div>
              </div>
            </div>

            {blockData && (
              <div className="space-y-4">
                <h3 className="text-sm font-semibold text-gray-900">
                  Adım Adım Alt Katman Tensör Durumları ({blockData.stages.length} Aşama)
                </h3>

                <div className="space-y-3">
                  {blockData.stages.map((stage, idx) => (
                    <div
                      key={idx}
                      className="p-4 bg-white rounded-2xl border border-gray-200/80 shadow-sm flex flex-col md:flex-row md:items-center justify-between gap-3 hover:border-violet-300 transition-colors"
                    >
                      <div className="space-y-1">
                        <div className="flex items-center space-x-2">
                          <span className="w-6 h-6 rounded-full bg-violet-100 text-violet-700 text-xs font-bold flex items-center justify-center font-mono">
                            {idx + 1}
                          </span>
                          <span className="font-bold text-sm text-gray-900">
                            {stage.sublayer_name}
                          </span>
                        </div>
                        <p className="text-xs text-gray-500 pl-8">{stage.description}</p>
                      </div>

                      <div className="flex flex-wrap items-center gap-2 pl-8 md:pl-0 font-mono text-xs">
                        <span className="px-2.5 py-1 bg-gray-100 text-gray-800 rounded-lg">
                          Şekil: [{stage.shape.join(', ')}]
                        </span>
                        <span className="px-2.5 py-1 bg-violet-50 text-violet-800 rounded-lg">
                          Ort: {stage.mean}
                        </span>
                        <span className="px-2.5 py-1 bg-emerald-50 text-emerald-800 rounded-lg">
                          Std: {stage.std}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* ================================================================== */}
        {/* TAB 4: ARCHITECTURE PARAMETER & VRAM CALCULATOR                    */}
        {/* ================================================================== */}
        {activeTab === 'params' && (
          <div className="space-y-6">
            <div className="bg-white rounded-2xl p-6 border border-gray-200/80 shadow-sm space-y-4">
              <h2 className="text-base font-semibold text-gray-900 flex items-center space-x-2">
                <BarChart2 className="w-4 h-4 text-violet-600" />
                <span>Model Parametre Dağılımı & VRAM Hesaplayıcı</span>
              </h2>

              {/* Presets */}
              <div>
                <label className="text-xs font-semibold text-gray-500 uppercase tracking-wider block mb-1.5">
                  Ön Tanımlı Mimari Seç
                </label>
                <div className="flex flex-wrap gap-2">
                  {presets.map((p) => (
                    <button
                      key={p.name}
                      onClick={() => applyPreset(p)}
                      className={`px-3 py-1.5 text-xs rounded-xl font-semibold border transition-all ${
                        selectedPreset === p.name
                          ? 'bg-violet-600 text-white border-violet-600 shadow-sm'
                          : 'bg-gray-50 text-gray-700 hover:bg-gray-100 border-gray-200'
                      }`}
                    >
                      {p.name}
                    </button>
                  ))}
                </div>
              </div>

              {/* Custom Sliders */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                <div>
                  <label className="text-xs font-semibold text-gray-500 block mb-1">
                    d_model ({modelDModel})
                  </label>
                  <input
                    type="number"
                    value={modelDModel}
                    onChange={(e) => {
                      const v = parseInt(e.target.value) || 64;
                      setModelDModel(v);
                      runParamCalculation(modelVocab, v, modelLayers, modelHeads, modelKVHeads, modelDFF);
                    }}
                    className="w-full px-3 py-1.5 border rounded-xl font-mono text-xs"
                  />
                </div>

                <div>
                  <label className="text-xs font-semibold text-gray-500 block mb-1">
                    Katman Sayısı ({modelLayers})
                  </label>
                  <input
                    type="number"
                    value={modelLayers}
                    onChange={(e) => {
                      const v = parseInt(e.target.value) || 1;
                      setModelLayers(v);
                      runParamCalculation(modelVocab, modelDModel, v, modelHeads, modelKVHeads, modelDFF);
                    }}
                    className="w-full px-3 py-1.5 border rounded-xl font-mono text-xs"
                  />
                </div>

                <div>
                  <label className="text-xs font-semibold text-gray-500 block mb-1">
                    d_ff ({modelDFF})
                  </label>
                  <input
                    type="number"
                    value={modelDFF}
                    onChange={(e) => {
                      const v = parseInt(e.target.value) || 64;
                      setModelDFF(v);
                      runParamCalculation(modelVocab, modelDModel, modelLayers, modelHeads, modelKVHeads, v);
                    }}
                    className="w-full px-3 py-1.5 border rounded-xl font-mono text-xs"
                  />
                </div>

                <div>
                  <label className="text-xs font-semibold text-gray-500 block mb-1">
                    Kelime Dağarcığı (Vocab: {modelVocab.toLocaleString()})
                  </label>
                  <input
                    type="number"
                    value={modelVocab}
                    onChange={(e) => {
                      const v = parseInt(e.target.value) || 1000;
                      setModelVocab(v);
                      runParamCalculation(v, modelDModel, modelLayers, modelHeads, modelKVHeads, modelDFF);
                    }}
                    className="w-full px-3 py-1.5 border rounded-xl font-mono text-xs"
                  />
                </div>
              </div>
            </div>

            {paramsData && (
              <div className="space-y-6">
                {/* Metric Cards */}
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                  <div className="p-4 bg-white rounded-2xl border border-gray-200/80 shadow-sm">
                    <span className="text-xs text-gray-500 font-medium">Toplam Parametre</span>
                    <p className="text-2xl font-bold font-mono text-violet-700 mt-1">
                      {paramsData.total_billions > 0.5
                        ? `${paramsData.total_billions} Milyar (B)`
                        : `${paramsData.total_millions} Milyon (M)`}
                    </p>
                    <span className="text-[11px] text-gray-400 mt-0.5 block">
                      {paramsData.total_parameters.toLocaleString()} Ağırlık
                    </span>
                  </div>

                  <div className="p-4 bg-white rounded-2xl border border-gray-200/80 shadow-sm">
                    <span className="text-xs text-gray-500 font-medium">FP16 Çıkarım VRAM</span>
                    <p className="text-xl font-bold font-mono text-gray-900 mt-1">
                      {paramsData.vram_inference.fp16}
                    </p>
                    <span className="text-[11px] text-gray-400 mt-0.5 block">
                      16-bit float model ağırlıkları
                    </span>
                  </div>

                  <div className="p-4 bg-white rounded-2xl border border-gray-200/80 shadow-sm">
                    <span className="text-xs text-gray-500 font-medium">INT4 Kuantize VRAM</span>
                    <p className="text-xl font-bold font-mono text-emerald-600 mt-1">
                      {paramsData.vram_inference.int4_quantized}
                    </p>
                    <span className="text-[11px] text-gray-400 mt-0.5 block">
                      AWQ / GPTQ ile 4x sıkıştırma
                    </span>
                  </div>

                  <div className="p-4 bg-white rounded-2xl border border-gray-200/80 shadow-sm">
                    <span className="text-xs text-gray-500 font-medium">AdamW Eğitim VRAM</span>
                    <p className="text-xl font-bold font-mono text-amber-600 mt-1">
                      {paramsData.vram_training_adamw}
                    </p>
                    <span className="text-[11px] text-gray-400 mt-0.5 block">
                      Model + Gradyanlar + Optimizer Durumları
                    </span>
                  </div>
                </div>

                {/* Percentage Distribution Bar */}
                <div className="bg-white rounded-2xl border border-gray-200/80 p-6 shadow-sm space-y-4">
                  <h3 className="text-sm font-semibold text-gray-900">
                    Parametre Yüzde Dağılımı
                  </h3>

                  <div className="h-4 w-full rounded-full overflow-hidden flex">
                    <div
                      className="bg-violet-600 h-full"
                      style={{ width: `${paramsData.percentages.ffn_pct}%` }}
                      title={`FFN: %${paramsData.percentages.ffn_pct}`}
                    />
                    <div
                      className="bg-indigo-500 h-full"
                      style={{ width: `${paramsData.percentages.attention_pct}%` }}
                      title={`Attention: %${paramsData.percentages.attention_pct}`}
                    />
                    <div
                      className="bg-amber-400 h-full"
                      style={{ width: `${paramsData.percentages.embeddings_pct}%` }}
                      title={`Embeddings: %${paramsData.percentages.embeddings_pct}`}
                    />
                  </div>

                  <div className="flex flex-wrap gap-4 text-xs font-mono">
                    <span className="flex items-center space-x-1.5">
                      <span className="w-3 h-3 rounded bg-violet-600 inline-block"></span>
                      <span>Feed-Forward Ağı (%{paramsData.percentages.ffn_pct})</span>
                    </span>
                    <span className="flex items-center space-x-1.5">
                      <span className="w-3 h-3 rounded bg-indigo-500 inline-block"></span>
                      <span>Attention Projeksiyonları (%{paramsData.percentages.attention_pct})</span>
                    </span>
                    <span className="flex items-center space-x-1.5">
                      <span className="w-3 h-3 rounded bg-amber-400 inline-block"></span>
                      <span>Token & LM Head Embeddings (%{paramsData.percentages.embeddings_pct})</span>
                    </span>
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
