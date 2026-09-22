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
  XCircle,
  Info,
  Maximize2,
  GitBranch,
  TrendingUp,
  Table,
  HelpCircle,
  ChevronRight,
  Hash,
  Play
} from 'lucide-react';
import {
  tensorLabApi,
  ShapeAnalyzeResponse,
  ReshapeResponse,
  TransposeResponse,
  BroadcastCheckResponse,
  BroadcastSimulateResponse,
  MatMulResponse,
  ActivationCurveResponse,
  SoftmaxTemperatureResponse,
  AutogradResponse,
} from '@/lib/api';

const SHAPE_PRESETS = [
  { name: 'LLaMA Attention Head', shape: [1, 32, 2048, 128], desc: '[Batch, Heads, SeqLen, HeadDim]' },
  { name: 'BERT Hidden Sequence', shape: [16, 512, 768], desc: '[Batch, SeqLen, HiddenDim]' },
  { name: 'ViT Image Patches', shape: [8, 197, 768], desc: '[Batch, NumPatches+1, EmbedDim]' },
  { name: 'Linear Projection Weights', shape: [4096, 11008], desc: '[HiddenDim, IntermediateDim]' },
  { name: 'Vocabulary Logits', shape: [1, 32000], desc: '[Batch, VocabSize]' },
];

const BROADCAST_PRESETS = [
  { name: 'Batch Bias Ekleme', a: [3, 1], b: [1, 4], desc: '(3, 1) + (1, 4) -> (3, 4)' },
  { name: 'Transformer Hidden + Bias', a: [16, 512, 768], b: [768], desc: '(16, 512, 768) + (768) -> (16, 512, 768)' },
  { name: 'Attention Mask Broadcast', a: [1, 32, 128, 128], b: [1, 1, 128, 128], desc: 'Multi-head mask broadcast' },
  { name: 'Uyumsuz Boyutlar (Hata)', a: [3, 2], b: [3, 3], desc: 'Eşleşmeyen ve 1 olmayan boyut hatası' },
];

export default function TensorLabPage() {
  const [activeTab, setActiveTab] = useState<'shape' | 'broadcast' | 'matmul' | 'activation'>('shape');

  // --------------------------------------------------------------------------
  // Tab 1: Shape & Memory State
  // --------------------------------------------------------------------------
  const [inputShapeStr, setInputShapeStr] = useState('1, 32, 512, 64');
  const [selectedDtype, setSelectedDtype] = useState('float32');
  const [shapeAnalysis, setShapeAnalysis] = useState<ShapeAnalyzeResponse | null>(null);
  const [shapeLoading, setShapeLoading] = useState(false);
  const [shapeError, setShapeError] = useState<string | null>(null);

  // Reshape simulator state
  const [targetReshapeStr, setTargetReshapeStr] = useState('32, -1');
  const [reshapeResult, setReshapeResult] = useState<ReshapeResponse | null>(null);
  const [reshapeError, setReshapeError] = useState<string | null>(null);

  // Transpose simulator state
  const [permutationStr, setPermutationStr] = useState('0, 2, 1, 3');
  const [transposeResult, setTransposeResult] = useState<TransposeResponse | null>(null);
  const [transposeError, setTransposeError] = useState<string | null>(null);

  // --------------------------------------------------------------------------
  // Tab 2: Broadcasting State
  // --------------------------------------------------------------------------
  const [shapeAStr, setShapeAStr] = useState('3, 1');
  const [shapeBStr, setShapeBStr] = useState('1, 4');
  const [broadcastAnalysis, setBroadcastAnalysis] = useState<BroadcastCheckResponse | null>(null);
  const [broadcastLoading, setBroadcastLoading] = useState(false);

  // 2D Matrix Broadcast Simulation
  const [matrixAInput, setMatrixAInput] = useState('[[1], [2], [3]]');
  const [matrixBInput, setMatrixBInput] = useState('[[10, 20, 30, 40]]');
  const [broadcastOp, setBroadcastOp] = useState<'add' | 'mul' | 'sub'>('add');
  const [broadcastSimResult, setBroadcastSimResult] = useState<BroadcastSimulateResponse | null>(null);
  const [broadcastSimError, setBroadcastSimError] = useState<string | null>(null);

  // --------------------------------------------------------------------------
  // Tab 3: MatMul (GEMM) State
  // --------------------------------------------------------------------------
  const [dimM, setDimM] = useState(3);
  const [dimK, setDimK] = useState(3);
  const [dimN, setDimN] = useState(3);
  const [matmulPreset, setMatmulPreset] = useState('simple');
  const [matrixA, setMatrixA] = useState<number[][]>([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
  const [matrixB, setMatrixB] = useState<number[][]>([[9, 8, 7], [6, 5, 4], [3, 2, 1]]);
  const [selectedCell, setSelectedCell] = useState<{ row: number; col: number }>({ row: 0, col: 0 });
  const [matmulResult, setMatmulResult] = useState<MatMulResponse | null>(null);
  const [matmulLoading, setMatmulLoading] = useState(false);
  const [matmulError, setMatmulError] = useState<string | null>(null);

  // --------------------------------------------------------------------------
  // Tab 4: Activation & Autograd State
  // --------------------------------------------------------------------------
  const [selectedActivation, setSelectedActivation] = useState('gelu');
  const [curveTemperature, setCurveTemperature] = useState(1.0);
  const [curveData, setCurveData] = useState<ActivationCurveResponse | null>(null);
  const [curveLoading, setCurveLoading] = useState(false);

  // Softmax Logits
  const [logitsInputStr, setLogitsInputStr] = useState('3.2, 1.8, 0.4, -0.9, -2.1');
  const [softmaxTemp, setSoftmaxTemp] = useState(1.0);
  const [softmaxResult, setSoftmaxResult] = useState<SoftmaxTemperatureResponse | null>(null);

  // Autograd Graph
  const [autogradX, setAutogradX] = useState('1.0, -0.5');
  const [autogradY, setAutogradY] = useState('1.0');
  const [autogradHiddenDim, setAutogradHiddenDim] = useState(3);
  const [autogradAct, setAutogradAct] = useState('relu');
  const [autogradResult, setAutogradResult] = useState<AutogradResponse | null>(null);
  const [autogradLoading, setAutogradLoading] = useState(false);

  // --------------------------------------------------------------------------
  // Actions: Shape Tab
  // --------------------------------------------------------------------------
  const runShapeAnalysis = async (customShape?: number[], customDtype?: string) => {
    setShapeLoading(true);
    setShapeError(null);
    try {
      let shapeArr: number[];
      if (customShape) {
        shapeArr = customShape;
      } else {
        shapeArr = inputShapeStr
          .split(',')
          .map((s) => parseInt(s.trim()))
          .filter((n) => !isNaN(n));
      }

      if (shapeArr.length === 0) {
        throw new Error('Lütfen geçerli tensör boyutları girin (ör: 1, 32, 512, 64)');
      }

      const res = await tensorLabApi.analyzeShape({
        shape: shapeArr,
        dtype: customDtype || selectedDtype,
      });
      setShapeAnalysis(res);
    } catch (err: any) {
      setShapeError(err.response?.data?.detail || err.message || 'Analiz hatası');
    } finally {
      setShapeLoading(false);
    }
  };

  const runReshapeSimulation = async () => {
    if (!shapeAnalysis) return;
    setReshapeError(null);
    try {
      const targetArr = targetReshapeStr
        .split(',')
        .map((s) => parseInt(s.trim()))
        .filter((n) => !isNaN(n));
      if (targetArr.length === 0) throw new Error('Hedef şekil giriniz');
      const res = await tensorLabApi.reshape({
        original_shape: shapeAnalysis.shape,
        target_shape: targetArr,
      });
      setReshapeResult(res);
    } catch (err: any) {
      setReshapeError(err.response?.data?.detail || err.message || 'Reshape hatası');
      setReshapeResult(null);
    }
  };

  const runTransposeSimulation = async () => {
    if (!shapeAnalysis) return;
    setTransposeError(null);
    try {
      const permArr = permutationStr
        .split(',')
        .map((s) => parseInt(s.trim()))
        .filter((n) => !isNaN(n));
      const res = await tensorLabApi.transpose({
        shape: shapeAnalysis.shape,
        permutation: permArr,
      });
      setTransposeResult(res);
    } catch (err: any) {
      setTransposeError(err.response?.data?.detail || err.message || 'Transpoze hatası');
      setTransposeResult(null);
    }
  };

  // --------------------------------------------------------------------------
  // Actions: Broadcast Tab
  // --------------------------------------------------------------------------
  const runBroadcastAnalysis = async (sA?: number[], sB?: number[]) => {
    setBroadcastLoading(true);
    try {
      const shapeA = sA || shapeAStr.split(',').map((s) => parseInt(s.trim())).filter((n) => !isNaN(n));
      const shapeB = sB || shapeBStr.split(',').map((s) => parseInt(s.trim())).filter((n) => !isNaN(n));
      const res = await tensorLabApi.checkBroadcast(shapeA, shapeB);
      setBroadcastAnalysis(res);
    } catch (err) {
      console.error(err);
    } finally {
      setBroadcastLoading(false);
    }
  };

  const runBroadcastSimulation = async () => {
    setBroadcastSimError(null);
    try {
      const mA = JSON.parse(matrixAInput);
      const mB = JSON.parse(matrixBInput);
      const res = await tensorLabApi.simulateBroadcast(mA, mB, broadcastOp);
      setBroadcastSimResult(res);
    } catch (err: any) {
      setBroadcastSimError(err.response?.data?.detail || err.message || 'Matris parse veya simülasyon hatası');
      setBroadcastSimResult(null);
    }
  };

  // --------------------------------------------------------------------------
  // Actions: MatMul Tab
  // --------------------------------------------------------------------------
  const fetchSampleMatrices = async (m = dimM, k = dimK, n = dimN, preset = matmulPreset) => {
    try {
      const res = await tensorLabApi.getMatmulSample(m, k, n, preset);
      setMatrixA(res.matrix_a);
      setMatrixB(res.matrix_b);
      setSelectedCell({ row: 0, col: 0 });
      executeMatMul(res.matrix_a, res.matrix_b, 0, 0);
    } catch (err) {
      console.error(err);
    }
  };

  const executeMatMul = async (mA = matrixA, mB = matrixB, r = selectedCell.row, c = selectedCell.col) => {
    setMatmulLoading(true);
    setMatmulError(null);
    try {
      const res = await tensorLabApi.matmul(mA, mB, r, c);
      setMatmulResult(res);
    } catch (err: any) {
      setMatmulError(err.response?.data?.detail || err.message || 'Matris çarpımı hatası');
    } finally {
      setMatmulLoading(false);
    }
  };

  const handleCellClick = (row: number, col: number) => {
    setSelectedCell({ row, col });
    if (matrixA.length > 0 && matrixB.length > 0) {
      executeMatMul(matrixA, matrixB, row, col);
    }
  };

  // --------------------------------------------------------------------------
  // Actions: Activation & Autograd Tab
  // --------------------------------------------------------------------------
  const fetchActivationCurve = async (act = selectedActivation, temp = curveTemperature) => {
    setCurveLoading(true);
    try {
      const res = await tensorLabApi.getActivationCurve(act, 81, temp);
      setCurveData(res);
    } catch (err) {
      console.error(err);
    } finally {
      setCurveLoading(false);
    }
  };

  const runSoftmaxSimulation = async (temp = softmaxTemp) => {
    try {
      const logits = logitsInputStr
        .split(',')
        .map((s) => parseFloat(s.trim()))
        .filter((n) => !isNaN(n));
      const res = await tensorLabApi.computeSoftmaxTemperature(logits, temp);
      setSoftmaxResult(res);
    } catch (err) {
      console.error(err);
    }
  };

  const runAutogradSimulation = async () => {
    setAutogradLoading(true);
    try {
      const x = autogradX.split(',').map((s) => parseFloat(s.trim())).filter((n) => !isNaN(n));
      const y = autogradY.split(',').map((s) => parseFloat(s.trim())).filter((n) => !isNaN(n));
      const res = await tensorLabApi.simulateAutograd(x, y, autogradHiddenDim, autogradAct, 42);
      setAutogradResult(res);
    } catch (err) {
      console.error(err);
    } finally {
      setAutogradLoading(false);
    }
  };

  // Initial loads
  useEffect(() => {
    runShapeAnalysis([1, 32, 512, 64], 'float32');
    runBroadcastAnalysis([3, 1], [1, 4]);
    fetchSampleMatrices(3, 3, 3, 'simple');
    fetchActivationCurve('gelu', 1.0);
    runSoftmaxSimulation(1.0);
    runAutogradSimulation();
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-50 via-white to-gray-50 text-gray-900 pb-16">
      {/* Top Header Banner */}
      <div className="border-b border-gray-200/80 bg-white/70 backdrop-blur-md sticky top-16 z-40">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
            <div>
              <div className="flex items-center space-x-2.5">
                <span className="text-2xl p-2 rounded-xl bg-indigo-50 border border-indigo-100 shadow-sm">
                  🧮
                </span>
                <div>
                  <h1 className="text-xl sm:text-2xl font-bold tracking-tight text-gray-900">
                    Tensor & Math Lab
                  </h1>
                  <p className="text-xs sm:text-sm text-gray-500">
                    Tensör Boyutları, Bellek Ayak İzi, Broadcasting, GEMM & Autograd Geri Yayılım Simülatörü
                  </p>
                </div>
              </div>
            </div>

            {/* Tab Buttons */}
            <div className="flex items-center bg-gray-100/80 p-1 rounded-xl border border-gray-200 text-xs sm:text-sm font-medium overflow-x-auto">
              <button
                onClick={() => setActiveTab('shape')}
                className={`flex items-center space-x-1.5 px-3 py-2 rounded-lg transition-all ${
                  activeTab === 'shape'
                    ? 'bg-white text-indigo-700 font-semibold shadow-sm border border-gray-200/50'
                    : 'text-gray-600 hover:text-gray-900'
                }`}
              >
                <Layers className="w-3.5 h-3.5" />
                <span>1. Şekil & Bellek</span>
              </button>

              <button
                onClick={() => setActiveTab('broadcast')}
                className={`flex items-center space-x-1.5 px-3 py-2 rounded-lg transition-all ${
                  activeTab === 'broadcast'
                    ? 'bg-white text-indigo-700 font-semibold shadow-sm border border-gray-200/50'
                    : 'text-gray-600 hover:text-gray-900'
                }`}
              >
                <GitBranch className="w-3.5 h-3.5" />
                <span>2. Broadcasting</span>
              </button>

              <button
                onClick={() => setActiveTab('matmul')}
                className={`flex items-center space-x-1.5 px-3 py-2 rounded-lg transition-all ${
                  activeTab === 'matmul'
                    ? 'bg-white text-indigo-700 font-semibold shadow-sm border border-gray-200/50'
                    : 'text-gray-600 hover:text-gray-900'
                }`}
              >
                <Table className="w-3.5 h-3.5" />
                <span>3. GEMM MatMul</span>
              </button>

              <button
                onClick={() => setActiveTab('activation')}
                className={`flex items-center space-x-1.5 px-3 py-2 rounded-lg transition-all ${
                  activeTab === 'activation'
                    ? 'bg-white text-indigo-700 font-semibold shadow-sm border border-gray-200/50'
                    : 'text-gray-600 hover:text-gray-900'
                }`}
              >
                <TrendingUp className="w-3.5 h-3.5" />
                <span>4. Aktivasyon & Autograd</span>
              </button>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 mt-6">
        {/* ================================================================== */}
        {/* TAB 1: SHAPE & MEMORY FOOTPRINT                                    */}
        {/* ================================================================== */}
        {activeTab === 'shape' && (
          <div className="space-y-6">
            {/* Input Controls Panel */}
            <div className="bg-white rounded-2xl p-6 border border-gray-200/80 shadow-sm">
              <h2 className="text-base font-semibold text-gray-900 mb-3 flex items-center space-x-2">
                <Sliders className="w-4 h-4 text-indigo-600" />
                <span>Tensör Parametreleri</span>
              </h2>

              {/* Presets */}
              <div className="mb-4">
                <label className="text-xs font-semibold text-gray-500 uppercase tracking-wider block mb-1.5">
                  Ön Tanımlı Derin Öğrenme Şekilleri
                </label>
                <div className="flex flex-wrap gap-2">
                  {SHAPE_PRESETS.map((p) => (
                    <button
                      key={p.name}
                      onClick={() => {
                        const str = p.shape.join(', ');
                        setInputShapeStr(str);
                        runShapeAnalysis(p.shape);
                        setTargetReshapeStr(`${p.shape[0]}, -1`);
                        setPermutationStr(p.shape.map((_, i) => i).reverse().join(', '));
                      }}
                      className="px-2.5 py-1.5 text-xs bg-gray-50 hover:bg-indigo-50 hover:text-indigo-700 hover:border-indigo-200 border border-gray-200 rounded-lg text-gray-700 transition-colors text-left"
                    >
                      <span className="font-semibold">{p.name}</span>
                      <span className="text-gray-400 ml-1.5 font-mono">[{p.shape.join(', ')}]</span>
                    </button>
                  ))}
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="md:col-span-2">
                  <label className="text-xs font-semibold text-gray-500 uppercase tracking-wider block mb-1.5">
                    Boyutlar [d0, d1, d2, ...]
                  </label>
                  <input
                    type="text"
                    value={inputShapeStr}
                    onChange={(e) => setInputShapeStr(e.target.value)}
                    placeholder="ör: 1, 32, 512, 64"
                    className="w-full px-3.5 py-2.5 rounded-xl border border-gray-200 font-mono text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                  />
                </div>

                <div>
                  <label className="text-xs font-semibold text-gray-500 uppercase tracking-wider block mb-1.5">
                    Veri Tipi (Dtype)
                  </label>
                  <select
                    value={selectedDtype}
                    onChange={(e) => {
                      setSelectedDtype(e.target.value);
                      runShapeAnalysis(undefined, e.target.value);
                    }}
                    className="w-full px-3.5 py-2.5 rounded-xl border border-gray-200 text-sm font-medium focus:outline-none focus:ring-2 focus:ring-indigo-500 bg-white"
                  >
                    <option value="float32">float32 (FP32 - 4 Bytes)</option>
                    <option value="float16">float16 (FP16 - 2 Bytes)</option>
                    <option value="bfloat16">bfloat16 (BF16 - 2 Bytes)</option>
                    <option value="int8">int8 (INT8 - 1 Byte)</option>
                    <option value="int64">int64 (Long - 8 Bytes)</option>
                  </select>
                </div>
              </div>

              <div className="mt-4 flex justify-end">
                <button
                  onClick={() => runShapeAnalysis()}
                  disabled={shapeLoading}
                  className="px-5 py-2.5 bg-indigo-600 hover:bg-indigo-700 text-white rounded-xl text-sm font-semibold shadow-sm transition-all flex items-center space-x-2"
                >
                  {shapeLoading ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
                  <span>Analiz Et</span>
                </button>
              </div>

              {shapeError && (
                <div className="mt-3 p-3 bg-red-50 text-red-700 border border-red-200 rounded-xl text-xs flex items-center space-x-2">
                  <XCircle className="w-4 h-4 flex-shrink-0" />
                  <span>{shapeError}</span>
                </div>
              )}
            </div>

            {/* Results Grid */}
            {shapeAnalysis && (
              <div className="space-y-6">
                {/* Metric Summary Cards */}
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                  <div className="p-4 bg-white rounded-2xl border border-gray-200/80 shadow-sm">
                    <span className="text-xs text-gray-500 font-medium">Toplam Eleman</span>
                    <p className="text-xl font-bold font-mono text-gray-900 mt-1">
                      {shapeAnalysis.total_elements.toLocaleString()}
                    </p>
                    <span className="text-[11px] text-gray-400 mt-0.5 block">
                      Rank {shapeAnalysis.rank}D
                    </span>
                  </div>

                  <div className="p-4 bg-white rounded-2xl border border-gray-200/80 shadow-sm">
                    <span className="text-xs text-gray-500 font-medium">Bellek Ayak İzi</span>
                    <p className="text-xl font-bold font-mono text-indigo-600 mt-1">
                      {shapeAnalysis.formatted_memory}
                    </p>
                    <span className="text-[11px] text-gray-400 mt-0.5 block">
                      {shapeAnalysis.total_bytes.toLocaleString()} Bytes ({shapeAnalysis.dtype.toUpperCase()})
                    </span>
                  </div>

                  <div className="p-4 bg-white rounded-2xl border border-gray-200/80 shadow-sm">
                    <span className="text-xs text-gray-500 font-medium">Strides (Adımlar)</span>
                    <p className="text-sm font-bold font-mono text-gray-900 mt-1 truncate">
                      [{shapeAnalysis.element_strides.join(', ')}]
                    </p>
                    <span className="text-[11px] text-gray-400 mt-0.5 block">
                      Eleman bazında C-Contiguous
                    </span>
                  </div>

                  <div className="p-4 bg-white rounded-2xl border border-gray-200/80 shadow-sm">
                    <span className="text-xs text-gray-500 font-medium">Süreklilik (Contiguity)</span>
                    <p className="text-base font-bold text-emerald-600 mt-1 flex items-center space-x-1">
                      <CheckCircle2 className="w-4 h-4" />
                      <span>C-Contiguous</span>
                    </p>
                    <span className="text-[11px] text-gray-400 mt-0.5 block">
                      Bellek kopyası gerektirmez
                    </span>
                  </div>
                </div>

                {/* Semantic Hint Box */}
                <div className="p-4 bg-indigo-50/70 border border-indigo-100 rounded-2xl flex items-start space-x-3 text-xs text-indigo-900">
                  <Info className="w-5 h-5 text-indigo-600 flex-shrink-0 mt-0.5" />
                  <div>
                    <span className="font-semibold block text-indigo-950">Semantik Model Yorumu:</span>
                    <p className="mt-0.5">{shapeAnalysis.semantic_interpretation}</p>
                    <p className="text-indigo-700/80 mt-1">{shapeAnalysis.dtype_description}</p>
                  </div>
                </div>

                {/* Multi-Dtype Comparison Table */}
                <div className="bg-white rounded-2xl border border-gray-200/80 p-6 shadow-sm">
                  <h3 className="text-sm font-semibold text-gray-900 mb-3 flex items-center space-x-2">
                    <Cpu className="w-4 h-4 text-indigo-600" />
                    <span>Farklı Veri Tiplerinde Bellek Tüketimi & LLM Tasarrufu</span>
                  </h3>

                  <div className="overflow-x-auto">
                    <table className="w-full text-left text-xs sm:text-sm">
                      <thead>
                        <tr className="border-b border-gray-200 text-gray-500 text-xs uppercase font-mono">
                          <th className="py-2.5 px-3">Veri Tipi</th>
                          <th className="py-2.5 px-3">Bayt / Eleman</th>
                          <th className="py-2.5 px-3">Toplam Boyut</th>
                          <th className="py-2.5 px-3">FP32 Oranı</th>
                          <th className="py-2.5 px-3">GPU Tasarrufu</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-gray-100 font-mono">
                        {Object.entries(shapeAnalysis.memory_comparison).map(([dt, data]) => {
                          const isCurrent = dt === shapeAnalysis.dtype;
                          const saving = Math.round((1 - data.ratio_vs_fp32) * 100);
                          return (
                            <tr
                              key={dt}
                              className={`hover:bg-gray-50/80 transition-colors ${
                                isCurrent ? 'bg-indigo-50/50 font-semibold' : ''
                              }`}
                            >
                              <td className="py-3 px-3 flex items-center space-x-2">
                                <span className="uppercase text-gray-800">{dt}</span>
                                {isCurrent && (
                                  <span className="px-2 py-0.5 text-[10px] bg-indigo-600 text-white rounded-md font-sans">
                                    Aktif
                                  </span>
                                )}
                              </td>
                              <td className="py-3 px-3 text-gray-600">
                                {dt.includes('64') ? 8 : dt.includes('32') ? 4 : dt.includes('16') ? 2 : 1} B
                              </td>
                              <td className="py-3 px-3 font-bold text-gray-900">{data.formatted}</td>
                              <td className="py-3 px-3 text-gray-600">{(data.ratio_vs_fp32 * 100).toFixed(0)}%</td>
                              <td className="py-3 px-3 font-sans">
                                {saving > 0 ? (
                                  <span className="text-emerald-600 font-semibold">%{saving} Tasarruf</span>
                                ) : saving < 0 ? (
                                  <span className="text-amber-600 font-semibold">%{Math.abs(saving)} Artış</span>
                                ) : (
                                  <span className="text-gray-400">Referans</span>
                                )}
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </div>

                {/* Reshape & Transpose Sandbox */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {/* Reshape Box */}
                  <div className="bg-white rounded-2xl border border-gray-200/80 p-5 shadow-sm space-y-3">
                    <h3 className="text-sm font-semibold text-gray-900 flex items-center space-x-2">
                      <Maximize2 className="w-4 h-4 text-indigo-600" />
                      <span>Reshape Simülasyonu (-1 İndeksi)</span>
                    </h3>
                    <p className="text-xs text-gray-500">
                      Orijinal eleman sayısını koruyan yeni şekli test edin. Otomatik boyut hesabı için tek bir <code>-1</code> kullanabilirsiniz.
                    </p>

                    <div className="flex space-x-2">
                      <input
                        type="text"
                        value={targetReshapeStr}
                        onChange={(e) => setTargetReshapeStr(e.target.value)}
                        placeholder="ör: 32, -1"
                        className="flex-1 px-3 py-2 rounded-xl border border-gray-200 text-sm font-mono focus:outline-none focus:ring-2 focus:ring-indigo-500"
                      />
                      <button
                        onClick={runReshapeSimulation}
                        className="px-4 py-2 bg-indigo-50 text-indigo-700 hover:bg-indigo-100 font-semibold text-xs rounded-xl transition-colors"
                      >
                        Uygula
                      </button>
                    </div>

                    {reshapeResult && (
                      <div className="p-3 bg-emerald-50 border border-emerald-200 rounded-xl text-xs text-emerald-900 space-y-1">
                        <span className="font-semibold block flex items-center space-x-1">
                          <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
                          <span>Çözümlenen Şekil: [{reshapeResult.resolved_shape.join(', ')}]</span>
                        </span>
                        <p className="text-emerald-700 font-mono text-[11px]">
                          Yeni Strides: [{reshapeResult.new_strides.join(', ')}]
                        </p>
                        <p className="text-emerald-800">{reshapeResult.explanation}</p>
                      </div>
                    )}

                    {reshapeError && (
                      <div className="p-3 bg-red-50 border border-red-200 rounded-xl text-xs text-red-700 flex items-center space-x-1.5">
                        <XCircle className="w-4 h-4 flex-shrink-0" />
                        <span>{reshapeError}</span>
                      </div>
                    )}
                  </div>

                  {/* Transpose Box */}
                  <div className="bg-white rounded-2xl border border-gray-200/80 p-5 shadow-sm space-y-3">
                    <h3 className="text-sm font-semibold text-gray-900 flex items-center space-x-2">
                      <Layers className="w-4 h-4 text-indigo-600" />
                      <span>Transpose & Permütasyon (Contiguity)</span>
                    </h3>
                    <p className="text-xs text-gray-500">
                      Boyut indislerini değiştirerek transpoze yapın ve bellekteki süreklilik (contiguity) durumunu inceleyin.
                    </p>

                    <div className="flex space-x-2">
                      <input
                        type="text"
                        value={permutationStr}
                        onChange={(e) => setPermutationStr(e.target.value)}
                        placeholder={`ör: ${shapeAnalysis.shape.map((_, i) => i).reverse().join(', ')}`}
                        className="flex-1 px-3 py-2 rounded-xl border border-gray-200 text-sm font-mono focus:outline-none focus:ring-2 focus:ring-indigo-500"
                      />
                      <button
                        onClick={runTransposeSimulation}
                        className="px-4 py-2 bg-indigo-50 text-indigo-700 hover:bg-indigo-100 font-semibold text-xs rounded-xl transition-colors"
                      >
                        Permüte Et
                      </button>
                    </div>

                    {transposeResult && (
                      <div
                        className={`p-3 rounded-xl border text-xs space-y-1 ${
                          transposeResult.is_contiguous
                            ? 'bg-emerald-50 border-emerald-200 text-emerald-900'
                            : 'bg-amber-50 border-amber-200 text-amber-900'
                        }`}
                      >
                        <span className="font-semibold block">
                          Yeni Şekil: [{transposeResult.transposed_shape.join(', ')}]
                        </span>
                        <p className="font-mono text-[11px]">
                          Transpoze Strides: [{transposeResult.transposed_strides.join(', ')}]
                        </p>
                        <p className="font-sans text-[11px] font-medium">
                          {transposeResult.note}
                        </p>
                      </div>
                    )}

                    {transposeError && (
                      <div className="p-3 bg-red-50 border border-red-200 rounded-xl text-xs text-red-700 flex items-center space-x-1.5">
                        <XCircle className="w-4 h-4 flex-shrink-0" />
                        <span>{transposeError}</span>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* ================================================================== */}
        {/* TAB 2: BROADCASTING RULES                                          */}
        {/* ================================================================== */}
        {activeTab === 'broadcast' && (
          <div className="space-y-6">
            {/* Input shapes */}
            <div className="bg-white rounded-2xl p-6 border border-gray-200/80 shadow-sm">
              <h2 className="text-base font-semibold text-gray-900 mb-2 flex items-center space-x-2">
                <GitBranch className="w-4 h-4 text-indigo-600" />
                <span>Broadcasting Kuralları (Sağdan Sola Eşleştirme)</span>
              </h2>
              <p className="text-xs text-gray-500 mb-4">
                NumPy ve PyTorch, farklı boyutlu tensörleri işlerken en sağdaki eksenlerden başlayarak sola doğru kontrol eder.
                Her eksende boyutlar eşit olmalı ya da en az biri <code>1</code> olmalıdır.
              </p>

              {/* Presets */}
              <div className="mb-4">
                <label className="text-xs font-semibold text-gray-500 uppercase tracking-wider block mb-1.5">
                  Örnek Broadcasting Durumları
                </label>
                <div className="flex flex-wrap gap-2">
                  {BROADCAST_PRESETS.map((p) => (
                    <button
                      key={p.name}
                      onClick={() => {
                        setShapeAStr(p.a.join(', '));
                        setShapeBStr(p.b.join(', '));
                        runBroadcastAnalysis(p.a, p.b);
                      }}
                      className="px-2.5 py-1.5 text-xs bg-gray-50 hover:bg-indigo-50 hover:text-indigo-700 hover:border-indigo-200 border border-gray-200 rounded-lg text-gray-700 transition-colors text-left"
                    >
                      <span className="font-semibold">{p.name}</span>
                      <span className="text-gray-400 ml-1.5 font-mono">[{p.a.join(', ')}] vs [{p.b.join(', ')}]</span>
                    </button>
                  ))}
                </div>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div>
                  <label className="text-xs font-semibold text-gray-500 uppercase tracking-wider block mb-1.5">
                    Tensör A Şekli [d0, d1, ...]
                  </label>
                  <input
                    type="text"
                    value={shapeAStr}
                    onChange={(e) => setShapeAStr(e.target.value)}
                    placeholder="ör: 3, 1"
                    className="w-full px-3.5 py-2.5 rounded-xl border border-gray-200 font-mono text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
                  />
                </div>

                <div>
                  <label className="text-xs font-semibold text-gray-500 uppercase tracking-wider block mb-1.5">
                    Tensör B Şekli [d0, d1, ...]
                  </label>
                  <input
                    type="text"
                    value={shapeBStr}
                    onChange={(e) => setShapeBStr(e.target.value)}
                    placeholder="ör: 1, 4"
                    className="w-full px-3.5 py-2.5 rounded-xl border border-gray-200 font-mono text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
                  />
                </div>
              </div>

              <div className="mt-4 flex justify-end">
                <button
                  onClick={() => runBroadcastAnalysis()}
                  disabled={broadcastLoading}
                  className="px-5 py-2.5 bg-indigo-600 hover:bg-indigo-700 text-white rounded-xl text-sm font-semibold shadow-sm transition-all flex items-center space-x-2"
                >
                  {broadcastLoading ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
                  <span>Uyumluluğu Kontrol Et</span>
                </button>
              </div>
            </div>

            {/* Analysis Step-by-Step Visualization */}
            {broadcastAnalysis && (
              <div className="bg-white rounded-2xl border border-gray-200/80 p-6 shadow-sm space-y-4">
                <div className="flex items-center justify-between">
                  <h3 className="text-sm font-semibold text-gray-900">
                    Boyutsal Hizalama ve Genişleme Adımları
                  </h3>

                  {broadcastAnalysis.compatible ? (
                    <span className="px-3 py-1 bg-emerald-100 text-emerald-800 rounded-full text-xs font-semibold flex items-center space-x-1">
                      <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
                      <span>Uyumlu: Sonuç [{broadcastAnalysis.result_shape?.join(', ')}]</span>
                    </span>
                  ) : (
                    <span className="px-3 py-1 bg-red-100 text-red-800 rounded-full text-xs font-semibold flex items-center space-x-1">
                      <XCircle className="w-3.5 h-3.5 text-red-600" />
                      <span>Uyumsuz Broadcasting</span>
                    </span>
                  )}
                </div>

                <div className="overflow-x-auto">
                  <table className="w-full text-left text-xs sm:text-sm">
                    <thead>
                      <tr className="border-b border-gray-200 text-gray-500 font-mono text-xs">
                        <th className="py-2 px-3">Eksen (Axis)</th>
                        <th className="py-2 px-3">Tensör A</th>
                        <th className="py-2 px-3">Tensör B</th>
                        <th className="py-2 px-3">Sonuç Boyut</th>
                        <th className="py-2 px-3">Kural & Açıklama</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-100 font-mono">
                      {broadcastAnalysis.steps.map((step, idx) => (
                        <tr
                          key={idx}
                          className={`hover:bg-gray-50/60 ${
                            !step.compatible ? 'bg-red-50/70 text-red-900' : ''
                          }`}
                        >
                          <td className="py-2.5 px-3 text-gray-500">axis {step.axis}</td>
                          <td className="py-2.5 px-3 font-bold text-gray-800">{step.dim_a}</td>
                          <td className="py-2.5 px-3 font-bold text-gray-800">{step.dim_b}</td>
                          <td className="py-2.5 px-3 font-bold text-indigo-600">
                            {step.result_dim === -1 ? 'Hata' : step.result_dim}
                          </td>
                          <td className="py-2.5 px-3 font-sans text-xs text-gray-700">
                            {step.action}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                {broadcastAnalysis.error && (
                  <div className="p-3 bg-red-50 border border-red-200 rounded-xl text-xs text-red-700">
                    {broadcastAnalysis.error}
                  </div>
                )}
              </div>
            )}

            {/* 2D Matrix Visual Simulation */}
            <div className="bg-white rounded-2xl border border-gray-200/80 p-6 shadow-sm space-y-4">
              <h3 className="text-sm font-semibold text-gray-900 flex items-center space-x-2">
                <Table className="w-4 h-4 text-indigo-600" />
                <span>2D Matris Broadcasting Canlı Simülatörü</span>
              </h3>
              <p className="text-xs text-gray-500">
                Sütun vektörü ile satır vektörünü toplayarak veya çarparak broadcasting'in tensörleri nasıl kopyasız (stride 0 ile) genişlettiğini canlı izleyin.
              </p>

              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                <div>
                  <label className="text-xs font-semibold text-gray-500 block mb-1">
                    Matris A JSON (ör: [[1], [2], [3]])
                  </label>
                  <input
                    type="text"
                    value={matrixAInput}
                    onChange={(e) => setMatrixAInput(e.target.value)}
                    className="w-full px-3 py-2 border rounded-xl font-mono text-xs focus:ring-2 focus:ring-indigo-500"
                  />
                </div>

                <div>
                  <label className="text-xs font-semibold text-gray-500 block mb-1">
                    Matris B JSON (ör: [[10, 20, 30, 40]])
                  </label>
                  <input
                    type="text"
                    value={matrixBInput}
                    onChange={(e) => setMatrixBInput(e.target.value)}
                    className="w-full px-3 py-2 border rounded-xl font-mono text-xs focus:ring-2 focus:ring-indigo-500"
                  />
                </div>

                <div>
                  <label className="text-xs font-semibold text-gray-500 block mb-1">İşlem</label>
                  <div className="flex space-x-2">
                    <select
                      value={broadcastOp}
                      onChange={(e: any) => setBroadcastOp(e.target.value)}
                      className="flex-1 px-3 py-2 border rounded-xl text-xs font-medium bg-white"
                    >
                      <option value="add">Toplama (+)</option>
                      <option value="mul">Çarpma (×)</option>
                      <option value="sub">Çıkarma (-)</option>
                    </select>
                    <button
                      onClick={runBroadcastSimulation}
                      className="px-4 py-2 bg-indigo-600 hover:bg-indigo-700 text-white rounded-xl text-xs font-semibold transition-colors"
                    >
                      Çalıştır
                    </button>
                  </div>
                </div>
              </div>

              {broadcastSimError && (
                <div className="p-3 bg-red-50 text-red-700 border border-red-200 rounded-xl text-xs">
                  {broadcastSimError}
                </div>
              )}

              {broadcastSimResult && (
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-4">
                  {/* Broadcasted A */}
                  <div className="p-3 bg-gray-50 rounded-xl border border-gray-200">
                    <span className="text-xs font-semibold text-gray-700 block mb-2">
                      Genişletilmiş Matris A
                    </span>
                    <div className="space-y-1 font-mono text-xs">
                      {broadcastSimResult.broadcasted_a.map((row, rIdx) => (
                        <div key={rIdx} className="flex space-x-1 justify-center">
                          {row.map((val, cIdx) => (
                            <span key={cIdx} className="w-10 py-1 bg-white border rounded text-center font-bold text-indigo-700">
                              {val}
                            </span>
                          ))}
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Broadcasted B */}
                  <div className="p-3 bg-gray-50 rounded-xl border border-gray-200">
                    <span className="text-xs font-semibold text-gray-700 block mb-2">
                      Genişletilmiş Matris B
                    </span>
                    <div className="space-y-1 font-mono text-xs">
                      {broadcastSimResult.broadcasted_b.map((row, rIdx) => (
                        <div key={rIdx} className="flex space-x-1 justify-center">
                          {row.map((val, cIdx) => (
                            <span key={cIdx} className="w-10 py-1 bg-white border rounded text-center font-bold text-amber-700">
                              {val}
                            </span>
                          ))}
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Result Matrix */}
                  <div className="p-3 bg-indigo-50/70 rounded-xl border border-indigo-200">
                    <span className="text-xs font-semibold text-indigo-950 block mb-2">
                      Sonuç Matrisi (C = A {broadcastOp === 'add' ? '+' : broadcastOp === 'mul' ? '×' : '-'} B)
                    </span>
                    <div className="space-y-1 font-mono text-xs">
                      {broadcastSimResult.result_matrix.map((row, rIdx) => (
                        <div key={rIdx} className="flex space-x-1 justify-center">
                          {row.map((val, cIdx) => (
                            <span key={cIdx} className="w-10 py-1 bg-indigo-600 text-white font-bold rounded text-center">
                              {val}
                            </span>
                          ))}
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* ================================================================== */}
        {/* TAB 3: GEMM & MATMUL CELL BREAKDOWN                                */}
        {/* ================================================================== */}
        {activeTab === 'matmul' && (
          <div className="space-y-6">
            {/* Dimension & Preset Controls */}
            <div className="bg-white rounded-2xl p-6 border border-gray-200/80 shadow-sm">
              <h2 className="text-base font-semibold text-gray-900 mb-2 flex items-center space-x-2">
                <Table className="w-4 h-4 text-indigo-600" />
                <span>Genel Matris Çarpımı (GEMM: C = A × B)</span>
              </h2>
              <p className="text-xs text-gray-500 mb-4">
                Sonuç matrisindeki (C) herhangi bir hücreye tıklayarak, A matrisinin ilgili satırı ve B matrisinin ilgili sütunu arasındaki skaler çarpım (dot product) adımlarını izleyin.
              </p>

              <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 items-end">
                <div>
                  <label className="text-xs font-semibold text-gray-500 block mb-1">M (A Satır)</label>
                  <input
                    type="number"
                    min={1}
                    max={6}
                    value={dimM}
                    onChange={(e) => {
                      const v = parseInt(e.target.value) || 1;
                      setDimM(v);
                      fetchSampleMatrices(v, dimK, dimN);
                    }}
                    className="w-full px-3 py-2 border rounded-xl font-mono text-sm"
                  />
                </div>

                <div>
                  <label className="text-xs font-semibold text-gray-500 block mb-1">K (İç Boyut)</label>
                  <input
                    type="number"
                    min={1}
                    max={6}
                    value={dimK}
                    onChange={(e) => {
                      const v = parseInt(e.target.value) || 1;
                      setDimK(v);
                      fetchSampleMatrices(dimM, v, dimN);
                    }}
                    className="w-full px-3 py-2 border rounded-xl font-mono text-sm"
                  />
                </div>

                <div>
                  <label className="text-xs font-semibold text-gray-500 block mb-1">N (B Sütun)</label>
                  <input
                    type="number"
                    min={1}
                    max={6}
                    value={dimN}
                    onChange={(e) => {
                      const v = parseInt(e.target.value) || 1;
                      setDimN(v);
                      fetchSampleMatrices(dimM, dimK, v);
                    }}
                    className="w-full px-3 py-2 border rounded-xl font-mono text-sm"
                  />
                </div>

                <div>
                  <label className="text-xs font-semibold text-gray-500 block mb-1">Matris Şablonu</label>
                  <select
                    value={matmulPreset}
                    onChange={(e) => {
                      setMatmulPreset(e.target.value);
                      fetchSampleMatrices(dimM, dimK, dimN, e.target.value);
                    }}
                    className="w-full px-3 py-2 border rounded-xl text-sm bg-white font-medium"
                  >
                    <option value="simple">Basit Tam Sayılar</option>
                    <option value="sparse">Seyrek (Sparse)</option>
                    <option value="gaussian">Gaussian Float</option>
                  </select>
                </div>
              </div>
            </div>

            {matmulError && (
              <div className="p-4 bg-red-50 border border-red-200 rounded-2xl text-xs text-red-700">
                {matmulError}
              </div>
            )}

            {matmulResult && (
              <div className="space-y-6">
                {/* Hardware FLOPs & Arithmetic Intensity Stats */}
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                  <div className="p-4 bg-white rounded-2xl border border-gray-200/80 shadow-sm">
                    <span className="text-xs text-gray-500 font-medium">Toplam FLOP</span>
                    <p className="text-xl font-bold font-mono text-indigo-600 mt-1">
                      {matmulResult.hardware_metrics.total_flops} FLOPs
                    </p>
                    <span className="text-[11px] text-gray-400 mt-0.5 block">2 × M × N × K</span>
                  </div>

                  <div className="p-4 bg-white rounded-2xl border border-gray-200/80 shadow-sm">
                    <span className="text-xs text-gray-500 font-medium">Aktarılan Veri</span>
                    <p className="text-xl font-bold font-mono text-gray-900 mt-1">
                      {matmulResult.hardware_metrics.formatted_memory}
                    </p>
                    <span className="text-[11px] text-gray-400 mt-0.5 block">FP32 Okuma + Yazma</span>
                  </div>

                  <div className="p-4 bg-white rounded-2xl border border-gray-200/80 shadow-sm">
                    <span className="text-xs text-gray-500 font-medium">Aritmetik Yoğunluk</span>
                    <p className="text-xl font-bold font-mono text-emerald-600 mt-1">
                      {matmulResult.hardware_metrics.arithmetic_intensity_flops_per_byte}
                    </p>
                    <span className="text-[11px] text-gray-400 mt-0.5 block">FLOPs / Byte (Roofline)</span>
                  </div>

                  <div className="p-4 bg-white rounded-2xl border border-gray-200/80 shadow-sm">
                    <span className="text-xs text-gray-500 font-medium">Seçili Hücre</span>
                    <p className="text-xl font-bold font-mono text-indigo-600 mt-1">
                      C[{selectedCell.row}, {selectedCell.col}] = {matmulResult.cell_value}
                    </p>
                    <span className="text-[11px] text-gray-400 mt-0.5 block">Nokta Çarpımı Sonucu</span>
                  </div>
                </div>

                {/* Interactive Matrix Grids */}
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 items-start">
                  {/* Matrix A */}
                  <div className="bg-white rounded-2xl border border-gray-200/80 p-5 shadow-sm space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-semibold text-gray-700 uppercase tracking-wider">
                        Matris A ({matmulResult.shape_a[0]} × {matmulResult.shape_a[1]})
                      </span>
                      <span className="text-[11px] text-indigo-600 font-mono font-medium">
                        Satır {selectedCell.row} Vurgulandı
                      </span>
                    </div>

                    <div className="space-y-1.5">
                      {matmulResult.matrix_a.map((row, rIdx) => {
                        const isSelectedRow = rIdx === selectedCell.row;
                        return (
                          <div
                            key={rIdx}
                            className={`flex space-x-1.5 p-1 rounded-xl transition-all ${
                              isSelectedRow ? 'bg-indigo-50 border border-indigo-200 shadow-sm' : ''
                            }`}
                          >
                            {row.map((val, cIdx) => (
                              <div
                                key={cIdx}
                                className={`flex-1 py-2 text-center font-mono text-xs rounded-lg font-semibold transition-all ${
                                  isSelectedRow
                                    ? 'bg-indigo-600 text-white shadow-sm'
                                    : 'bg-gray-50 text-gray-700 border border-gray-200/60'
                                }`}
                              >
                                {val}
                              </div>
                            ))}
                          </div>
                        );
                      })}
                    </div>
                  </div>

                  {/* Matrix B */}
                  <div className="bg-white rounded-2xl border border-gray-200/80 p-5 shadow-sm space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-semibold text-gray-700 uppercase tracking-wider">
                        Matris B ({matmulResult.shape_b[0]} × {matmulResult.shape_b[1]})
                      </span>
                      <span className="text-[11px] text-amber-600 font-mono font-medium">
                        Sütun {selectedCell.col} Vurgulandı
                      </span>
                    </div>

                    <div className="space-y-1.5">
                      {matmulResult.matrix_b.map((row, rIdx) => (
                        <div key={rIdx} className="flex space-x-1.5 p-1">
                          {row.map((val, cIdx) => {
                            const isSelectedCol = cIdx === selectedCell.col;
                            return (
                              <div
                                key={cIdx}
                                className={`flex-1 py-2 text-center font-mono text-xs rounded-lg font-semibold transition-all ${
                                  isSelectedCol
                                    ? 'bg-amber-500 text-white shadow-sm border border-amber-600'
                                    : 'bg-gray-50 text-gray-700 border border-gray-200/60'
                                }`}
                              >
                                {val}
                              </div>
                            );
                          })}
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Result Matrix C (Interactive Clickable) */}
                  <div className="bg-white rounded-2xl border-2 border-indigo-200 p-5 shadow-sm space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-semibold text-indigo-900 uppercase tracking-wider">
                        Sonuç C = A × B ({matmulResult.shape_c[0]} × {matmulResult.shape_c[1]})
                      </span>
                      <span className="text-[11px] text-gray-400 font-sans">
                        Hücreye Tıkla 👆
                      </span>
                    </div>

                    <div className="space-y-1.5">
                      {matmulResult.matrix_c.map((row, rIdx) => (
                        <div key={rIdx} className="flex space-x-1.5 p-1">
                          {row.map((val, cIdx) => {
                            const isSelected = rIdx === selectedCell.row && cIdx === selectedCell.col;
                            return (
                              <button
                                key={cIdx}
                                onClick={() => handleCellClick(rIdx, cIdx)}
                                className={`flex-1 py-2 text-center font-mono text-xs rounded-lg font-bold transition-all transform hover:scale-105 ${
                                  isSelected
                                    ? 'bg-gradient-to-r from-indigo-600 to-violet-600 text-white shadow-md ring-2 ring-indigo-400'
                                    : 'bg-indigo-50/60 hover:bg-indigo-100 text-indigo-950 border border-indigo-100'
                                }`}
                              >
                                {val}
                              </button>
                            );
                          })}
                        </div>
                      ))}
                    </div>
                  </div>
                </div>

                {/* Detailed Cell Calculation Panel */}
                <div className="bg-white rounded-2xl border border-gray-200/80 p-6 shadow-sm space-y-4">
                  <h3 className="text-sm font-semibold text-gray-900 flex items-center space-x-2">
                    <Activity className="w-4 h-4 text-indigo-600" />
                    <span>Hücre Ayrışımı: C[{selectedCell.row}, {selectedCell.col}] = A[{selectedCell.row}, :] · B[:, {selectedCell.col}]</span>
                  </h3>

                  {/* Formula view */}
                  <div className="p-4 bg-gray-900 text-white rounded-xl font-mono text-xs sm:text-sm overflow-x-auto">
                    {matmulResult.formula_string}
                  </div>

                  {/* Step-by-Step Pairwise Multiplication Cards */}
                  <div className="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-6 gap-3">
                    {matmulResult.pairwise_terms.map((term) => (
                      <div key={term.k_index} className="p-3 bg-gray-50 border border-gray-200/80 rounded-xl text-center">
                        <span className="text-[10px] text-gray-400 uppercase font-mono block">k = {term.k_index}</span>
                        <div className="flex items-center justify-center space-x-1 font-mono text-xs mt-1">
                          <span className="font-bold text-indigo-700">{term.a_val}</span>
                          <span className="text-gray-400">×</span>
                          <span className="font-bold text-amber-700">{term.b_val}</span>
                        </div>
                        <span className="text-xs font-bold text-gray-900 mt-1 block font-mono">
                          = {term.product}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* ================================================================== */}
        {/* TAB 4: ACTIVATION & AUTOGRAD GRAPH                                 */}
        {/* ================================================================== */}
        {activeTab === 'activation' && (
          <div className="space-y-8">
            {/* Section 4A: Activation Function Curves & Derivatives */}
            <div className="bg-white rounded-2xl p-6 border border-gray-200/80 shadow-sm space-y-4">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                <div>
                  <h2 className="text-base font-semibold text-gray-900 flex items-center space-x-2">
                    <TrendingUp className="w-4 h-4 text-indigo-600" />
                    <span>Aktivasyon Fonksiyonları & Türev Eğrileri</span>
                  </h2>
                  <p className="text-xs text-gray-500">
                    GELU, SiLU (SwiGLU), ReLU, Sigmoid ve Tanh fonksiyonlarının çıktı ve gradyan eğrilerini karşılaştırın.
                  </p>
                </div>

                <div className="flex items-center space-x-2">
                  <select
                    value={selectedActivation}
                    onChange={(e) => {
                      setSelectedActivation(e.target.value);
                      fetchActivationCurve(e.target.value);
                    }}
                    className="px-3 py-2 border rounded-xl text-xs font-semibold bg-white text-gray-800 focus:ring-2 focus:ring-indigo-500"
                  >
                    <option value="gelu">GELU (Gaussian Error Linear Unit - GPT/BERT)</option>
                    <option value="silu">SiLU / Swish (SwiGLU - LLaMA/Mistral)</option>
                    <option value="relu">ReLU (Rectified Linear Unit)</option>
                    <option value="sigmoid">Sigmoid (Logistic Gate)</option>
                    <option value="tanh">Tanh (Hyperbolic Tangent)</option>
                    <option value="softmax">Softmax (Attention Probabilities)</option>
                  </select>
                </div>
              </div>

              {curveData && (
                <div className="space-y-4">
                  {/* Formula and LLM context */}
                  <div className="p-3 bg-indigo-50/70 border border-indigo-100 rounded-xl text-xs space-y-1">
                    <div className="flex items-center justify-between">
                      <span className="font-bold text-indigo-950 font-mono">{curveData.formula}</span>
                      <span className="text-[11px] text-indigo-700 bg-white px-2 py-0.5 rounded border border-indigo-200">
                        {selectedActivation.toUpperCase()}
                      </span>
                    </div>
                    <p className="text-indigo-900">{curveData.description}</p>
                  </div>

                  {/* SVG Line Chart for Curve & Derivative */}
                  <div className="bg-gray-900 p-4 rounded-xl text-white">
                    <div className="flex items-center justify-between text-xs mb-2">
                      <span className="text-gray-400 font-mono">f(x) vs f&apos;(x) Gradyan [-4.0, +4.0]</span>
                      <div className="flex space-x-4">
                        <span className="flex items-center space-x-1.5">
                          <span className="w-3 h-1 bg-indigo-400 inline-block rounded"></span>
                          <span>f(x) Aktivasyon</span>
                        </span>
                        <span className="flex items-center space-x-1.5">
                          <span className="w-3 h-1 bg-amber-400 inline-block rounded"></span>
                          <span>f&apos;(x) Türev / Gradyan</span>
                        </span>
                      </div>
                    </div>

                    <div className="relative h-48 w-full border-t border-b border-gray-800">
                      {/* SVG Canvas */}
                      <svg className="w-full h-full" viewBox="0 0 400 160" preserveAspectRatio="none">
                        {/* Zero Axes */}
                        <line x1="0" y1="80" x2="400" y2="80" stroke="#374151" strokeWidth="1" strokeDasharray="3 3" />
                        <line x1="200" y1="0" x2="200" y2="160" stroke="#374151" strokeWidth="1" strokeDasharray="3 3" />

                        {/* Curve f(x) */}
                        <path
                          d={curveData.points.reduce((acc, pt, idx) => {
                            const xCoord = (idx / (curveData.points.length - 1)) * 400;
                            // scale y: map [-3, 3] to [160, 0]
                            const yCoord = 80 - pt.y * 22;
                            return `${acc} ${idx === 0 ? 'M' : 'L'} ${xCoord} ${Math.max(5, Math.min(155, yCoord))}`;
                          }, '')}
                          fill="none"
                          stroke="#818cf8"
                          strokeWidth="2.5"
                        />

                        {/* Derivative f'(x) */}
                        <path
                          d={curveData.points.reduce((acc, pt, idx) => {
                            const xCoord = (idx / (curveData.points.length - 1)) * 400;
                            const yCoord = 80 - pt.derivative * 25;
                            return `${acc} ${idx === 0 ? 'M' : 'L'} ${xCoord} ${Math.max(5, Math.min(155, yCoord))}`;
                          }, '')}
                          fill="none"
                          stroke="#fbbf24"
                          strokeWidth="2"
                          strokeDasharray="4 2"
                        />
                      </svg>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Section 4B: Temperature-Scaled Softmax & Shannon Entropy */}
            <div className="bg-white rounded-2xl p-6 border border-gray-200/80 shadow-sm space-y-4">
              <h2 className="text-base font-semibold text-gray-900 flex items-center space-x-2">
                <Sliders className="w-4 h-4 text-indigo-600" />
                <span>Sıcaklık (Temperature - τ) Softmax & Shannon Entropisi</span>
              </h2>
              <p className="text-xs text-gray-500">
                LLM metin üretiminde sıcaklık katsayısı logits değerlerini ölçeklendirerek çıktının deterministik (greedy) veya yaratıcı (high-entropy) olmasını belirler.
              </p>

              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 items-end">
                <div className="sm:col-span-2">
                  <label className="text-xs font-semibold text-gray-500 block mb-1">
                    Giriş Logits (Virgülle Ayrılmış)
                  </label>
                  <input
                    type="text"
                    value={logitsInputStr}
                    onChange={(e) => setLogitsInputStr(e.target.value)}
                    className="w-full px-3 py-2 border rounded-xl font-mono text-sm"
                  />
                </div>

                <div>
                  <div className="flex justify-between items-center mb-1">
                    <label className="text-xs font-semibold text-gray-500">Sıcaklık (τ = {softmaxTemp.toFixed(2)})</label>
                  </div>
                  <input
                    type="range"
                    min={0.1}
                    max={3.0}
                    step={0.05}
                    value={softmaxTemp}
                    onChange={(e) => {
                      const v = parseFloat(e.target.value);
                      setSoftmaxTemp(v);
                      runSoftmaxSimulation(v);
                    }}
                    className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-indigo-600"
                  />
                </div>
              </div>

              {softmaxResult && (
                <div className="space-y-3">
                  {/* Entropy & Interpretation badge */}
                  <div className="flex flex-wrap items-center justify-between p-3 bg-gray-50 border border-gray-200 rounded-xl text-xs gap-2">
                    <div>
                      <span className="font-semibold text-gray-900">Shannon Entropisi: </span>
                      <span className="font-mono font-bold text-indigo-600">{softmaxResult.entropy_bits} Bits</span>
                      <span className="text-gray-400 ml-2">En Yüksek Olasılık: %{(softmaxResult.max_probability * 100).toFixed(1)}</span>
                    </div>
                    <span className="text-[11px] font-medium text-gray-600">
                      {softmaxResult.interpretation}
                    </span>
                  </div>

                  {/* Probability distribution bars */}
                  <div className="space-y-2">
                    {softmaxResult.elements.map((el) => (
                      <div key={el.index} className="space-y-1">
                        <div className="flex justify-between text-xs font-mono">
                          <span className="text-gray-700">
                            Token #{el.index} (Logit: {el.raw_logit} → Ölçekli: {el.scaled_logit})
                          </span>
                          <span className="font-bold text-gray-900">
                            %{el.percentage.toFixed(2)} ({el.probability.toFixed(4)})
                          </span>
                        </div>
                        <div className="h-2.5 w-full bg-gray-100 rounded-full overflow-hidden">
                          <div
                            className={`h-full rounded-full transition-all duration-300 ${
                              el.index === softmaxResult.argmax_index
                                ? 'bg-gradient-to-r from-indigo-500 to-indigo-600'
                                : 'bg-indigo-300'
                            }`}
                            style={{ width: `${Math.max(1, el.percentage)}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {/* Section 4C: 2-Layer MLP Autograd Computation Graph */}
            <div className="bg-white rounded-2xl p-6 border border-gray-200/80 shadow-sm space-y-4">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                <div>
                  <h2 className="text-base font-semibold text-gray-900 flex items-center space-x-2">
                    <Activity className="w-4 h-4 text-indigo-600" />
                    <span>2-Katmanlı MLP Autograd Geri Yayılım Grafı (Backward Pass)</span>
                  </h2>
                  <p className="text-xs text-gray-500">
                    PyTorch Autograd motorunun zincir kuralı (Chain Rule) ile her tensör ve ağırlık için gradyan hesaplamasını inceleyin.
                  </p>
                </div>

                <button
                  onClick={runAutogradSimulation}
                  disabled={autogradLoading}
                  className="px-4 py-2 bg-indigo-600 hover:bg-indigo-700 text-white rounded-xl text-xs font-semibold shadow-sm transition-all flex items-center space-x-1.5 self-start sm:self-auto"
                >
                  {autogradLoading ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <Play className="w-3.5 h-3.5" />}
                  <span>Geri Yayılımı Hesapla</span>
                </button>
              </div>

              <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                <div>
                  <label className="text-xs font-semibold text-gray-500 block mb-1">Giriş x</label>
                  <input
                    type="text"
                    value={autogradX}
                    onChange={(e) => setAutogradX(e.target.value)}
                    className="w-full px-3 py-1.5 border rounded-xl font-mono text-xs"
                  />
                </div>

                <div>
                  <label className="text-xs font-semibold text-gray-500 block mb-1">Hedef y</label>
                  <input
                    type="text"
                    value={autogradY}
                    onChange={(e) => setAutogradY(e.target.value)}
                    className="w-full px-3 py-1.5 border rounded-xl font-mono text-xs"
                  />
                </div>

                <div>
                  <label className="text-xs font-semibold text-gray-500 block mb-1">Gizli Boyut</label>
                  <input
                    type="number"
                    min={1}
                    max={6}
                    value={autogradHiddenDim}
                    onChange={(e) => setAutogradHiddenDim(parseInt(e.target.value) || 1)}
                    className="w-full px-3 py-1.5 border rounded-xl font-mono text-xs"
                  />
                </div>

                <div>
                  <label className="text-xs font-semibold text-gray-500 block mb-1">Aktivasyon</label>
                  <select
                    value={autogradAct}
                    onChange={(e) => setAutogradAct(e.target.value)}
                    className="w-full px-3 py-1.5 border rounded-xl text-xs bg-white"
                  >
                    <option value="relu">ReLU</option>
                    <option value="gelu">GELU</option>
                    <option value="sigmoid">Sigmoid</option>
                    <option value="tanh">Tanh</option>
                  </select>
                </div>
              </div>

              {autogradResult && (
                <div className="space-y-4">
                  {/* Summary bar */}
                  <div className="flex flex-wrap items-center justify-between p-3 bg-indigo-50 border border-indigo-100 rounded-xl text-xs gap-2">
                    <span className="font-semibold text-indigo-950">{autogradResult.architecture}</span>
                    <span className="font-mono text-indigo-900">
                      MSE Loss: <strong className="text-indigo-600">{autogradResult.loss}</strong> | Tahmin: [{autogradResult.y_pred.join(', ')}]
                    </span>
                  </div>

                  {/* Computation Nodes Cards */}
                  <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-3">
                    {autogradResult.nodes.map((node) => {
                      const isWeight = node.type === 'parameter';
                      const isLoss = node.type === 'loss';
                      return (
                        <div
                          key={node.id}
                          className={`p-3.5 rounded-xl border transition-all ${
                            isLoss
                              ? 'bg-red-50/70 border-red-200'
                              : isWeight
                              ? 'bg-amber-50/70 border-amber-200'
                              : 'bg-gray-50 border-gray-200'
                          }`}
                        >
                          <div className="flex items-center justify-between mb-1.5">
                            <span className="font-semibold text-xs text-gray-900 truncate">
                              {node.label}
                            </span>
                            <span className="text-[10px] uppercase font-mono px-1.5 py-0.5 bg-white border rounded text-gray-500">
                              {node.type}
                            </span>
                          </div>

                          <div className="space-y-1 text-[11px] font-mono">
                            <div>
                              <span className="text-gray-400">Şekil: </span>
                              <span className="text-gray-700">[{node.shape.join(', ')}]</span>
                            </div>

                            <div className="truncate">
                              <span className="text-gray-400">İleri Değer: </span>
                              <span className="text-indigo-700 font-semibold">
                                {JSON.stringify(node.forward_val)}
                              </span>
                            </div>

                            <div className="truncate">
                              <span className="text-gray-400">Gradyan (∂L/∂): </span>
                              <span className="text-amber-700 font-bold">
                                {node.grad_val !== null ? JSON.stringify(node.grad_val) : 'N/A'}
                              </span>
                            </div>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
