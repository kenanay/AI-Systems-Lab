'use client';

import React, { useState, useEffect, useRef, useCallback } from 'react';
import Link from 'next/link';
import {
  Calculator,
  ArrowRight,
  Play,
  Pause,
  RotateCcw,
  CheckCircle2,
  AlertCircle,
  Info,
  Sparkles,
  Sliders,
  ChevronRight,
  TrendingDown,
  Activity,
  Layers,
  Compass,
  Zap,
  Cpu,
  BarChart2,
  RefreshCw,
} from 'lucide-react';
import {
  api,
  MatrixOperationResult,
  VectorOperationResult,
  DerivativeResult,
  GradientDescentResult,
  ChainRuleResult,
  EigenvaluesResult,
} from '@/lib/api';

// ============================================================================
// Types & Presets
// ============================================================================

type ActiveTab = 'linalg' | 'calculus' | 'chainrule';
type MatrixOp = 'multiply' | 'transpose' | 'inverse' | 'determinant' | 'eigenvalues';

const MATRIX_DIM_PRESETS = [
  { label: '2 × 2', rowsA: 2, colsA: 2, rowsB: 2, colsB: 2 },
  { label: '2 × 3 & 3 × 2', rowsA: 2, colsA: 3, rowsB: 3, colsB: 2 },
  { label: '3 × 3', rowsA: 3, colsA: 3, rowsB: 3, colsB: 3 },
  { label: '1 × 3 & 3 × 1 (İç Çarpım)', rowsA: 1, colsA: 3, rowsB: 3, colsB: 1 },
];

const MATRIX_PRESETS_2X2 = [
  {
    name: 'Birim Matris (Identity)',
    desc: 'Çarpımda etkisiz eleman',
    a: [[1, 0], [0, 1]],
    b: [[3, 4], [5, 6]],
  },
  {
    name: 'Rotasyon (45°)',
    desc: 'Vektör uzayını döndürür',
    a: [[0.707, -0.707], [0.707, 0.707]],
    b: [[2, 0], [0, 2]],
  },
  {
    name: 'Simetrik (Covariance)',
    desc: 'A = A^T özelliği',
    a: [[2, 3], [3, 5]],
    b: [[1, 2], [3, 4]],
  },
  {
    name: 'Yapay Sinir Katmanı',
    desc: 'X @ W projeksiyonu',
    a: [[0.5, -0.2], [0.8, 0.3]],
    b: [[1.2, 0.4], [-0.5, 0.9]],
  },
];

const VECTOR_PRESETS = [
  { name: 'Dik Vektörler (90°)', a: [1, 0], b: [0, 1], desc: 'a · b = 0, Cosine = 0' },
  { name: 'Paralel (0°)', a: [2, 0], b: [4, 0], desc: 'a · b = max, Cosine = 1.0' },
  { name: '45° Açı', a: [2, 0], b: [2, 2], desc: 'Cosine = 0.707' },
  { name: 'Zıt Yönlü (180°)', a: [3, 0], b: [-3, 0], desc: 'Cosine = -1.0' },
];

const FUNCTION_PRESETS = [
  { id: 'quadratic', name: 'Kuadratik: f(x) = x²', formula: 'x²', desc: 'Tipik kase tipi MSE kayıp fonksiyonu tabanı' },
  { id: 'cubic', name: 'Kübik: f(x) = x³ - 3x', formula: 'x³ - 3x', desc: 'Yerel minimum ve maksimum (non-convex yüzey)' },
  { id: 'sin', name: 'Trigonometrik: f(x) = sin(x)', formula: 'sin(x)', desc: 'Periyodik dalga fonksiyonu' },
  { id: 'exp', name: 'Üstel: f(x) = e^x', formula: 'e^x', desc: 'Üstel patlama / gradyan artışı' },
  { id: 'sigmoid', name: 'Sigmoid: f(x) = σ(x)', formula: '1 / (1 + e^(-x))', desc: 'S-şekilli lojistik aktivasyon' },
];

// ============================================================================
// Main MathLabPage Component
// ============================================================================

export default function MathLabPage() {
  const [activeTab, setActiveTab] = useState<ActiveTab>('linalg');

  // --------------------------------------------------------------------------
  // Tab 1: Linear Algebra State
  // --------------------------------------------------------------------------
  const [dimConfig, setDimConfig] = useState(MATRIX_DIM_PRESETS[0]);
  const [matrixA, setMatrixA] = useState<number[][]>([[1, 2], [3, 4]]);
  const [matrixB, setMatrixB] = useState<number[][]>([[5, 6], [7, 8]]);
  const [matrixOp, setMatrixOp] = useState<MatrixOp>('multiply');
  const [selectedCell, setSelectedCell] = useState<[number, number]>([0, 0]);

  // Operations Results
  const [matmulResult, setMatmulResult] = useState<MatrixOperationResult | null>(null);
  const [transposeResult, setTransposeResult] = useState<MatrixOperationResult | null>(null);
  const [inverseResult, setInverseResult] = useState<MatrixOperationResult | null>(null);
  const [determinantResult, setDeterminantResult] = useState<{ success: boolean; determinant?: number; is_singular?: boolean; error?: string } | null>(null);
  const [eigenResult, setEigenResult] = useState<EigenvaluesResult | null>(null);
  const [linalgLoading, setLinalgLoading] = useState(false);
  const [linalgError, setLinalgError] = useState<string | null>(null);

  // Vector Dot Product State
  const [vectorDim, setVectorDim] = useState<number>(2);
  const [vectorA, setVectorA] = useState<number[]>([3, 4]);
  const [vectorB, setVectorB] = useState<number[]>([4, 3]);
  const [vectorResult, setVectorResult] = useState<VectorOperationResult | null>(null);
  const [vectorLoading, setVectorLoading] = useState(false);

  // --------------------------------------------------------------------------
  // Tab 2: Calculus & Gradient Descent State
  // --------------------------------------------------------------------------
  const [selectedFunc, setSelectedFunc] = useState<string>('quadratic');
  const [derivData, setDerivData] = useState<DerivativeResult | null>(null);
  const [derivLoading, setDerivLoading] = useState(false);

  // Gradient Descent
  const [initialX, setInitialX] = useState<number>(2.5);
  const [learningRate, setLearningRate] = useState<number>(0.15);
  const [maxIterations, setMaxIterations] = useState<number>(25);
  const [gdResult, setGdResult] = useState<GradientDescentResult | null>(null);
  const [gdLoading, setGdLoading] = useState(false);
  const [scrubberStep, setScrubberStep] = useState<number>(0);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const timerRef = useRef<NodeJS.Timeout | null>(null);

  // --------------------------------------------------------------------------
  // Tab 3: Chain Rule State
  // --------------------------------------------------------------------------
  const [outerFn, setOuterFn] = useState<string>('square');
  const [innerFn, setInnerFn] = useState<string>('linear');
  const [chainX, setChainX] = useState<number>(1.5);
  const [chainResult, setChainResult] = useState<ChainRuleResult | null>(null);
  const [chainLoading, setChainLoading] = useState(false);

  // --------------------------------------------------------------------------
  // Initial Computations
  // --------------------------------------------------------------------------
  useEffect(() => {
    runMatrixMultiply();
    runVectorDot();
    fetchDerivative('quadratic');
    runGradientDescent('quadratic', 2.5, 0.15, 25);
    runChainRule('square', 'linear', 1.5);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // --------------------------------------------------------------------------
  // Linear Algebra Handlers
  // --------------------------------------------------------------------------
  const updateMatrixDimensions = (preset: typeof MATRIX_DIM_PRESETS[0]) => {
    setDimConfig(preset);
    const newA: number[][] = Array.from({ length: preset.rowsA }, (_, r) =>
      Array.from({ length: preset.colsA }, (_, c) => (r === c ? 1 : 0))
    );
    const newB: number[][] = Array.from({ length: preset.rowsB }, (_, r) =>
      Array.from({ length: preset.colsB }, (_, c) => (r === c ? 1 : 0))
    );
    setMatrixA(newA);
    setMatrixB(newB);
    setSelectedCell([0, 0]);
  };

  const handleCellChangeA = (r: number, c: number, val: number) => {
    const updated = matrixA.map((row, i) =>
      row.map((cell, j) => (i === r && j === c ? val : cell))
    );
    setMatrixA(updated);
  };

  const handleCellChangeB = (r: number, c: number, val: number) => {
    const updated = matrixB.map((row, i) =>
      row.map((cell, j) => (i === r && j === c ? val : cell))
    );
    setMatrixB(updated);
  };

  const runMatrixMultiply = async (cellToInspect?: [number, number]) => {
    setLinalgLoading(true);
    setLinalgError(null);
    const cell = cellToInspect || selectedCell;
    try {
      const res = await api.mathLab.multiplyMatrices({
        matrix_a: matrixA,
        matrix_b: matrixB,
        detailed_cell: cell,
      });
      setMatmulResult(res);
      setMatrixOp('multiply');
    } catch (err: unknown) {
      const errorMsg = err instanceof Error ? err.message : 'Matris çarpımı başarısız oldu.';
      setLinalgError(errorMsg);
    } finally {
      setLinalgLoading(false);
    }
  };

  const runTranspose = async () => {
    setLinalgLoading(true);
    setLinalgError(null);
    try {
      const res = await api.mathLab.transposeMatrix({ matrix: matrixA });
      setTransposeResult(res);
      setMatrixOp('transpose');
    } catch (err: unknown) {
      setLinalgError(err instanceof Error ? err.message : 'Transpoz hatası');
    } finally {
      setLinalgLoading(false);
    }
  };

  const runInverse = async () => {
    setLinalgLoading(true);
    setLinalgError(null);
    try {
      const res = await api.mathLab.inverseMatrix({ matrix: matrixA });
      setInverseResult(res);
      setMatrixOp('inverse');
    } catch (err: unknown) {
      setLinalgError(err instanceof Error ? err.message : 'Ters matris hesabı başarısız');
    } finally {
      setLinalgLoading(false);
    }
  };

  const runDeterminant = async () => {
    setLinalgLoading(true);
    setLinalgError(null);
    try {
      const res = await api.mathLab.calculateDeterminant({ matrix: matrixA });
      setDeterminantResult(res);
      setMatrixOp('determinant');
    } catch (err: unknown) {
      setLinalgError(err instanceof Error ? err.message : 'Determinant hesabı başarısız');
    } finally {
      setLinalgLoading(false);
    }
  };

  const runEigenvalues = async () => {
    setLinalgLoading(true);
    setLinalgError(null);
    try {
      const res = await api.mathLab.calculateEigenvalues({ matrix: matrixA });
      setEigenResult(res);
      setMatrixOp('eigenvalues');
    } catch (err: unknown) {
      setLinalgError(err instanceof Error ? err.message : 'Öz değer hesabı başarısız');
    } finally {
      setLinalgLoading(false);
    }
  };

  const runVectorDot = async (customA?: number[], customB?: number[]) => {
    setVectorLoading(true);
    try {
      const a = customA || vectorA;
      const b = customB || vectorB;
      const res = await api.mathLab.vectorDotProduct({
        vector_a: a,
        vector_b: b,
      });
      setVectorResult(res);
    } catch (err: unknown) {
      console.error('Vector dot product error:', err);
    } finally {
      setVectorLoading(false);
    }
  };

  const handleVectorDimChange = (dim: number) => {
    setVectorDim(dim);
    const newA = Array.from({ length: dim }, (_, i) => (i < vectorA.length ? vectorA[i] : 1));
    const newB = Array.from({ length: dim }, (_, i) => (i < vectorB.length ? vectorB[i] : 1));
    setVectorA(newA);
    setVectorB(newB);
    runVectorDot(newA, newB);
  };

  // --------------------------------------------------------------------------
  // Calculus Handlers
  // --------------------------------------------------------------------------
  const fetchDerivative = async (fnName: string) => {
    setDerivLoading(true);
    try {
      const res = await api.mathLab.computeDerivative({
        function_name: fnName,
        x_min: -3.5,
        x_max: 3.5,
        num_points: 71,
      });
      setDerivData(res);
    } catch (err: unknown) {
      console.error('Compute derivative error:', err);
    } finally {
      setDerivLoading(false);
    }
  };

  const runGradientDescent = async (
    fnName = selectedFunc,
    x0 = initialX,
    lr = learningRate,
    iters = maxIterations
  ) => {
    setGdLoading(true);
    setIsPlaying(false);
    if (timerRef.current) clearInterval(timerRef.current);
    try {
      const res = await api.mathLab.simulateGradientDescent({
        function_name: fnName,
        initial_x: x0,
        learning_rate: lr,
        max_iterations: iters,
      });
      setGdResult(res);
      setScrubberStep(res.history.length > 0 ? res.history.length - 1 : 0);
    } catch (err: unknown) {
      console.error('Gradient descent error:', err);
    } finally {
      setGdLoading(false);
    }
  };

  // Scrubber animation player
  const togglePlay = useCallback(() => {
    if (isPlaying) {
      setIsPlaying(false);
      if (timerRef.current) clearInterval(timerRef.current);
    } else {
      if (!gdResult || gdResult.history.length === 0) return;
      setIsPlaying(true);
      if (scrubberStep >= gdResult.history.length - 1) {
        setScrubberStep(0);
      }
      timerRef.current = setInterval(() => {
        setScrubberStep((prev) => {
          if (!gdResult || prev >= gdResult.history.length - 1) {
            setIsPlaying(false);
            if (timerRef.current) clearInterval(timerRef.current);
            return prev;
          }
          return prev + 1;
        });
      }, 350);
    }
  }, [isPlaying, gdResult, scrubberStep]);

  useEffect(() => {
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, []);

  // --------------------------------------------------------------------------
  // Chain Rule Handlers
  // --------------------------------------------------------------------------
  const runChainRule = async (outFn = outerFn, inFn = innerFn, xVal = chainX) => {
    setChainLoading(true);
    try {
      const res = await api.mathLab.demonstrateChainRule({
        outer_function: outFn,
        inner_function: inFn,
        x_value: xVal,
      });
      setChainResult(res);
    } catch (err: unknown) {
      console.error('Chain rule error:', err);
    } finally {
      setChainLoading(false);
    }
  };

  // --------------------------------------------------------------------------
  // Calculus Graph Coordinate Converter (SVG)
  // --------------------------------------------------------------------------
  const SVG_WIDTH = 580;
  const SVG_HEIGHT = 280;
  const X_RANGE = [-3.5, 3.5];
  const Y_RANGE = [-4.0, 10.0];

  const toSvgX = (x: number) => {
    return ((x - X_RANGE[0]) / (X_RANGE[1] - X_RANGE[0])) * SVG_WIDTH;
  };

  const toSvgY = (y: number) => {
    const clampedY = Math.max(Y_RANGE[0], Math.min(Y_RANGE[1], y));
    return SVG_HEIGHT - ((clampedY - Y_RANGE[0]) / (Y_RANGE[1] - Y_RANGE[0])) * SVG_HEIGHT;
  };

  // Build SVG path for f(x) and f'(x)
  const fPath = derivData?.x_values?.length
    ? derivData.x_values
        .map((x, i) => `${i === 0 ? 'M' : 'L'} ${toSvgX(x).toFixed(1)} ${toSvgY(derivData.y_values[i]).toFixed(1)}`)
        .join(' ')
    : '';

  const fPrimePath = derivData?.x_values?.length
    ? derivData.x_values
        .map((x, i) => `${i === 0 ? 'M' : 'L'} ${toSvgX(x).toFixed(1)} ${toSvgY(derivData.dy_dx_values[i]).toFixed(1)}`)
        .join(' ')
    : '';

  // Current gradient descent point
  const currentGdStep = gdResult?.history?.[scrubberStep] || null;

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 py-8 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto space-y-8">
        {/* ================================================================== */}
        {/* Header & Breadcrumbs                                               */}
        {/* ================================================================== */}
        <div className="bg-white rounded-3xl border border-gray-200/80 p-6 sm:p-8 shadow-sm relative overflow-hidden">
          <div className="absolute -right-16 -top-16 w-64 h-64 bg-indigo-50 rounded-full blur-3xl pointer-events-none" />
          <div className="absolute right-32 -bottom-20 w-56 h-56 bg-emerald-50 rounded-full blur-3xl pointer-events-none" />

          <div className="relative z-10 flex flex-col md:flex-row md:items-center justify-between gap-6">
            <div>
              <div className="flex items-center space-x-2 text-xs font-semibold text-indigo-600 uppercase tracking-wider mb-2">
                <Link href="/" className="hover:underline">Ana Sayfa</Link>
                <ChevronRight className="w-3.5 h-3.5 text-gray-400" />
                <span className="text-gray-500">Modül 4.1</span>
                <ChevronRight className="w-3.5 h-3.5 text-gray-400" />
                <span className="text-indigo-600 font-bold">Math Lab</span>
              </div>
              <h1 className="text-2xl sm:text-3xl lg:text-4xl font-extrabold text-gray-900 tracking-tight flex items-center space-x-3">
                <span className="text-3xl sm:text-4xl">📐</span>
                <span>Math Lab: Lineer Cebir & Kalkülüs</span>
              </h1>
              <p className="mt-2 text-sm sm:text-base text-gray-600 max-w-3xl leading-relaxed">
                Yapay zeka, derin öğrenme ve büyük dil modellerinin temel yapı taşlarını interaktif olarak keşfedin:
                matris çarpımı, öz değerler, kosinüs benzerliği, sayısal türevler, gradiyen inişi ve geriye yayılımın (backpropagation) dayandığı zincir kuralı.
              </p>
            </div>

            {/* Quick Stat Badges */}
            <div className="grid grid-cols-2 gap-3 sm:flex sm:items-center sm:space-x-3 text-left">
              <div className="p-3 bg-indigo-50/80 rounded-2xl border border-indigo-100">
                <div className="text-[11px] font-medium text-indigo-700">Lineer Cebir</div>
                <div className="text-sm font-bold text-indigo-950 flex items-center space-x-1">
                  <Calculator className="w-3.5 h-3.5 text-indigo-600" />
                  <span>Matris & Vektör</span>
                </div>
              </div>
              <div className="p-3 bg-emerald-50/80 rounded-2xl border border-emerald-100">
                <div className="text-[11px] font-medium text-emerald-700">Kalkülüs Motoru</div>
                <div className="text-sm font-bold text-emerald-950 flex items-center space-x-1">
                  <TrendingDown className="w-3.5 h-3.5 text-emerald-600" />
                  <span>Gradient Descent</span>
                </div>
              </div>
            </div>
          </div>

          {/* Navigation Tabs */}
          <div className="flex border-b border-gray-100 mt-8 space-x-2 sm:space-x-4 overflow-x-auto">
            <button
              onClick={() => setActiveTab('linalg')}
              className={`pb-3 px-4 text-sm font-semibold border-b-2 flex items-center space-x-2 whitespace-nowrap transition-all ${
                activeTab === 'linalg'
                  ? 'border-indigo-600 text-indigo-600'
                  : 'border-transparent text-gray-500 hover:text-gray-800'
              }`}
            >
              <Calculator className="w-4 h-4" />
              <span>1. Lineer Cebir (Matris & Vektör)</span>
            </button>
            <button
              onClick={() => setActiveTab('calculus')}
              className={`pb-3 px-4 text-sm font-semibold border-b-2 flex items-center space-x-2 whitespace-nowrap transition-all ${
                activeTab === 'calculus'
                  ? 'border-indigo-600 text-indigo-600'
                  : 'border-transparent text-gray-500 hover:text-gray-800'
              }`}
            >
              <TrendingDown className="w-4 h-4" />
              <span>2. Kalkülüs & Gradiyen İnişi</span>
            </button>
            <button
              onClick={() => setActiveTab('chainrule')}
              className={`pb-3 px-4 text-sm font-semibold border-b-2 flex items-center space-x-2 whitespace-nowrap transition-all ${
                activeTab === 'chainrule'
                  ? 'border-indigo-600 text-indigo-600'
                  : 'border-transparent text-gray-500 hover:text-gray-800'
              }`}
            >
              <Activity className="w-4 h-4" />
              <span>3. Zincir Kuralı & Backprop</span>
            </button>
          </div>
        </div>

        {/* ================================================================== */}
        {/* TAB 1: LINEAR ALGEBRA                                              */}
        {/* ================================================================== */}
        {activeTab === 'linalg' && (
          <div className="space-y-8 animate-fadeIn">
            {/* Matrix Section Card */}
            <div className="bg-white rounded-3xl border border-gray-200/80 p-6 sm:p-8 shadow-sm">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-6 border-b border-gray-100 gap-4">
                <div>
                  <h2 className="text-xl font-bold text-gray-900 flex items-center space-x-2">
                    <span className="p-1.5 bg-indigo-100 rounded-lg text-indigo-600">
                      <Calculator className="w-5 h-5" />
                    </span>
                    <span>İnteraktif Matris İşlemleri (A × B, A^T, A⁻¹, det, λ)</span>
                  </h2>
                  <p className="text-xs sm:text-sm text-gray-500 mt-1">
                    Hücreleri doğrudan düzenleyin, hazır matris şablonlarını yükleyin ve sonuç matrisindeki herhangi bir hücreye tıklayarak satır-sütun nokta çarpımı açılımını görün.
                  </p>
                </div>

                {/* Dimension Presets */}
                <div className="flex items-center space-x-2 flex-wrap gap-y-2">
                  <span className="text-xs text-gray-400 font-medium">Boyut:</span>
                  {MATRIX_DIM_PRESETS.map((preset, idx) => (
                    <button
                      key={idx}
                      onClick={() => updateMatrixDimensions(preset)}
                      className={`text-xs px-3 py-1.5 rounded-xl font-medium transition-all ${
                        dimConfig.label === preset.label
                          ? 'bg-indigo-600 text-white shadow-sm'
                          : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                      }`}
                    >
                      {preset.label}
                    </button>
                  ))}
                </div>
              </div>

              {/* Quick Preset Buttons for 2x2 */}
              {dimConfig.rowsA === 2 && dimConfig.colsA === 2 && (
                <div className="mt-4 flex items-center space-x-2 overflow-x-auto pb-2">
                  <span className="text-xs text-gray-400 font-medium flex items-center space-x-1">
                    <Sparkles className="w-3.5 h-3.5 text-amber-500" />
                    <span>Şablonlar:</span>
                  </span>
                  {MATRIX_PRESETS_2X2.map((p, idx) => (
                    <button
                      key={idx}
                      onClick={() => {
                        setMatrixA(p.a);
                        setMatrixB(p.b);
                      }}
                      className="text-xs px-3 py-1 bg-indigo-50 text-indigo-700 border border-indigo-100 rounded-lg hover:bg-indigo-100 font-medium whitespace-nowrap transition-colors"
                      title={p.desc}
                    >
                      {p.name}
                    </button>
                  ))}
                </div>
              )}

              {/* Matrices Input Grids & Operator */}
              <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-center mt-6">
                {/* Matrix A */}
                <div className="lg:col-span-5 bg-slate-50/80 p-5 rounded-2xl border border-gray-200/80">
                  <div className="flex items-center justify-between mb-3">
                    <span className="text-xs font-bold text-gray-700 tracking-wide flex items-center space-x-1.5">
                      <span className="w-2.5 h-2.5 rounded-full bg-indigo-600" />
                      <span>Matris A ({matrixA.length} × {matrixA[0]?.length})</span>
                    </span>
                    <span className="text-[11px] text-gray-400 font-mono">Row [i]</span>
                  </div>

                  <div
                    className="grid gap-2"
                    style={{ gridTemplateColumns: `repeat(${matrixA[0]?.length || 2}, minmax(0, 1fr))` }}
                  >
                    {matrixA.map((row, r) =>
                      row.map((val, c) => {
                        const isHighlightedRow = matrixOp === 'multiply' && selectedCell[0] === r;
                        return (
                          <input
                            key={`a-${r}-${c}`}
                            type="number"
                            step="any"
                            value={val}
                            onChange={(e) => handleCellChangeA(r, c, parseFloat(e.target.value) || 0)}
                            className={`w-full text-center py-2.5 px-2 rounded-xl text-sm font-mono font-bold transition-all border ${
                              isHighlightedRow
                                ? 'bg-indigo-100 border-indigo-500 text-indigo-900 shadow-sm ring-2 ring-indigo-400/30'
                                : 'bg-white border-gray-200 text-gray-800 hover:border-gray-300 focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500'
                            }`}
                          />
                        );
                      })
                    )}
                  </div>
                </div>

                {/* Operator Symbol / Controls */}
                <div className="lg:col-span-2 flex flex-col items-center justify-center space-y-3">
                  <span className="text-2xl font-black text-gray-300">×</span>
                  <div className="flex flex-col space-y-2 w-full max-w-[140px]">
                    <button
                      onClick={() => runMatrixMultiply()}
                      disabled={linalgLoading}
                      className="w-full text-xs font-bold py-2 px-3 bg-indigo-600 hover:bg-indigo-700 text-white rounded-xl shadow-sm transition-all flex items-center justify-center space-x-1.5"
                    >
                      {linalgLoading ? (
                        <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                      ) : (
                        <Play className="w-3.5 h-3.5" />
                      )}
                      <span>A × B Çarp</span>
                    </button>
                  </div>
                </div>

                {/* Matrix B */}
                <div className="lg:col-span-5 bg-slate-50/80 p-5 rounded-2xl border border-gray-200/80">
                  <div className="flex items-center justify-between mb-3">
                    <span className="text-xs font-bold text-gray-700 tracking-wide flex items-center space-x-1.5">
                      <span className="w-2.5 h-2.5 rounded-full bg-emerald-600" />
                      <span>Matris B ({matrixB.length} × {matrixB[0]?.length})</span>
                    </span>
                    <span className="text-[11px] text-gray-400 font-mono">Col [j]</span>
                  </div>

                  <div
                    className="grid gap-2"
                    style={{ gridTemplateColumns: `repeat(${matrixB[0]?.length || 2}, minmax(0, 1fr))` }}
                  >
                    {matrixB.map((row, r) =>
                      row.map((val, c) => {
                        const isHighlightedCol = matrixOp === 'multiply' && selectedCell[1] === c;
                        return (
                          <input
                            key={`b-${r}-${c}`}
                            type="number"
                            step="any"
                            value={val}
                            onChange={(e) => handleCellChangeB(r, c, parseFloat(e.target.value) || 0)}
                            className={`w-full text-center py-2.5 px-2 rounded-xl text-sm font-mono font-bold transition-all border ${
                              isHighlightedCol
                                ? 'bg-emerald-100 border-emerald-500 text-emerald-900 shadow-sm ring-2 ring-emerald-400/30'
                                : 'bg-white border-gray-200 text-gray-800 hover:border-gray-300 focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500'
                            }`}
                          />
                        );
                      })
                    )}
                  </div>
                </div>
              </div>

              {/* Action Toolbar for Single-Matrix Operations */}
              <div className="mt-6 pt-5 border-t border-gray-100 flex flex-wrap items-center justify-between gap-3">
                <span className="text-xs font-bold text-gray-500">Matris A Analiz İşlemleri:</span>
                <div className="flex items-center space-x-2 flex-wrap gap-y-2">
                  <button
                    onClick={() => runTranspose()}
                    className={`text-xs px-3 py-1.5 rounded-xl font-medium transition-all ${
                      matrixOp === 'transpose'
                        ? 'bg-purple-600 text-white shadow-sm'
                        : 'bg-purple-50 text-purple-700 hover:bg-purple-100 border border-purple-200'
                    }`}
                  >
                    Transpoz (A^T)
                  </button>
                  <button
                    onClick={() => runInverse()}
                    className={`text-xs px-3 py-1.5 rounded-xl font-medium transition-all ${
                      matrixOp === 'inverse'
                        ? 'bg-rose-600 text-white shadow-sm'
                        : 'bg-rose-50 text-rose-700 hover:bg-rose-100 border border-rose-200'
                    }`}
                  >
                    Tersi (A⁻¹)
                  </button>
                  <button
                    onClick={() => runDeterminant()}
                    className={`text-xs px-3 py-1.5 rounded-xl font-medium transition-all ${
                      matrixOp === 'determinant'
                        ? 'bg-amber-600 text-white shadow-sm'
                        : 'bg-amber-50 text-amber-700 hover:bg-amber-100 border border-amber-200'
                    }`}
                  >
                    Determinant (|A|)
                  </button>
                  <button
                    onClick={() => runEigenvalues()}
                    className={`text-xs px-3 py-1.5 rounded-xl font-medium transition-all ${
                      matrixOp === 'eigenvalues'
                        ? 'bg-teal-600 text-white shadow-sm'
                        : 'bg-teal-50 text-teal-700 hover:bg-teal-100 border border-teal-200'
                    }`}
                  >
                    Öz Değerler (Eigen)
                  </button>
                </div>
              </div>

              {/* Error Alert */}
              {linalgError && (
                <div className="mt-4 p-4 bg-rose-50 border border-rose-200 rounded-2xl flex items-center space-x-3 text-rose-800 text-sm">
                  <AlertCircle className="w-5 h-5 flex-shrink-0 text-rose-600" />
                  <span>{linalgError}</span>
                </div>
              )}

              {/* ============================================================ */}
              {/* Output Display Area                                          */}
              {/* ============================================================ */}
              <div className="mt-6 pt-6 border-t border-gray-100">
                {/* 1. MatMul Result */}
                {matrixOp === 'multiply' && matmulResult?.success && (
                  <div className="space-y-6">
                    <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                      <div>
                        <span className="text-xs font-bold text-gray-500 uppercase tracking-wider">
                          Sonuç Matrisi C = A × B ({matmulResult.result?.length} × {matmulResult.result?.[0]?.length})
                        </span>
                        <p className="text-xs text-gray-400 mt-0.5">
                          Herhangi bir hücreye tıklayarak iç çarpım hesaplama adımlarını inceleyin:
                        </p>
                      </div>

                      {matmulResult.properties?.total_flops && (
                        <div className="flex items-center space-x-3 text-xs">
                          <span className="px-2.5 py-1 bg-indigo-50 text-indigo-700 rounded-lg font-mono font-medium">
                            {matmulResult.properties.total_flops} FLOPs
                          </span>
                          <span className="px-2.5 py-1 bg-gray-100 text-gray-600 rounded-lg font-mono">
                            Frobenius: {matmulResult.properties.frobenius_norm?.toFixed(3)}
                          </span>
                        </div>
                      )}
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-12 gap-6 items-start">
                      {/* Interactive Result Matrix C */}
                      <div className="md:col-span-6 bg-slate-50/80 p-5 rounded-2xl border border-gray-200/80">
                        <div
                          className="grid gap-2"
                          style={{
                            gridTemplateColumns: `repeat(${matmulResult.result?.[0]?.length || 2}, minmax(0, 1fr))`,
                          }}
                        >
                          {matmulResult.result?.map((row, r) =>
                            row.map((val, c) => {
                              const isSelected = selectedCell[0] === r && selectedCell[1] === c;
                              return (
                                <button
                                  key={`c-${r}-${c}`}
                                  onClick={() => {
                                    setSelectedCell([r, c]);
                                    runMatrixMultiply([r, c]);
                                  }}
                                  className={`p-3 rounded-xl font-mono text-base font-bold transition-all text-center border ${
                                    isSelected
                                      ? 'bg-amber-100 border-amber-500 text-amber-900 shadow-md ring-2 ring-amber-400/30 scale-105'
                                      : 'bg-white border-gray-200 text-gray-800 hover:border-indigo-300 hover:bg-indigo-50/30'
                                  }`}
                                >
                                  {val.toFixed(2)}
                                </button>
                              );
                            })
                          )}
                        </div>
                      </div>

                      {/* Step-by-Step Dot Product Cell Inspector */}
                      <div className="md:col-span-6 bg-indigo-50/50 p-5 rounded-2xl border border-indigo-100/80">
                        <div className="flex items-center justify-between mb-3">
                          <span className="text-xs font-bold text-indigo-950 flex items-center space-x-1.5">
                            <span className="w-2 h-2 rounded-full bg-amber-500" />
                            <span>Hücre C[{selectedCell[0]}, {selectedCell[1]}] Hesaplama Detayı</span>
                          </span>
                          <span className="text-[11px] font-mono text-indigo-600 bg-indigo-100/70 px-2 py-0.5 rounded">
                            Σ(k) A[{selectedCell[0]}, k] × B[k, {selectedCell[1]}]
                          </span>
                        </div>

                        {/* Cell calculation breakdown */}
                        <div className="space-y-2 mt-3 font-mono text-xs">
                          {matrixA[selectedCell[0]]?.map((a_val, k) => {
                            const b_val = matrixB[k]?.[selectedCell[1]] ?? 0;
                            const prod = a_val * b_val;
                            return (
                              <div
                                key={k}
                                className="flex items-center justify-between p-2 bg-white rounded-xl border border-indigo-100 shadow-xs"
                              >
                                <div className="flex items-center space-x-2">
                                  <span className="text-gray-400 text-[10px]">k={k}:</span>
                                  <span className="text-indigo-600 font-semibold">{a_val}</span>
                                  <span className="text-gray-400">×</span>
                                  <span className="text-emerald-600 font-semibold">{b_val}</span>
                                </div>
                                <span className="text-gray-900 font-bold">= {prod.toFixed(2)}</span>
                              </div>
                            );
                          })}
                        </div>

                        {/* Total cell sum */}
                        <div className="mt-4 pt-3 border-t border-indigo-100/80 flex items-center justify-between text-sm font-bold">
                          <span className="text-gray-600">Toplam Hücre Değeri:</span>
                          <span className="text-amber-700 bg-amber-50 border border-amber-200 px-3 py-1 rounded-xl font-mono">
                            {matmulResult.result?.[selectedCell[0]]?.[selectedCell[1]]?.toFixed(2)}
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                {/* 2. Transpose Result */}
                {matrixOp === 'transpose' && transposeResult?.success && (
                  <div className="bg-purple-50/50 p-6 rounded-2xl border border-purple-100 space-y-4">
                    <div className="flex items-center justify-between">
                      <h3 className="text-sm font-bold text-purple-950 flex items-center space-x-2">
                        <span>Matris Transpozu (A^T)</span>
                        {transposeResult.properties?.is_symmetric && (
                          <span className="px-2 py-0.5 text-[10px] bg-emerald-100 text-emerald-800 rounded-md font-sans">
                            Simetrik (A = A^T)
                          </span>
                        )}
                      </h3>
                      <span className="text-xs font-mono text-purple-700">
                        {transposeResult.properties?.transposed_shape?.join(' × ')}
                      </span>
                    </div>

                    <div
                      className="grid gap-2 max-w-sm"
                      style={{
                        gridTemplateColumns: `repeat(${transposeResult.result?.[0]?.length || 2}, minmax(0, 1fr))`,
                      }}
                    >
                      {transposeResult.result?.map((row, r) =>
                        row.map((val, c) => (
                          <div
                            key={`t-${r}-${c}`}
                            className="p-3 bg-white border border-purple-200 rounded-xl text-center font-mono font-bold text-purple-900 shadow-xs"
                          >
                            {val.toFixed(2)}
                          </div>
                        ))
                      )}
                    </div>
                  </div>
                )}

                {/* 3. Inverse Result */}
                {matrixOp === 'inverse' && inverseResult?.success && (
                  <div className="bg-rose-50/50 p-6 rounded-2xl border border-rose-100 space-y-4">
                    <div className="flex items-center justify-between">
                      <h3 className="text-sm font-bold text-rose-950">Matris Tersi: A⁻¹ (Gauss-Jordan)</h3>
                      <div className="flex items-center space-x-2 text-xs">
                        <span className="px-2 py-0.5 bg-rose-100 text-rose-800 rounded-md font-mono">
                          det(A) = {inverseResult.properties?.determinant?.toFixed(4)}
                        </span>
                        <span className="px-2 py-0.5 bg-emerald-100 text-emerald-800 rounded-md font-sans flex items-center space-x-1">
                          <CheckCircle2 className="w-3 h-3" />
                          <span>A × A⁻¹ = I</span>
                        </span>
                      </div>
                    </div>

                    <div
                      className="grid gap-2 max-w-sm"
                      style={{
                        gridTemplateColumns: `repeat(${inverseResult.result?.[0]?.length || 2}, minmax(0, 1fr))`,
                      }}
                    >
                      {inverseResult.result?.map((row, r) =>
                        row.map((val, c) => (
                          <div
                            key={`inv-${r}-${c}`}
                            className="p-3 bg-white border border-rose-200 rounded-xl text-center font-mono font-bold text-rose-900 shadow-xs"
                          >
                            {val.toFixed(3)}
                          </div>
                        ))
                      )}
                    </div>
                  </div>
                )}

                {/* 4. Determinant Result */}
                {matrixOp === 'determinant' && determinantResult?.success && (
                  <div className="bg-amber-50/50 p-6 rounded-2xl border border-amber-100 space-y-3">
                    <div className="flex items-center justify-between">
                      <h3 className="text-sm font-bold text-amber-950">Determinant Değeri</h3>
                      <span className={`text-xs px-2.5 py-0.5 rounded-full font-medium ${
                        determinantResult.is_singular ? 'bg-rose-100 text-rose-800' : 'bg-emerald-100 text-emerald-800'
                      }`}>
                        {determinantResult.is_singular ? 'Tekil (Tersi Yok)' : 'Tersine Çevrilebilir'}
                      </span>
                    </div>
                    <div className="text-3xl font-extrabold font-mono text-amber-900">
                      det(A) = {determinantResult.determinant?.toFixed(4)}
                    </div>
                    <p className="text-xs text-amber-800/80 leading-relaxed">
                      Geometrik olarak determinant, matrisin n-boyutlu uzayda alan veya hacimleri ne kadar ölçeklediğini (ve yönünü) gösterir. det(A) = 0 ise matris uzayı daha alt bir boyuta çökerterek tersinemez hale getirir.
                    </p>
                  </div>
                )}

                {/* 5. Eigenvalues Result */}
                {matrixOp === 'eigenvalues' && eigenResult?.success && (
                  <div className="bg-teal-50/50 p-6 rounded-2xl border border-teal-100 space-y-4">
                    <div className="flex items-center justify-between">
                      <h3 className="text-sm font-bold text-teal-950">Öz Değerler (Eigenvalues) & Öz Vektörler (Eigenvectors)</h3>
                      <span className="text-xs font-mono text-teal-700">
                        İz (Trace): {eigenResult.trace?.toFixed(2)}
                      </span>
                    </div>

                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                      {eigenResult.eigen_pairs?.map((pair) => (
                        <div key={pair.index} className="p-4 bg-white rounded-xl border border-teal-200 shadow-xs space-y-2">
                          <div className="flex items-center justify-between">
                            <span className="text-xs font-bold text-teal-800">
                              λ_{pair.index + 1} = {typeof pair.eigenvalue === 'number' ? pair.eigenvalue.toFixed(4) : JSON.stringify(pair.eigenvalue)}
                            </span>
                            <span className="text-[10px] text-gray-400 font-mono">
                              Büyüklük: {pair.magnitude.toFixed(3)}
                            </span>
                          </div>
                          <div className="text-xs text-gray-600 font-mono bg-slate-50 p-2 rounded-lg border border-gray-100">
                            v_{pair.index + 1} = [{Array.isArray(pair.eigenvector) ? (pair.eigenvector as Array<number | { real: number; imag: number }>).map((x) => typeof x === 'number' ? x.toFixed(3) : `${x.real.toFixed(2)}+${x.imag.toFixed(2)}i`).join(', ') : ''}]
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* ============================================================== */}
            {/* Vector Dot Product & Cosine Similarity Section                 */}
            {/* ============================================================== */}
            <div className="bg-white rounded-3xl border border-gray-200/80 p-6 sm:p-8 shadow-sm">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-6 border-b border-gray-100 gap-4">
                <div>
                  <h2 className="text-xl font-bold text-gray-900 flex items-center space-x-2">
                    <span className="p-1.5 bg-emerald-100 rounded-lg text-emerald-600">
                      <Compass className="w-5 h-5" />
                    </span>
                    <span>Vektör Nokta Çarpımı & Kosinüs Benzerliği (Cosine Similarity)</span>
                  </h2>
                  <p className="text-xs sm:text-sm text-gray-500 mt-1">
                    Embedding karşılaştırmalarında ve Attention mekanizmasında anahtar rol oynayan skaler çarpım ve vektör açısı geometrisi.
                  </p>
                </div>

                {/* Dimension & Presets */}
                <div className="flex items-center space-x-2">
                  <span className="text-xs text-gray-400 font-medium">Boyut:</span>
                  {[2, 3, 4].map((d) => (
                    <button
                      key={d}
                      onClick={() => handleVectorDimChange(d)}
                      className={`text-xs px-3 py-1.5 rounded-xl font-medium transition-all ${
                        vectorDim === d
                          ? 'bg-emerald-600 text-white'
                          : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                      }`}
                    >
                      {d}D
                    </button>
                  ))}
                </div>
              </div>

              {/* Vector Presets */}
              {vectorDim === 2 && (
                <div className="mt-4 flex items-center space-x-2 overflow-x-auto pb-2">
                  <span className="text-xs text-gray-400 font-medium">Örnekler:</span>
                  {VECTOR_PRESETS.map((p, idx) => (
                    <button
                      key={idx}
                      onClick={() => {
                        setVectorA(p.a);
                        setVectorB(p.b);
                        runVectorDot(p.a, p.b);
                      }}
                      className="text-xs px-3 py-1 bg-emerald-50 text-emerald-700 border border-emerald-100 rounded-lg hover:bg-emerald-100 font-medium whitespace-nowrap transition-colors"
                      title={p.desc}
                    >
                      {p.name}
                    </button>
                  ))}
                </div>
              )}

              {/* Vector Inputs & 2D SVG Graph */}
              <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-center mt-6">
                {/* Inputs */}
                <div className="lg:col-span-6 space-y-4">
                  {/* Vector A */}
                  <div className="p-4 bg-slate-50/80 rounded-2xl border border-gray-200/80">
                    <span className="text-xs font-bold text-indigo-700 block mb-2">Vektör a:</span>
                    <div className="flex items-center space-x-2">
                      {vectorA.map((val, i) => (
                        <input
                          key={`va-${i}`}
                          type="number"
                          step="any"
                          value={val}
                          onChange={(e) => {
                            const newA = [...vectorA];
                            newA[i] = parseFloat(e.target.value) || 0;
                            setVectorA(newA);
                            runVectorDot(newA, vectorB);
                          }}
                          className="w-full text-center py-2 px-2 rounded-xl text-sm font-mono font-bold bg-white border border-gray-200 text-gray-800"
                        />
                      ))}
                    </div>
                  </div>

                  {/* Vector B */}
                  <div className="p-4 bg-slate-50/80 rounded-2xl border border-gray-200/80">
                    <span className="text-xs font-bold text-emerald-700 block mb-2">Vektör b:</span>
                    <div className="flex items-center space-x-2">
                      {vectorB.map((val, i) => (
                        <input
                          key={`vb-${i}`}
                          type="number"
                          step="any"
                          value={val}
                          onChange={(e) => {
                            const newB = [...vectorB];
                            newB[i] = parseFloat(e.target.value) || 0;
                            setVectorB(newB);
                            runVectorDot(vectorA, newB);
                          }}
                          className="w-full text-center py-2 px-2 rounded-xl text-sm font-mono font-bold bg-white border border-gray-200 text-gray-800"
                        />
                      ))}
                    </div>
                  </div>

                  {/* Result Stats Grid */}
                  {vectorResult && (
                    <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                      <div className="p-3 bg-white rounded-xl border border-gray-200 text-center">
                        <span className="text-[11px] text-gray-400 block font-medium">Nokta Çarpımı (a · b)</span>
                        <span className="text-lg font-bold font-mono text-gray-900 mt-1 block">
                          {vectorResult.result?.[0]?.toFixed(2)}
                        </span>
                      </div>
                      <div className="p-3 bg-white rounded-xl border border-gray-200 text-center">
                        <span className="text-[11px] text-gray-400 block font-medium">Cosine Benzerliği</span>
                        <span className="text-lg font-bold font-mono text-emerald-600 mt-1 block">
                          {vectorResult.properties?.cosine_similarity?.toFixed(4) ?? '0.0000'}
                        </span>
                      </div>
                      <div className="p-3 bg-white rounded-xl border border-gray-200 text-center col-span-2 sm:col-span-1">
                        <span className="text-[11px] text-gray-400 block font-medium">Açı (θ)</span>
                        <span className="text-lg font-bold font-mono text-indigo-600 mt-1 block">
                          {vectorResult.properties?.angle_degrees?.toFixed(1) ?? '0.0'}°
                        </span>
                      </div>
                    </div>
                  )}
                </div>

                {/* 2D Vector Geometry Canvas */}
                <div className="lg:col-span-6 bg-slate-900 rounded-2xl p-4 flex flex-col items-center justify-center relative overflow-hidden shadow-inner min-h-[260px]">
                  <div className="text-[11px] font-mono text-slate-400 mb-2 flex items-center space-x-3">
                    <span className="flex items-center space-x-1">
                      <span className="w-2.5 h-2.5 rounded-full bg-indigo-400" />
                      <span>Vektör a [{vectorA.slice(0, 2).join(', ')}]</span>
                    </span>
                    <span className="flex items-center space-x-1">
                      <span className="w-2.5 h-2.5 rounded-full bg-emerald-400" />
                      <span>Vektör b [{vectorB.slice(0, 2).join(', ')}]</span>
                    </span>
                  </div>

                  <svg viewBox="-60 -60 120 120" className="w-56 h-56">
                    {/* Grid Circles */}
                    <circle cx="0" cy="0" r="20" fill="none" stroke="#334155" strokeWidth="0.5" strokeDasharray="2,2" />
                    <circle cx="0" cy="0" r="40" fill="none" stroke="#334155" strokeWidth="0.5" strokeDasharray="2,2" />
                    {/* Axes */}
                    <line x1="-55" y1="0" x2="55" y2="0" stroke="#475569" strokeWidth="0.8" />
                    <line x1="0" y1="-55" x2="0" y2="55" stroke="#475569" strokeWidth="0.8" />

                    {/* Vector a arrow */}
                    {(() => {
                      const scale = 8;
                      const ax = (vectorA[0] || 0) * scale;
                      const ay = -(vectorA[1] || 0) * scale;
                      return (
                        <g>
                          <line x1="0" y1="0" x2={ax} y2={ay} stroke="#818cf8" strokeWidth="2.5" strokeLinecap="round" />
                          <circle cx={ax} cy={ay} r="3" fill="#818cf8" />
                        </g>
                      );
                    })()}

                    {/* Vector b arrow */}
                    {(() => {
                      const scale = 8;
                      const bx = (vectorB[0] || 0) * scale;
                      const by = -(vectorB[1] || 0) * scale;
                      return (
                        <g>
                          <line x1="0" y1="0" x2={bx} y2={by} stroke="#34d399" strokeWidth="2.5" strokeLinecap="round" />
                          <circle cx={bx} cy={by} r="3" fill="#34d399" />
                        </g>
                      );
                    })()}
                  </svg>

                  {vectorResult?.properties?.is_orthogonal && (
                    <span className="absolute bottom-3 right-3 px-2 py-0.5 bg-emerald-500/20 text-emerald-300 text-[10px] font-mono rounded-md border border-emerald-500/40">
                      Ortogonal (Dik Vektörler)
                    </span>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ================================================================== */}
        {/* TAB 2: CALCULUS & GRADIENT DESCENT                                 */}
        {/* ================================================================== */}
        {activeTab === 'calculus' && (
          <div className="space-y-8 animate-fadeIn">
            {/* Top Interactive Function Selector */}
            <div className="bg-white rounded-3xl border border-gray-200/80 p-6 sm:p-8 shadow-sm">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-6 border-b border-gray-100 gap-4">
                <div>
                  <h2 className="text-xl font-bold text-gray-900 flex items-center space-x-2">
                    <span className="p-1.5 bg-indigo-100 rounded-lg text-indigo-600">
                      <TrendingDown className="w-5 h-5" />
                    </span>
                    <span>1D Gradiyen İnişi & Sayısal Türev Simülatörü</span>
                  </h2>
                  <p className="text-xs sm:text-sm text-gray-500 mt-1">
                    Fonksiyonu ve başlangıç noktasını seçin, öğrenme oranını (learning rate) ayarlayın ve optimizasyon adım adım inerken teğet eğimini görselleştirin.
                  </p>
                </div>

                {/* Function Selector Pills */}
                <div className="flex items-center space-x-2 flex-wrap gap-y-2">
                  {FUNCTION_PRESETS.map((fn) => (
                    <button
                      key={fn.id}
                      onClick={() => {
                        setSelectedFunc(fn.id);
                        fetchDerivative(fn.id);
                        runGradientDescent(fn.id);
                      }}
                      className={`text-xs px-3.5 py-1.5 rounded-xl font-medium transition-all ${
                        selectedFunc === fn.id
                          ? 'bg-indigo-600 text-white shadow-sm'
                          : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                      }`}
                    >
                      {fn.formula}
                    </button>
                  ))}
                </div>
              </div>

              {/* Calculus Curves & Dual SVG Viewer */}
              <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 mt-6 items-start">
                {/* SVG Graph Visualizer */}
                <div className="lg:col-span-8 bg-slate-900 rounded-2xl p-5 border border-slate-800 shadow-inner relative">
                  <div className="flex items-center justify-between text-xs font-mono text-slate-400 mb-3">
                    <div className="flex items-center space-x-4">
                      <span className="flex items-center space-x-1.5">
                        <span className="w-3 h-0.5 bg-indigo-400" />
                        <span className="text-indigo-300">f(x) = {derivData?.formula}</span>
                      </span>
                      <span className="flex items-center space-x-1.5">
                        <span className="w-3 h-0.5 bg-amber-400 border-b border-dashed" />
                        <span className="text-amber-300">f'(x) = {derivData?.derivative_formula}</span>
                      </span>
                    </div>

                    {currentGdStep && (
                      <span className="text-emerald-400 bg-emerald-950/70 border border-emerald-800/80 px-2 py-0.5 rounded">
                        Adım {scrubberStep}: x = {currentGdStep.x.toFixed(3)}, f(x) = {currentGdStep.y.toFixed(3)}
                      </span>
                    )}
                  </div>

                  {/* The SVG Canvas */}
                  <svg
                    viewBox={`0 0 ${SVG_WIDTH} ${SVG_HEIGHT}`}
                    className="w-full h-64 sm:h-72"
                  >
                    {/* Coordinate axes */}
                    <line
                      x1="0"
                      y1={toSvgY(0)}
                      x2={SVG_WIDTH}
                      y2={toSvgY(0)}
                      stroke="#334155"
                      strokeWidth="1"
                    />
                    <line
                      x1={toSvgX(0)}
                      y1="0"
                      x2={toSvgX(0)}
                      y2={SVG_HEIGHT}
                      stroke="#334155"
                      strokeWidth="1"
                    />

                    {/* f(x) curve */}
                    {fPath && (
                      <path
                        d={fPath}
                        fill="none"
                        stroke="#818cf8"
                        strokeWidth="2.5"
                        strokeLinecap="round"
                      />
                    )}

                    {/* f'(x) curve (dashed) */}
                    {fPrimePath && (
                      <path
                        d={fPrimePath}
                        fill="none"
                        stroke="#fbbf24"
                        strokeWidth="1.8"
                        strokeDasharray="4,4"
                        strokeOpacity="0.75"
                      />
                    )}

                    {/* Gradient Descent Trajectory Path */}
                    {gdResult?.history && gdResult.history.length > 1 && (
                      <g>
                        {/* Trajectory dots up to current scrubber step */}
                        {gdResult.history.slice(0, scrubberStep + 1).map((pt, idx) => (
                          <circle
                            key={idx}
                            cx={toSvgX(pt.x)}
                            cy={toSvgY(pt.y)}
                            r={idx === scrubberStep ? 5.5 : 3}
                            fill={idx === scrubberStep ? '#ef4444' : '#10b981'}
                            stroke="#ffffff"
                            strokeWidth={idx === scrubberStep ? 2 : 1}
                          />
                        ))}

                        {/* Tangent line at current point */}
                        {currentGdStep && (
                          <g>
                            {(() => {
                              const x0 = currentGdStep.x;
                              const y0 = currentGdStep.y;
                              const slope = currentGdStep.gradient;
                              const dx = 0.8;
                              const xLeft = x0 - dx;
                              const yLeft = y0 - slope * dx;
                              const xRight = x0 + dx;
                              const yRight = y0 + slope * dx;
                              return (
                                <line
                                  x1={toSvgX(xLeft)}
                                  y1={toSvgY(yLeft)}
                                  x2={toSvgX(xRight)}
                                  y2={toSvgY(yRight)}
                                  stroke="#ef4444"
                                  strokeWidth="2"
                                  strokeDasharray="3,3"
                                />
                              );
                            })()}
                          </g>
                        )}
                      </g>
                    )}
                  </svg>
                </div>

                {/* Simulation Parameters & Scrubber Controls */}
                <div className="lg:col-span-4 space-y-4">
                  <div className="p-5 bg-slate-50/80 rounded-2xl border border-gray-200/80 space-y-4">
                    <span className="text-xs font-bold text-gray-700 block tracking-wide">
                      Optimizasyon Parametreleri
                    </span>

                    {/* Initial X */}
                    <div>
                      <div className="flex justify-between text-xs text-gray-500 mb-1">
                        <span>Başlangıç Noktası (x₀):</span>
                        <span className="font-mono font-bold text-gray-800">{initialX}</span>
                      </div>
                      <input
                        type="range"
                        min="-3.0"
                        max="3.0"
                        step="0.1"
                        value={initialX}
                        onChange={(e) => {
                          const val = parseFloat(e.target.value);
                          setInitialX(val);
                          runGradientDescent(selectedFunc, val, learningRate, maxIterations);
                        }}
                        className="w-full accent-indigo-600"
                      />
                    </div>

                    {/* Learning Rate */}
                    <div>
                      <div className="flex justify-between text-xs text-gray-500 mb-1">
                        <span>Öğrenme Oranı (η / lr):</span>
                        <span className="font-mono font-bold text-gray-800">{learningRate}</span>
                      </div>
                      <input
                        type="range"
                        min="0.01"
                        max="0.8"
                        step="0.01"
                        value={learningRate}
                        onChange={(e) => {
                          const val = parseFloat(e.target.value);
                          setLearningRate(val);
                          runGradientDescent(selectedFunc, initialX, val, maxIterations);
                        }}
                        className="w-full accent-indigo-600"
                      />
                    </div>

                    {/* Max Iterations */}
                    <div>
                      <div className="flex justify-between text-xs text-gray-500 mb-1">
                        <span>Maksimum Adım:</span>
                        <span className="font-mono font-bold text-gray-800">{maxIterations}</span>
                      </div>
                      <input
                        type="range"
                        min="5"
                        max="60"
                        step="1"
                        value={maxIterations}
                        onChange={(e) => {
                          const val = parseInt(e.target.value);
                          setMaxIterations(val);
                          runGradientDescent(selectedFunc, initialX, learningRate, val);
                        }}
                        className="w-full accent-indigo-600"
                      />
                    </div>

                    <button
                      onClick={() => runGradientDescent()}
                      disabled={gdLoading}
                      className="w-full py-2.5 px-4 bg-indigo-600 hover:bg-indigo-700 text-white font-bold text-xs rounded-xl shadow-sm transition-all flex items-center justify-center space-x-2"
                    >
                      <RotateCcw className="w-3.5 h-3.5" />
                      <span>Simülasyonu Yeniden Başlat</span>
                    </button>
                  </div>

                  {/* Scrubber & Player Card */}
                  {gdResult && gdResult.history.length > 0 && (
                    <div className="p-5 bg-white rounded-2xl border border-gray-200/80 shadow-sm space-y-3">
                      <div className="flex items-center justify-between">
                        <span className="text-xs font-bold text-gray-700">Adım Oynatıcı & Scrubber</span>
                        <button
                          onClick={togglePlay}
                          className="p-1.5 bg-indigo-50 text-indigo-600 hover:bg-indigo-100 rounded-lg transition-colors flex items-center space-x-1 text-xs font-medium"
                        >
                          {isPlaying ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
                          <span>{isPlaying ? 'Durdur' : 'Oynat'}</span>
                        </button>
                      </div>

                      <input
                        type="range"
                        min="0"
                        max={gdResult.history.length - 1}
                        value={scrubberStep}
                        onChange={(e) => {
                          setIsPlaying(false);
                          if (timerRef.current) clearInterval(timerRef.current);
                          setScrubberStep(parseInt(e.target.value));
                        }}
                        className="w-full accent-indigo-600"
                      />

                      {/* Current Step Equation Box */}
                      {currentGdStep && (
                        <div className="p-3 bg-indigo-50/70 border border-indigo-100 rounded-xl text-xs space-y-1">
                          <span className="text-[10px] text-indigo-700 font-bold uppercase tracking-wider block">
                            Gradiyen Güncellemesi ({currentGdStep.iteration}. Adım):
                          </span>
                          <div className="font-mono text-indigo-950 font-bold">
                            x_{currentGdStep.iteration + 1} = x_{currentGdStep.iteration} - η × f'(x)
                          </div>
                          <div className="text-[11px] text-indigo-800/80 font-mono">
                            {currentGdStep.x.toFixed(4)} - ({learningRate} × {currentGdStep.gradient.toFixed(4)})
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </div>

              {/* Optimization Summary Bar */}
              {gdResult && (
                <div className="mt-6 pt-6 border-t border-gray-100 grid grid-cols-2 sm:grid-cols-4 gap-4">
                  <div className="p-4 bg-slate-50 rounded-2xl border border-gray-100 text-center">
                    <span className="text-xs text-gray-400 block font-medium">Son Konum (x*)</span>
                    <span className="text-xl font-extrabold font-mono text-gray-900 mt-1 block">
                      {gdResult.final_x?.toFixed(4)}
                    </span>
                  </div>
                  <div className="p-4 bg-slate-50 rounded-2xl border border-gray-100 text-center">
                    <span className="text-xs text-gray-400 block font-medium">Minimum Değer f(x*)</span>
                    <span className="text-xl font-extrabold font-mono text-indigo-600 mt-1 block">
                      {gdResult.final_y?.toFixed(4)}
                    </span>
                  </div>
                  <div className="p-4 bg-slate-50 rounded-2xl border border-gray-100 text-center">
                    <span className="text-xs text-gray-400 block font-medium">Toplam İterasyon</span>
                    <span className="text-xl font-extrabold font-mono text-gray-900 mt-1 block">
                      {gdResult.iterations}
                    </span>
                  </div>
                  <div className="p-4 bg-slate-50 rounded-2xl border border-gray-100 text-center">
                    <span className="text-xs text-gray-400 block font-medium">Yakınsama Durumu</span>
                    <span className={`text-sm font-bold mt-2 inline-flex items-center space-x-1 px-2.5 py-0.5 rounded-full ${
                      gdResult.converged ? 'bg-emerald-100 text-emerald-800' : 'bg-amber-100 text-amber-800'
                    }`}>
                      {gdResult.converged ? <CheckCircle2 className="w-3.5 h-3.5" /> : <AlertCircle className="w-3.5 h-3.5" />}
                      <span>{gdResult.converged ? 'Yakınsadı (Converged)' : 'Maksimum Adım'}</span>
                    </span>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* ================================================================== */}
        {/* TAB 3: CHAIN RULE & BACKPROPAGATION                                */}
        {/* ================================================================== */}
        {activeTab === 'chainrule' && (
          <div className="space-y-8 animate-fadeIn">
            <div className="bg-white rounded-3xl border border-gray-200/80 p-6 sm:p-8 shadow-sm">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-6 border-b border-gray-100 gap-4">
                <div>
                  <h2 className="text-xl font-bold text-gray-900 flex items-center space-x-2">
                    <span className="p-1.5 bg-purple-100 rounded-lg text-purple-600">
                      <Activity className="w-5 h-5" />
                    </span>
                    <span>Bileşke Fonksiyonlar & Zincir Kuralı (Chain Rule)</span>
                  </h2>
                  <p className="text-xs sm:text-sm text-gray-500 mt-1">
                    Yapay sinir ağlarındaki geriye yayılımın (backpropagation) temel kuralı: d/dx[f(g(x))] = f'(g(x)) × g'(x).
                  </p>
                </div>

                {/* X slider */}
                <div className="flex items-center space-x-3 bg-slate-50 p-2.5 rounded-2xl border border-gray-200">
                  <span className="text-xs text-gray-500 font-medium font-mono">x = {chainX.toFixed(1)}</span>
                  <input
                    type="range"
                    min="-3.0"
                    max="3.0"
                    step="0.1"
                    value={chainX}
                    onChange={(e) => {
                      const val = parseFloat(e.target.value);
                      setChainX(val);
                      runChainRule(outerFn, innerFn, val);
                    }}
                    className="accent-purple-600 w-28"
                  />
                </div>
              </div>

              {/* Function Pickers */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-6 mt-6">
                {/* Outer function selector */}
                <div className="p-5 bg-slate-50/80 rounded-2xl border border-gray-200/80">
                  <span className="text-xs font-bold text-purple-900 block mb-2">
                    Dış Fonksiyon f(u):
                  </span>
                  <div className="grid grid-cols-3 gap-2">
                    {[
                      { id: 'square', label: 'u² (Kare)' },
                      { id: 'sin', label: 'sin(u)' },
                      { id: 'exp', label: 'e^u (Üstel)' },
                    ].map((opt) => (
                      <button
                        key={opt.id}
                        onClick={() => {
                          setOuterFn(opt.id);
                          runChainRule(opt.id, innerFn, chainX);
                        }}
                        className={`text-xs p-2.5 rounded-xl font-medium text-center transition-all ${
                          outerFn === opt.id
                            ? 'bg-purple-600 text-white shadow-sm font-bold'
                            : 'bg-white text-gray-700 border border-gray-200 hover:bg-gray-100'
                        }`}
                      >
                        {opt.label}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Inner function selector */}
                <div className="p-5 bg-slate-50/80 rounded-2xl border border-gray-200/80">
                  <span className="text-xs font-bold text-emerald-900 block mb-2">
                    İç Fonksiyon g(x):
                  </span>
                  <div className="grid grid-cols-3 gap-2">
                    {[
                      { id: 'linear', label: '2x + 1' },
                      { id: 'quadratic', label: 'x²' },
                      { id: 'cubic', label: 'x³' },
                    ].map((opt) => (
                      <button
                        key={opt.id}
                        onClick={() => {
                          setInnerFn(opt.id);
                          runChainRule(outerFn, opt.id, chainX);
                        }}
                        className={`text-xs p-2.5 rounded-xl font-medium text-center transition-all ${
                          innerFn === opt.id
                            ? 'bg-emerald-600 text-white shadow-sm font-bold'
                            : 'bg-white text-gray-700 border border-gray-200 hover:bg-gray-100'
                        }`}
                      >
                        {opt.label}
                      </button>
                    ))}
                  </div>
                </div>
              </div>

              {/* Dynamic Forward & Backward Computation Graph */}
              {chainResult?.success && (
                <div className="mt-8 space-y-6">
                  {/* Composite formula badge */}
                  <div className="text-center p-3 bg-purple-50/50 rounded-2xl border border-purple-100">
                    <span className="text-xs text-purple-600 font-medium">Bileşke Fonksiyon:</span>
                    <div className="text-lg font-bold font-mono text-purple-950 mt-0.5">
                      {chainResult.composite_formula}
                    </div>
                  </div>

                  {/* 3 Step Decomposition Cards */}
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                    {/* Step 1: Inner */}
                    <div className="p-5 bg-white rounded-2xl border border-emerald-200/90 shadow-xs space-y-3">
                      <div className="flex items-center space-x-2 text-xs font-bold text-emerald-700">
                        <span className="w-5 h-5 rounded-full bg-emerald-100 flex items-center justify-center text-[10px]">1</span>
                        <span>İç Fonksiyon (u = g(x))</span>
                      </div>
                      <div className="font-mono text-sm font-bold text-gray-800">
                        g({chainX}) = {chainResult.steps[0]?.value_at_x?.toFixed(3)}
                      </div>
                      <div className="text-xs text-emerald-800 bg-emerald-50 p-2.5 rounded-xl border border-emerald-100 font-mono">
                        g'(x) = {chainResult.steps[0]?.derivative} <br />
                        <span className="font-bold text-emerald-950">g'({chainX}) = {chainResult.steps[0]?.derivative_at_x?.toFixed(3)}</span>
                      </div>
                    </div>

                    {/* Step 2: Outer */}
                    <div className="p-5 bg-white rounded-2xl border border-purple-200/90 shadow-xs space-y-3">
                      <div className="flex items-center space-x-2 text-xs font-bold text-purple-700">
                        <span className="w-5 h-5 rounded-full bg-purple-100 flex items-center justify-center text-[10px]">2</span>
                        <span>Dış Fonksiyon (y = f(u))</span>
                      </div>
                      <div className="font-mono text-sm font-bold text-gray-800">
                        f({chainResult.steps[0]?.value_at_x?.toFixed(2)}) = {chainResult.steps[1]?.value_at_g_x?.toFixed(3)}
                      </div>
                      <div className="text-xs text-purple-800 bg-purple-50 p-2.5 rounded-xl border border-purple-100 font-mono">
                        f'(u) = {chainResult.steps[1]?.derivative} <br />
                        <span className="font-bold text-purple-950">f'(u) = {chainResult.steps[1]?.derivative_at_g_x?.toFixed(3)}</span>
                      </div>
                    </div>

                    {/* Step 3: Synthesis */}
                    <div className="p-5 bg-white rounded-2xl border border-indigo-200/90 shadow-xs space-y-3">
                      <div className="flex items-center space-x-2 text-xs font-bold text-indigo-700">
                        <span className="w-5 h-5 rounded-full bg-indigo-100 flex items-center justify-center text-[10px]">3</span>
                        <span>Zincir Kuralı Çarpımı</span>
                      </div>
                      <div className="font-mono text-xs text-gray-600">
                        df/dx = f'(g(x)) × g'(x)
                      </div>
                      <div className="text-xs text-indigo-900 bg-indigo-50 p-2.5 rounded-xl border border-indigo-100 font-mono">
                        {chainResult.steps[2]?.calculation}
                      </div>
                      <div className="text-lg font-black font-mono text-indigo-600">
                        df/dx = {chainResult.final_derivative?.toFixed(4)}
                      </div>
                    </div>
                  </div>

                  {/* Deep Learning Backpropagation Callout */}
                  <div className="p-6 bg-gradient-to-r from-purple-50 via-indigo-50 to-emerald-50 rounded-2xl border border-indigo-100/80 flex items-start space-x-4">
                    <Sparkles className="w-6 h-6 text-indigo-600 flex-shrink-0 mt-0.5" />
                    <div className="space-y-1.5 text-xs text-gray-700 leading-relaxed">
                      <span className="font-bold text-sm text-gray-900 block">
                        Yapay Zekada Neden Hayati Önem Taşır?
                      </span>
                      <p>
                        Yüz milyarlarca parametreli Transformer ve LLM modellerinde, kaybın (loss) her bir ağırlığa ($W$) göre türevi hesaplanırken, bu zincir kuralı geriye doğru topoojik sıra ile uygulanır (Backpropagation / Tersine Yayılım).
                      </p>
                      <p className="text-indigo-800 font-medium">
                        Her katman kendi yerel türevini hesaplar ve yukarıdan gelen gradyanla çarparak önceki katmana aktarır: <code>dL/dx = (dL/dy) × (dy/dx)</code>.
                      </p>
                    </div>
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
