'use client';

import React, { useState, useEffect, useMemo } from 'react';
import Link from 'next/link';
import {
  Brain,
  Play,
  RotateCcw,
  Sparkles,
  Layers,
  Activity,
  AlertTriangle,
  CheckCircle2,
  Trophy,
  Zap,
  Info,
  Sliders,
  ChevronRight,
  TrendingDown,
  RefreshCw,
  Eye,
} from 'lucide-react';
import {
  api,
  MLPSimulateResponse,
  MLPNode,
  MLPSynapse,
  OptimizerRaceResponse,
  LandscapesListResponse,
  ActivationCurvesResponse,
} from '@/lib/api';

// Presets for MLP Tab
const MLP_PRESETS = [
  {
    name: 'XOR Tarzı Doğrusal Olmayan',
    inputs: [0.9, -0.8],
    targets: [1.0, 0.0],
    hidden: [4, 4],
    activation: 'gelu',
    desc: 'İki zıt kutuplu giriş; gizli katmanlar doğrusal olmayan karar sınırları oluşturur.',
  },
  {
    name: 'Ölü ReLU Senaryosu (Dying ReLU)',
    inputs: [-2.5, -3.0],
    targets: [0.0, 0.5],
    hidden: [4, 4],
    activation: 'relu',
    desc: 'Tüm girişler negatif; ReLU nöronları sıfıra kilitlenir ve gradiyen akışı durur.',
  },
  {
    name: 'Simetrik Doygunluk (Tanh / Sigmoid)',
    inputs: [3.5, -3.5],
    targets: [0.8, -0.8],
    hidden: [4, 3],
    activation: 'tanh',
    desc: 'Uç değerler aktivasyon fonksiyonunu doyurur (vanishing gradient riski).',
  },
];

export default function NNLabPage() {
  const [activeTab, setActiveTab] = useState<'mlp' | 'race' | 'activations'>('mlp');

  // ==========================================================================
  // Tab 1: MLP Simulator State
  // ==========================================================================
  const [mlpInputs, setMlpInputs] = useState<string>('0.8, -0.5');
  const [mlpTargets, setMlpTargets] = useState<string>('1.0, 0.0');
  const [mlpHidden, setMlpHidden] = useState<string>('4, 4');
  const [mlpActivation, setMlpActivation] = useState<string>('gelu');
  const [mlpLossFn, setMlpLossFn] = useState<string>('mse');
  const [mlpLr, setMlpLr] = useState<number>(0.05);
  const [mlpSeed, setMlpSeed] = useState<number>(42);

  const [mlpLoading, setMlpLoading] = useState<boolean>(false);
  const [mlpData, setMlpData] = useState<MLPSimulateResponse | null>(null);
  const [mlpError, setMlpError] = useState<string | null>(null);

  const [selectedNode, setSelectedNode] = useState<MLPNode | null>(null);
  const [selectedSynapse, setSelectedSynapse] = useState<MLPSynapse | null>(null);
  const [graphDisplayMode, setGraphDisplayMode] = useState<'activations' | 'gradients'>('activations');

  // ==========================================================================
  // Tab 2: Optimizer Race State
  // ==========================================================================
  const [landscapesMeta, setLandscapesMeta] = useState<LandscapesListResponse | null>(null);
  const [selectedLandscape, setSelectedLandscape] = useState<string>('quadratic_bowl');
  const [selectedOptimizers, setSelectedOptimizers] = useState<string[]>([
    'adamw',
    'adam',
    'rmsprop',
    'momentum',
    'sgd',
  ]);
  const [raceLr, setRaceLr] = useState<number>(0.03);
  const [raceMomentum, setRaceMomentum] = useState<number>(0.9);
  const [raceWeightDecay, setRaceWeightDecay] = useState<number>(0.01);
  const [raceSteps, setRaceSteps] = useState<number>(50);

  const [raceLoading, setRaceLoading] = useState<boolean>(false);
  const [raceData, setRaceData] = useState<OptimizerRaceResponse | null>(null);
  const [raceError, setRaceError] = useState<string | null>(null);

  // Playback Animation
  const [currentStep, setCurrentStep] = useState<number>(0);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);

  // ==========================================================================
  // Tab 3: Activation Dynamics State
  // ==========================================================================
  const [actData, setActData] = useState<ActivationCurvesResponse | null>(null);
  const [actLoading, setActLoading] = useState<boolean>(false);
  const [probeX, setProbeX] = useState<number>(0.5);

  // Initial Data Load
  useEffect(() => {
    runMlpSimulation();
    loadLandscapes();
    loadActivations();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Animation effect for Optimizer Race
  useEffect(() => {
    let timer: any;
    if (isPlaying && raceData) {
      timer = setInterval(() => {
        setCurrentStep((prev) => {
          if (prev >= raceSteps) {
            setIsPlaying(false);
            return prev;
          }
          return prev + 1;
        });
      }, 120);
    }
    return () => clearInterval(timer);
  }, [isPlaying, raceData, raceSteps]);

  // Load Landscapes metadata
  const loadLandscapes = async () => {
    try {
      const data = await api.nnLab.getLandscapes();
      setLandscapesMeta(data);
    } catch (err) {
      console.warn('Could not load landscapes', err);
    }
  };

  // Load Activations metadata
  const loadActivations = async () => {
    setActLoading(true);
    try {
      const data = await api.nnLab.getActivations(80);
      setActData(data);
    } catch (err) {
      console.warn('Could not load activations', err);
    } finally {
      setActLoading(false);
    }
  };

  // Run MLP Simulation
  const runMlpSimulation = async () => {
    setMlpLoading(true);
    setMlpError(null);
    try {
      const parsedInputs = mlpInputs
        .split(',')
        .map((s) => parseFloat(s.trim()))
        .filter((n) => !isNaN(n));
      const parsedTargets = mlpTargets
        .split(',')
        .map((s) => parseFloat(s.trim()))
        .filter((n) => !isNaN(n));
      const parsedHidden = mlpHidden
        .split(',')
        .map((s) => parseInt(s.trim(), 10))
        .filter((n) => !isNaN(n) && n > 0);

      if (parsedInputs.length === 0 || parsedTargets.length === 0) {
        setMlpError('Lütfen geçerli giriş ve hedef değerleri giriniz.');
        setMlpLoading(false);
        return;
      }

      const res = await api.nnLab.simulateMLP({
        inputs: parsedInputs,
        targets: parsedTargets,
        hidden_dims: parsedHidden.length > 0 ? parsedHidden : [4, 4],
        activation: mlpActivation,
        loss_function: mlpLossFn,
        learning_rate: mlpLr,
        seed: mlpSeed,
      });

      setMlpData(res);
      setSelectedNode(null);
      setSelectedSynapse(null);
    } catch (err: any) {
      setMlpError(err?.response?.data?.detail || 'MLP simülasyonu çalıştırılırken hata oluştu.');
    } finally {
      setMlpLoading(false);
    }
  };

  // Run Optimizer Race
  const runOptimizerRace = async () => {
    setRaceLoading(true);
    setRaceError(null);
    setIsPlaying(false);
    try {
      const res = await api.nnLab.raceOptimizers({
        landscape: selectedLandscape,
        optimizers: selectedOptimizers,
        learning_rate: raceLr,
        momentum: raceMomentum,
        weight_decay: raceWeightDecay,
        steps: raceSteps,
      });
      setRaceData(res);
      setCurrentStep(raceSteps);
    } catch (err: any) {
      setRaceError(err?.response?.data?.detail || 'Optimizer yarışı çalıştırılırken hata oluştu.');
    } finally {
      setRaceLoading(false);
    }
  };

  // Toggle Optimizer Selection
  const toggleOptimizer = (optKey: string) => {
    setSelectedOptimizers((prev) => {
      if (prev.includes(optKey)) {
        if (prev.length <= 1) return prev;
        return prev.filter((k) => k !== optKey);
      } else {
        return [...prev, optKey];
      }
    });
  };

  // Apply Preset
  const applyPreset = (preset: (typeof MLP_PRESETS)[0]) => {
    setMlpInputs(preset.inputs.join(', '));
    setMlpTargets(preset.targets.join(', '));
    setMlpHidden(preset.hidden.join(', '));
    setMlpActivation(preset.activation);
  };

  // SVG Geometry for MLP Graph
  const graphLayout = useMemo(() => {
    if (!mlpData) return null;
    const width = 640;
    const height = 360;
    const paddingX = 70;
    const paddingY = 40;

    const layerCount = mlpData.architecture.length;
    const colSpacing = (width - 2 * paddingX) / (layerCount - 1);

    const layerNodes: Record<number, MLPNode[]> = {};
    mlpData.nodes.forEach((n) => {
      if (!layerNodes[n.layer_idx]) layerNodes[n.layer_idx] = [];
      layerNodes[n.layer_idx].push(n);
    });

    const coords: Record<string, { x: number; y: number }> = {};

    Object.entries(layerNodes).forEach(([lIdxStr, nodes]) => {
      const lIdx = parseInt(lIdxStr, 10);
      const x = paddingX + lIdx * colSpacing;
      const count = nodes.length;
      const rowSpacing = (height - 2 * paddingY) / (count + 1);

      nodes.forEach((n, idx) => {
        const y = paddingY + (idx + 1) * rowSpacing;
        coords[n.id] = { x, y };
      });
    });

    return { width, height, coords };
  }, [mlpData]);

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 pb-16">
      {/* Top Banner */}
      <div className="bg-gradient-to-r from-slate-900 via-indigo-950 to-slate-900 text-white border-b border-indigo-900/50 shadow-md">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
            <div>
              <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-500/20 text-indigo-300 text-xs font-semibold tracking-wide uppercase mb-3 border border-indigo-400/30">
                <Brain className="w-3.5 h-3.5 text-indigo-400" />
                Bölüm 4 • Derin Öğrenme Temelleri &amp; Geriye Yayılım
              </div>
              <h1 className="text-3xl font-extrabold tracking-tight sm:text-4xl text-white">
                Neural Network &amp; Backprop Lab
              </h1>
              <p className="mt-2 text-base text-slate-300 max-w-3xl leading-relaxed">
                Yapay sinir ağlarında ileri ve geri geçiş (autograd) dinamiklerini, zincir kuralı türevlerini ve 2D kayıp
                manzarasında AdamW vs SGD optimizer yarışını interaktif keşfedin.
              </p>
            </div>

            <div className="flex items-center gap-3">
              <Link
                href="/tensor-lab"
                className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-slate-800/80 hover:bg-slate-700/80 text-slate-200 text-sm font-medium transition border border-slate-700/60 shadow-sm"
              >
                <Layers className="w-4 h-4 text-indigo-400" />
                Tensor Lab
              </Link>
              <Link
                href="/transformer-lab"
                className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white text-sm font-semibold transition shadow-md shadow-indigo-600/30"
              >
                <Zap className="w-4 h-4" />
                Transformer Lab
              </Link>
            </div>
          </div>

          {/* Navigation Tabs */}
          <div className="flex space-x-1 mt-8 overflow-x-auto border-b border-slate-800 pb-1 scrollbar-none">
            <button
              id="tab-mlp"
              onClick={() => setActiveTab('mlp')}
              className={`flex items-center gap-2 px-4 py-2.5 text-sm font-medium rounded-t-xl transition-colors whitespace-nowrap ${
                activeTab === 'mlp'
                  ? 'bg-slate-50 text-indigo-900 shadow-sm font-semibold'
                  : 'text-slate-300 hover:text-white hover:bg-white/5'
              }`}
            >
              <Brain className="w-4 h-4 text-indigo-600" />
              MLP Ağ Mimarisi &amp; Geriye Yayılım (Backprop Graph)
            </button>
            <button
              id="tab-race"
              onClick={() => {
                setActiveTab('race');
                if (!raceData) runOptimizerRace();
              }}
              className={`flex items-center gap-2 px-4 py-2.5 text-sm font-medium rounded-t-xl transition-colors whitespace-nowrap ${
                activeTab === 'race'
                  ? 'bg-slate-50 text-indigo-900 shadow-sm font-semibold'
                  : 'text-slate-300 hover:text-white hover:bg-white/5'
              }`}
            >
              <Trophy className="w-4 h-4 text-amber-500" />
              2D Kayıp Manzarası &amp; Optimizer Yarışı (Loss Race)
            </button>
            <button
              id="tab-activations"
              onClick={() => setActiveTab('activations')}
              className={`flex items-center gap-2 px-4 py-2.5 text-sm font-medium rounded-t-xl transition-colors whitespace-nowrap ${
                activeTab === 'activations'
                  ? 'bg-slate-50 text-indigo-900 shadow-sm font-semibold'
                  : 'text-slate-300 hover:text-white hover:bg-white/5'
              }`}
            >
              <Activity className="w-4 h-4 text-emerald-500" />
              Aktivasyon Dinamikleri &amp; Gradiyen Doyumu
            </button>
          </div>
        </div>
      </div>

      {/* Main Container */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 mt-6">
        {/* ================================================================== */}
        {/* TAB 1: MLP ARCHITECTURE & BACKPROPAGATION GRAPH                     */}
        {/* ================================================================== */}
        {activeTab === 'mlp' && (
          <div className="space-y-6">
            {/* Presets & Configurations */}
            <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200 space-y-4">
              <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 border-b border-slate-100 pb-3">
                <div>
                  <h2 className="text-base font-bold text-slate-900 flex items-center gap-2">
                    <Sliders className="w-5 h-5 text-indigo-600" />
                    MLP Mimari &amp; Hesaplama Parametreleri
                  </h2>
                  <p className="text-xs text-slate-500 mt-0.5">
                    Giriş vektörlerini, gizli katman boyutlarını ve aktivasyon türünü belirleyerek ileri ve geri yayılımı
                    tetikleyin.
                  </p>
                </div>

                <button
                  id="btn-simulate-mlp"
                  onClick={runMlpSimulation}
                  disabled={mlpLoading}
                  className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white font-semibold text-xs transition shadow-md shadow-indigo-600/20 disabled:opacity-50"
                >
                  {mlpLoading ? (
                    <>
                      <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                      Hesaplanıyor...
                    </>
                  ) : (
                    <>
                      <Play className="w-3.5 h-3.5" />
                      Ağı Simüle Et (Forward &amp; Backward)
                    </>
                  )}
                </button>
              </div>

              {/* Quick Presets */}
              <div className="flex flex-wrap items-center gap-2 pt-1">
                <span className="text-xs text-slate-500 font-medium">Hazır Senaryolar:</span>
                {MLP_PRESETS.map((p, idx) => (
                  <button
                    key={idx}
                    type="button"
                    onClick={() => applyPreset(p)}
                    className="px-3 py-1 rounded-lg text-xs font-medium bg-slate-100 hover:bg-indigo-50 hover:text-indigo-700 text-slate-700 border border-slate-200 transition"
                  >
                    {p.name}
                  </button>
                ))}
              </div>

              {/* Form Grid */}
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-6 gap-4 pt-2">
                <div>
                  <label htmlFor="input-mlp-x" className="block text-xs font-bold text-slate-700 mb-1">
                    Giriş Vektörü (x)
                  </label>
                  <input
                    id="input-mlp-x"
                    type="text"
                    value={mlpInputs}
                    onChange={(e) => setMlpInputs(e.target.value)}
                    placeholder="0.8, -0.5"
                    className="w-full p-2 rounded-xl border border-slate-200 font-mono text-xs bg-slate-50"
                  />
                </div>

                <div>
                  <label htmlFor="input-mlp-y" className="block text-xs font-bold text-slate-700 mb-1">
                    Hedef Vektör (y)
                  </label>
                  <input
                    id="input-mlp-y"
                    type="text"
                    value={mlpTargets}
                    onChange={(e) => setMlpTargets(e.target.value)}
                    placeholder="1.0, 0.0"
                    className="w-full p-2 rounded-xl border border-slate-200 font-mono text-xs bg-slate-50"
                  />
                </div>

                <div>
                  <label htmlFor="input-mlp-hidden" className="block text-xs font-bold text-slate-700 mb-1">
                    Gizli Katmanlar
                  </label>
                  <input
                    id="input-mlp-hidden"
                    type="text"
                    value={mlpHidden}
                    onChange={(e) => setMlpHidden(e.target.value)}
                    placeholder="4, 4"
                    className="w-full p-2 rounded-xl border border-slate-200 font-mono text-xs bg-slate-50"
                  />
                </div>

                <div>
                  <label htmlFor="select-mlp-act" className="block text-xs font-bold text-slate-700 mb-1">
                    Aktivasyon Fonk.
                  </label>
                  <select
                    id="select-mlp-act"
                    value={mlpActivation}
                    onChange={(e) => setMlpActivation(e.target.value)}
                    className="w-full p-2 rounded-xl border border-slate-200 text-xs bg-white font-medium"
                  >
                    <option value="relu">ReLU (max(0, z))</option>
                    <option value="gelu">GELU (Gaussian Error)</option>
                    <option value="swiglu">SwiGLU (SiLU * z)</option>
                    <option value="sigmoid">Sigmoid (1/(1+e^-z))</option>
                    <option value="tanh">Tanh (Hiperbolik Tanjant)</option>
                    <option value="leaky_relu">LeakyReLU (α=0.1)</option>
                  </select>
                </div>

                <div>
                  <label htmlFor="select-mlp-loss" className="block text-xs font-bold text-slate-700 mb-1">
                    Kayıp Fonksiyonu
                  </label>
                  <select
                    id="select-mlp-loss"
                    value={mlpLossFn}
                    onChange={(e) => setMlpLossFn(e.target.value)}
                    className="w-full p-2 rounded-xl border border-slate-200 text-xs bg-white font-medium"
                  >
                    <option value="mse">MSE (Mean Squared Error)</option>
                    <option value="cross_entropy">Cross-Entropy</option>
                    <option value="binary_cross_entropy">Binary Cross-Entropy</option>
                    <option value="smooth_l1">Smooth L1 (Huber Loss)</option>
                  </select>
                </div>

                <div>
                  <label htmlFor="input-mlp-lr" className="block text-xs font-bold text-slate-700 mb-1">
                    Öğrenme Oranı (η): {mlpLr}
                  </label>
                  <input
                    id="input-mlp-lr"
                    type="range"
                    min="0.01"
                    max="0.5"
                    step="0.01"
                    value={mlpLr}
                    onChange={(e) => setMlpLr(parseFloat(e.target.value))}
                    className="w-full accent-indigo-600 mt-2"
                  />
                </div>
              </div>
            </div>

            {/* Error Banner */}
            {mlpError && (
              <div className="p-4 rounded-xl bg-red-50 border border-red-200 text-red-700 flex items-center gap-3 text-sm">
                <AlertTriangle className="w-5 h-5 flex-shrink-0" />
                <span>{mlpError}</span>
              </div>
            )}

            {/* Simulation Results */}
            {mlpData && (
              <div className="space-y-6">
                {/* Health & Summary Banner */}
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                  <div className="p-4 rounded-xl bg-white border border-slate-200 shadow-sm">
                    <span className="text-xs text-slate-400 font-bold uppercase tracking-wider block">Kayıp (Loss)</span>
                    <span className="text-2xl font-black text-slate-900 font-mono mt-0.5 block">{mlpData.loss}</span>
                    <span className="text-[11px] text-slate-500 font-mono mt-1 block uppercase">
                      {mlpData.loss_function}
                    </span>
                  </div>

                  <div className="p-4 rounded-xl bg-white border border-slate-200 shadow-sm">
                    <span className="text-xs text-slate-400 font-bold uppercase tracking-wider block">
                      Ağ Mimarisi (Katmanlar)
                    </span>
                    <span className="text-base font-bold text-indigo-700 font-mono mt-1 block">
                      {mlpData.architecture.join(' ➔ ')}
                    </span>
                    <span className="text-[11px] text-slate-500 mt-1 block">
                      Toplam {mlpData.nodes.length} Nöron, {mlpData.synapses.length} Ağırlık
                    </span>
                  </div>

                  <div className="p-4 rounded-xl bg-white border border-slate-200 shadow-sm">
                    <span className="text-xs text-slate-400 font-bold uppercase tracking-wider block">
                      Model Çıktısı vs Hedef
                    </span>
                    <span className="text-xs font-mono font-bold text-slate-800 mt-1 block">
                      Tahmin: [{mlpData.predictions.join(', ')}]
                    </span>
                    <span className="text-[11px] font-mono text-slate-500 mt-0.5 block">
                      Hedef: [{mlpData.targets.join(', ')}]
                    </span>
                  </div>

                  <div
                    className={`p-4 rounded-xl border shadow-sm ${
                      mlpData.overall_health === 'HEALTHY'
                        ? 'bg-emerald-50/50 border-emerald-200 text-emerald-900'
                        : mlpData.overall_health === 'DEAD_NEURONS_DETECTED'
                        ? 'bg-amber-50/50 border-amber-200 text-amber-900'
                        : 'bg-rose-50/50 border-rose-200 text-rose-900'
                    }`}
                  >
                    <span className="text-xs font-bold uppercase tracking-wider block">Gradiyen Sağlığı</span>
                    <div className="flex items-center gap-2 mt-1">
                      {mlpData.overall_health === 'HEALTHY' ? (
                        <CheckCircle2 className="w-5 h-5 text-emerald-600" />
                      ) : (
                        <AlertTriangle className="w-5 h-5 text-amber-600" />
                      )}
                      <span className="font-bold text-sm">{mlpData.overall_health}</span>
                    </div>
                    {mlpData.dead_neurons.length > 0 && (
                      <span className="text-[11px] text-amber-700 block mt-1 font-mono">
                        {mlpData.dead_neurons.length} Ölü Nöron: {mlpData.dead_neurons.join(', ')}
                      </span>
                    )}
                  </div>
                </div>

                {/* Main Graph & Inspector Grid */}
                <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
                  {/* Left: SVG Neural Graph */}
                  <div className="lg:col-span-8 bg-white p-6 rounded-2xl shadow-sm border border-slate-200 space-y-4">
                    <div className="flex items-center justify-between border-b border-slate-100 pb-3">
                      <div>
                        <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
                          <Brain className="w-4 h-4 text-indigo-600" />
                          İnteraktif Hesaplama Grafiği (Autograd Flow)
                        </h3>
                        <p className="text-xs text-slate-400">
                          Nöron veya ağırlık çizgilerine tıklayarak detaylı matematiksel durumu inceleyin.
                        </p>
                      </div>

                      {/* View Mode Toggle */}
                      <div className="flex items-center p-1 bg-slate-100 rounded-xl text-xs">
                        <button
                          type="button"
                          onClick={() => setGraphDisplayMode('activations')}
                          className={`px-3 py-1 rounded-lg font-semibold transition ${
                            graphDisplayMode === 'activations'
                              ? 'bg-white text-indigo-900 shadow-sm'
                              : 'text-slate-600 hover:text-slate-900'
                          }`}
                        >
                          Aktivasyonlar (a)
                        </button>
                        <button
                          type="button"
                          onClick={() => setGraphDisplayMode('gradients')}
                          className={`px-3 py-1 rounded-lg font-semibold transition ${
                            graphDisplayMode === 'gradients'
                              ? 'bg-white text-indigo-900 shadow-sm'
                              : 'text-slate-600 hover:text-slate-900'
                          }`}
                        >
                          Gradiyenler (∂L/∂z)
                        </button>
                      </div>
                    </div>

                    {/* SVG Canvas */}
                    {graphLayout && (
                      <div className="relative w-full overflow-hidden bg-slate-950 rounded-xl border border-slate-800 p-2 flex items-center justify-center">
                        <svg
                          viewBox={`0 0 ${graphLayout.width} ${graphLayout.height}`}
                          className="w-full h-auto max-h-[420px]"
                        >
                          <defs>
                            <radialGradient id="node-glow-active" cx="50%" cy="50%" r="50%">
                              <stop offset="0%" stopColor="#818cf8" stopOpacity="0.8" />
                              <stop offset="100%" stopColor="#4f46e5" stopOpacity="0.1" />
                            </radialGradient>
                          </defs>

                          {/* Synapse Lines */}
                          {mlpData.synapses.map((syn) => {
                            const src = graphLayout.coords[syn.source_id];
                            const tgt = graphLayout.coords[syn.target_id];
                            if (!src || !tgt) return null;

                            const isSelected = selectedSynapse?.id === syn.id;
                            const absWeight = Math.abs(syn.weight);
                            const strokeWidth = Math.max(1, Math.min(4, absWeight * 2));
                            const strokeColor =
                              graphDisplayMode === 'gradients'
                                ? Math.abs(syn.gradient_w) > 0.1
                                  ? '#ec4899'
                                  : '#64748b'
                                : syn.weight >= 0
                                ? '#6366f1'
                                : '#f43f5e';

                            return (
                              <line
                                key={syn.id}
                                x1={src.x}
                                y1={src.y}
                                x2={tgt.x}
                                y2={tgt.y}
                                stroke={isSelected ? '#38bdf8' : strokeColor}
                                strokeWidth={isSelected ? 4 : strokeWidth}
                                strokeOpacity={isSelected ? 1.0 : 0.45}
                                className="cursor-pointer transition-all hover:stroke-opacity-100"
                                onClick={() => {
                                  setSelectedSynapse(syn);
                                  setSelectedNode(null);
                                }}
                              >
                                <title>{`Ağırlık: ${syn.weight}\nGradiyen: ${syn.gradient_w}`}</title>
                              </line>
                            );
                          })}

                          {/* Nodes */}
                          {mlpData.nodes.map((node) => {
                            const coord = graphLayout.coords[node.id];
                            if (!coord) return null;

                            const isSelected = selectedNode?.id === node.id;
                            const isDead = node.is_dead;

                            // Color based on activation or gradient
                            let nodeFill = '#1e1b4b'; // dark indigo
                            if (isDead) {
                              nodeFill = '#450a0a'; // dead red
                            } else if (graphDisplayMode === 'activations') {
                              const actVal = node.post_activation_a ?? 0;
                              if (actVal > 0.5) nodeFill = '#4f46e5';
                              else if (actVal > 0.1) nodeFill = '#312e81';
                            } else {
                              const gradVal = Math.abs(node.gradient_z ?? 0);
                              if (gradVal > 0.2) nodeFill = '#be185d';
                              else if (gradVal > 0.05) nodeFill = '#831843';
                            }

                            return (
                              <g
                                key={node.id}
                                transform={`translate(${coord.x}, ${coord.y})`}
                                className="cursor-pointer select-none"
                                onClick={() => {
                                  setSelectedNode(node);
                                  setSelectedSynapse(null);
                                }}
                              >
                                <circle
                                  r={isSelected ? 19 : 16}
                                  fill={nodeFill}
                                  stroke={isSelected ? '#38bdf8' : isDead ? '#ef4444' : '#a5b4fc'}
                                  strokeWidth={isSelected ? 2.5 : 1.5}
                                  strokeDasharray={isDead ? '3,3' : 'none'}
                                  className="transition-all hover:r-20"
                                />
                                <text
                                  textAnchor="middle"
                                  dy=".3em"
                                  fill="#ffffff"
                                  fontSize="10"
                                  fontFamily="monospace"
                                  fontWeight="bold"
                                >
                                  {node.label}
                                </text>
                              </g>
                            );
                          })}
                        </svg>
                      </div>
                    )}

                    {/* Legend */}
                    <div className="flex flex-wrap items-center justify-between text-xs text-slate-500 pt-1">
                      <div className="flex items-center gap-3">
                        <span className="flex items-center gap-1">
                          <span className="w-2.5 h-2.5 rounded-full bg-indigo-600"></span> Pozitif Ağırlık (+)
                        </span>
                        <span className="flex items-center gap-1">
                          <span className="w-2.5 h-2.5 rounded-full bg-rose-500"></span> Negatif Ağırlık (-)
                        </span>
                        <span className="flex items-center gap-1">
                          <span className="w-2.5 h-2.5 rounded-full bg-red-600 border border-red-300"></span> Ölü ReLU
                        </span>
                      </div>
                      <span className="font-mono text-[11px] text-slate-400">
                        {mlpData.activation_derivative_formula}
                      </span>
                    </div>
                  </div>

                  {/* Right: Selected Node / Synapse Detail Card */}
                  <div className="lg:col-span-4 space-y-4">
                    <div className="bg-white p-5 rounded-2xl shadow-sm border border-slate-200 space-y-3">
                      <h3 className="font-bold text-slate-900 text-sm flex items-center gap-2 border-b border-slate-100 pb-2">
                        <Eye className="w-4 h-4 text-indigo-600" />
                        Teşhis &amp; Eleman Detay İnceleyici
                      </h3>

                      {selectedNode ? (
                        <div className="space-y-3 text-xs">
                          <div className="p-3 bg-indigo-50/50 rounded-xl border border-indigo-200 space-y-1">
                            <span className="text-[10px] uppercase font-bold text-indigo-800 tracking-wider">
                              Seçili Nöron
                            </span>
                            <h4 className="font-mono font-bold text-sm text-slate-900">{selectedNode.label}</h4>
                            <span className="text-[11px] text-slate-600 capitalize">
                              Katman: {selectedNode.layer_type} (Katman {selectedNode.layer_idx})
                            </span>
                          </div>

                          <div className="space-y-2">
                            <div className="flex justify-between p-2 rounded-lg bg-slate-50 border border-slate-200">
                              <span className="text-slate-500">Pre-Activation (z):</span>
                              <span className="font-mono font-bold text-slate-800">
                                {selectedNode.pre_activation_z !== null ? selectedNode.pre_activation_z : '-'}
                              </span>
                            </div>

                            <div className="flex justify-between p-2 rounded-lg bg-slate-50 border border-slate-200">
                              <span className="text-slate-500">Post-Activation (a = σ(z)):</span>
                              <span className="font-mono font-bold text-indigo-700">
                                {selectedNode.post_activation_a !== null ? selectedNode.post_activation_a : '-'}
                              </span>
                            </div>

                            <div className="flex justify-between p-2 rounded-lg bg-slate-50 border border-slate-200">
                              <span className="text-slate-500">Gradiyen (∂L / ∂z):</span>
                              <span className="font-mono font-bold text-rose-600">
                                {selectedNode.gradient_z !== null ? selectedNode.gradient_z : '-'}
                              </span>
                            </div>

                            <div className="flex justify-between p-2 rounded-lg bg-slate-50 border border-slate-200">
                              <span className="text-slate-500">Durum:</span>
                              <span
                                className={`font-semibold ${
                                  selectedNode.is_dead ? 'text-red-600 font-bold' : 'text-emerald-600'
                                }`}
                              >
                                {selectedNode.is_dead ? 'Ölü ReLU (Sıfır Gradiyen)' : 'Aktif'}
                              </span>
                            </div>
                          </div>
                        </div>
                      ) : selectedSynapse ? (
                        <div className="space-y-3 text-xs">
                          <div className="p-3 bg-indigo-50/50 rounded-xl border border-indigo-200 space-y-1">
                            <span className="text-[10px] uppercase font-bold text-indigo-800 tracking-wider">
                              Seçili Sinaps (Ağırlık)
                            </span>
                            <h4 className="font-mono font-bold text-xs text-slate-900">
                              {selectedSynapse.source_id} ➔ {selectedSynapse.target_id}
                            </h4>
                          </div>

                          <div className="space-y-2">
                            <div className="flex justify-between p-2 rounded-lg bg-slate-50 border border-slate-200">
                              <span className="text-slate-500">Ağırlık Değeri (W):</span>
                              <span className="font-mono font-bold text-slate-800">{selectedSynapse.weight}</span>
                            </div>

                            <div className="flex justify-between p-2 rounded-lg bg-slate-50 border border-slate-200">
                              <span className="text-slate-500">Gradiyen (∂L / ∂W):</span>
                              <span className="font-mono font-bold text-rose-600">{selectedSynapse.gradient_w}</span>
                            </div>

                            <div className="flex justify-between p-2 rounded-lg bg-emerald-50/60 border border-emerald-200">
                              <span className="text-emerald-800 font-semibold">Güncelleme (-η · ∂L/∂W):</span>
                              <span className="font-mono font-bold text-emerald-900">
                                {selectedSynapse.weight_update}
                              </span>
                            </div>
                          </div>
                        </div>
                      ) : (
                        <div className="p-6 text-center text-slate-400 space-y-2">
                          <Info className="w-6 h-6 mx-auto text-slate-300" />
                          <p className="text-xs">
                            İncelemek için hesaplama grafiğindeki herhangi bir nörona veya ağırlık çizgisine tıklayın.
                          </p>
                        </div>
                      )}
                    </div>

                    {/* Layer Health Table */}
                    <div className="bg-white p-5 rounded-2xl shadow-sm border border-slate-200 space-y-3">
                      <h3 className="font-bold text-slate-900 text-sm flex items-center gap-2 border-b border-slate-100 pb-2">
                        <Activity className="w-4 h-4 text-emerald-600" />
                        Katman Bazlı Gradiyen Büyüklükleri
                      </h3>

                      <div className="space-y-2 text-xs">
                        {mlpData.layer_health.map((lh) => (
                          <div
                            key={lh.layer_idx}
                            className="flex items-center justify-between p-2.5 rounded-xl bg-slate-50 border border-slate-200"
                          >
                            <span className="font-mono font-semibold text-slate-700">Katman {lh.layer_idx}</span>
                            <div className="flex items-center gap-2">
                              <span className="font-mono text-slate-500 text-[11px]">Maks: {lh.max_grad}</span>
                              <span
                                className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                                  lh.status === 'HEALTHY'
                                    ? 'bg-emerald-100 text-emerald-800'
                                    : 'bg-rose-100 text-rose-800'
                                }`}
                              >
                                {lh.status}
                              </span>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>

                {/* Step-by-Step Chain Rule Explanations */}
                <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200 space-y-4">
                  <div className="flex items-center justify-between border-b border-slate-100 pb-3">
                    <h3 className="text-base font-bold text-slate-900 flex items-center gap-2">
                      <Sparkles className="w-5 h-5 text-indigo-600" />
                      Adım Adım Geriye Yayılım &amp; Zincir Kuralı (Chain Rule Derivations)
                    </h3>
                    <span className="text-xs font-mono text-slate-400">Analitik Autograd Türevleri</span>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {mlpData.chain_rule_steps.map((step) => (
                      <div
                        key={step.step}
                        className="p-4 rounded-xl bg-slate-50/80 border border-slate-200 text-xs space-y-2"
                      >
                        <div className="flex items-center justify-between">
                          <span className="font-bold text-slate-900 font-mono">Adım {step.step}</span>
                          <span className="px-2 py-0.5 rounded bg-indigo-50 text-indigo-700 font-semibold text-[11px]">
                            {step.layer}
                          </span>
                        </div>

                        <div className="p-2.5 bg-white rounded-lg border border-slate-200 font-mono text-indigo-900 text-center font-bold">
                          {step.formula}
                        </div>

                        <p className="text-slate-600 leading-relaxed text-[11px]">{step.description}</p>

                        {step.grad_norm !== undefined && (
                          <div className="flex justify-between pt-1 border-t border-slate-200/80 text-[11px] font-mono">
                            <span className="text-slate-400">L2 Gradiyen Normu:</span>
                            <span className="font-bold text-slate-800">{step.grad_norm}</span>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* ================================================================== */}
        {/* TAB 2: OPTIMIZER RACE ON 2D LOSS LANDSCAPE                          */}
        {/* ================================================================== */}
        {activeTab === 'race' && (
          <div className="space-y-6">
            {/* Control & Hyperparameter Bar */}
            <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200 space-y-4">
              <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 border-b border-slate-100 pb-3">
                <div>
                  <h2 className="text-base font-bold text-slate-900 flex items-center gap-2">
                    <Trophy className="w-5 h-5 text-amber-500" />
                    2D Kayıp Yüzeyi &amp; Çoklu Optimizer Yarışı
                  </h2>
                  <p className="text-xs text-slate-500 mt-0.5">
                    AdamW, Adam, RMSprop, Momentum ve SGD algoritmalarını zorlu analitik yüzeylerde aynı anda yarıştırın.
                  </p>
                </div>

                <button
                  id="btn-start-race"
                  onClick={runOptimizerRace}
                  disabled={raceLoading}
                  className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-amber-600 hover:bg-amber-500 text-white font-semibold text-xs transition shadow-md shadow-amber-600/20 disabled:opacity-50"
                >
                  {raceLoading ? (
                    <>
                      <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                      Yarış Simüle Ediliyor...
                    </>
                  ) : (
                    <>
                      <Play className="w-3.5 h-3.5" />
                      Optimizer Yarışını Başlat
                    </>
                  )}
                </button>
              </div>

              {/* Landscape & Optimizer Selector */}
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4 pt-1">
                <div>
                  <label htmlFor="select-landscape" className="block text-xs font-bold text-slate-700 mb-1">
                    Kayıp Yüzeyi (Landscape)
                  </label>
                  <select
                    id="select-landscape"
                    value={selectedLandscape}
                    onChange={(e) => setSelectedLandscape(e.target.value)}
                    className="w-full p-2.5 rounded-xl border border-slate-200 text-xs bg-white font-medium"
                  >
                    <option value="quadratic_bowl">Kötü Şartlanmış Çanak (Ill-conditioned)</option>
                    <option value="saddle">Eyer Noktası (Saddle Point)</option>
                    <option value="rosenbrock">Rosenbrock (Muz Vadisi)</option>
                    <option value="beale">Beale Fonksiyonu</option>
                  </select>
                </div>

                <div>
                  <label htmlFor="input-race-lr" className="block text-xs font-bold text-slate-700 mb-1">
                    Öğrenme Oranı (η): {raceLr}
                  </label>
                  <input
                    id="input-race-lr"
                    type="range"
                    min="0.005"
                    max="0.1"
                    step="0.005"
                    value={raceLr}
                    onChange={(e) => setRaceLr(parseFloat(e.target.value))}
                    className="w-full accent-amber-600 mt-2"
                  />
                </div>

                <div>
                  <label htmlFor="input-race-decay" className="block text-xs font-bold text-slate-700 mb-1">
                    Weight Decay (AdamW farkı): {raceWeightDecay}
                  </label>
                  <input
                    id="input-race-decay"
                    type="range"
                    min="0.0"
                    max="0.1"
                    step="0.01"
                    value={raceWeightDecay}
                    onChange={(e) => setRaceWeightDecay(parseFloat(e.target.value))}
                    className="w-full accent-amber-600 mt-2"
                  />
                </div>

                <div>
                  <label htmlFor="input-race-steps" className="block text-xs font-bold text-slate-700 mb-1">
                    Adım Sayısı: {raceSteps}
                  </label>
                  <input
                    id="input-race-steps"
                    type="range"
                    min="20"
                    max="100"
                    step="5"
                    value={raceSteps}
                    onChange={(e) => setRaceSteps(parseInt(e.target.value, 10))}
                    className="w-full accent-amber-600 mt-2"
                  />
                </div>
              </div>

              {/* Optimizer Chips */}
              <div className="flex flex-wrap items-center gap-2 pt-2 border-t border-slate-100">
                <span className="text-xs text-slate-500 font-medium">Yarışacak Algoritmalar:</span>
                {landscapesMeta?.optimizers &&
                  Object.entries(landscapesMeta.optimizers).map(([key, opt]) => {
                    const isSelected = selectedOptimizers.includes(key);
                    return (
                      <button
                        key={key}
                        type="button"
                        onClick={() => toggleOptimizer(key)}
                        className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-xl text-xs font-semibold border transition ${
                          isSelected
                            ? 'bg-slate-900 text-white border-slate-900 shadow-sm'
                            : 'bg-slate-100 text-slate-400 border-slate-200 line-through'
                        }`}
                      >
                        <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: opt.color }}></span>
                        <span>{opt.name}</span>
                      </button>
                    );
                  })}
              </div>
            </div>

            {/* Error */}
            {raceError && (
              <div className="p-4 rounded-xl bg-red-50 border border-red-200 text-red-700 flex items-center gap-3 text-sm">
                <AlertTriangle className="w-5 h-5 flex-shrink-0" />
                <span>{raceError}</span>
              </div>
            )}

            {/* Race Results & Canvas */}
            {raceData && (
              <div className="space-y-6">
                {/* Winner Banner */}
                <div className="p-5 rounded-2xl bg-gradient-to-r from-amber-500/10 via-amber-500/20 to-indigo-500/10 border border-amber-200 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
                  <div className="flex items-center gap-4">
                    <div className="w-12 h-12 rounded-2xl bg-amber-500 text-white flex items-center justify-center text-2xl shadow-md">
                      🏆
                    </div>
                    <div>
                      <span className="text-[11px] font-bold text-amber-900 uppercase tracking-wider block">
                        Yarış Kazananı (Fastest Convergence)
                      </span>
                      <h3 className="text-xl font-extrabold text-slate-900 font-mono">{raceData.winner_name}</h3>
                      <p className="text-xs text-slate-600 mt-0.5">{raceData.landscape.description}</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-xs bg-amber-100 text-amber-900 font-semibold px-3 py-1.5 rounded-xl border border-amber-300 font-mono">
                      Formül: {raceData.landscape.formula}
                    </span>
                  </div>
                </div>

                {/* 2D Contour Map & Leaderboard Grid */}
                <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
                  {/* Left: SVG 2D Contour Visualizer */}
                  <div className="lg:col-span-8 bg-white p-6 rounded-2xl shadow-sm border border-slate-200 space-y-4 flex flex-col items-center">
                    <div className="w-full flex items-center justify-between border-b border-slate-100 pb-3">
                      <h3 className="font-bold text-slate-900 text-sm flex items-center gap-2">
                        <Activity className="w-4 h-4 text-indigo-600" />
                        2D Kayıp Manzarası &amp; Yörünge İzleri (Trajectories)
                      </h3>
                      <span className="text-xs font-mono text-slate-400">
                        Adım {currentStep} / {raceSteps}
                      </span>
                    </div>

                    {/* SVG 2D Contour Canvas */}
                    <div className="relative w-full max-w-[540px] aspect-square bg-slate-950 rounded-2xl border border-slate-800 p-2 overflow-hidden shadow-inner flex items-center justify-center">
                      <svg viewBox="0 0 500 500" className="w-full h-full">
                        {/* Contour Grid Rings (concentric SVG circles/ellipses scaled to range) */}
                        {[0.2, 0.4, 0.6, 0.8, 1.0].map((lvl, idx) => (
                          <ellipse
                            key={`c-ring-${idx}`}
                            cx="250"
                            cy="250"
                            rx={lvl * 210}
                            ry={lvl * 150}
                            fill="none"
                            stroke="#334155"
                            strokeWidth="1"
                            strokeDasharray="4 4"
                          />
                        ))}

                        {/* Axes */}
                        <line x1="25" y1="250" x2="475" y2="250" stroke="#475569" strokeWidth="1" strokeDasharray="2 2" />
                        <line x1="250" y1="25" x2="250" y2="475" stroke="#475569" strokeWidth="1" strokeDasharray="2 2" />

                        {/* Optimum marker (Target) */}
                        <g transform="translate(250, 250)">
                          <circle r="6" fill="#10b981" stroke="#ffffff" strokeWidth="2" />
                          <text x="8" y="4" fill="#10b981" fontSize="10" fontFamily="monospace" fontWeight="bold">
                            Optimum (0, 0)
                          </text>
                        </g>

                        {/* Optimizer Trajectories */}
                        {Object.entries(raceData.optimizers).map(([optKey, optData]) => {
                          const xMin = raceData.contours.x_range[0];
                          const xMax = raceData.contours.x_range[1];
                          const yMin = raceData.contours.y_range[0];
                          const yMax = raceData.contours.y_range[1];

                          // Map x, y to SVG coords (25 to 475)
                          const mapX = (x: number) => 25 + ((x - xMin) / (xMax - xMin)) * 450;
                          const mapY = (y: number) => 475 - ((y - yMin) / (yMax - yMin)) * 450;

                          const visiblePoints = optData.trajectory.slice(0, currentStep + 1);
                          if (visiblePoints.length === 0) return null;

                          const pathStr = visiblePoints
                            .map((p, idx) => `${idx === 0 ? 'M' : 'L'} ${mapX(p.x).toFixed(1)} ${mapY(p.y).toFixed(1)}`)
                            .join(' ');

                          const curP = visiblePoints[visiblePoints.length - 1];

                          return (
                            <g key={`traj-${optKey}`}>
                              {/* Path line */}
                              <path
                                d={pathStr}
                                fill="none"
                                stroke={optData.color}
                                strokeWidth="2.5"
                                strokeLinecap="round"
                                strokeLinejoin="round"
                                className="transition-all"
                              />

                              {/* Trajectory dots */}
                              {visiblePoints.map((p, idx) => (
                                <circle
                                  key={`pt-${optKey}-${idx}`}
                                  cx={mapX(p.x)}
                                  cy={mapY(p.y)}
                                  r={idx === visiblePoints.length - 1 ? 5 : 2}
                                  fill={optData.color}
                                  stroke="#ffffff"
                                  strokeWidth={idx === visiblePoints.length - 1 ? 2 : 0}
                                />
                              ))}

                              {/* Head tag */}
                              {curP && (
                                <g transform={`translate(${mapX(curP.x)}, ${mapY(curP.y) - 8})`}>
                                  <text
                                    textAnchor="middle"
                                    fill={optData.color}
                                    fontSize="9"
                                    fontWeight="bold"
                                    fontFamily="monospace"
                                  >
                                    {optData.name.split(' ')[0]}
                                  </text>
                                </g>
                              )}
                            </g>
                          );
                        })}
                      </svg>
                    </div>

                    {/* Animation Controls & Scrubber */}
                    <div className="w-full pt-2 flex flex-col sm:flex-row items-center justify-between gap-4 border-t border-slate-100">
                      <div className="flex items-center gap-2">
                        <button
                          type="button"
                          onClick={() => {
                            if (currentStep >= raceSteps) setCurrentStep(0);
                            setIsPlaying(!isPlaying);
                          }}
                          className="p-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white transition shadow-sm"
                        >
                          {isPlaying ? <RotateCcw className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                        </button>
                        <button
                          type="button"
                          onClick={() => {
                            setIsPlaying(false);
                            setCurrentStep(0);
                          }}
                          className="px-3 py-1.5 rounded-xl bg-slate-100 hover:bg-slate-200 text-xs font-semibold text-slate-700 transition"
                        >
                          Sıfırla
                        </button>
                        <span className="text-xs font-mono text-slate-600">Adım: {currentStep}</span>
                      </div>

                      {/* Step Slider */}
                      <div className="flex items-center gap-2 w-full sm:w-64">
                        <input
                          type="range"
                          min="0"
                          max={raceSteps}
                          value={currentStep}
                          onChange={(e) => {
                            setIsPlaying(false);
                            setCurrentStep(parseInt(e.target.value, 10));
                          }}
                          className="w-full accent-indigo-600"
                        />
                      </div>
                    </div>
                  </div>

                  {/* Right: Leaderboard & Metrics Breakdown */}
                  <div className="lg:col-span-4 space-y-4">
                    <div className="bg-white p-5 rounded-2xl shadow-sm border border-slate-200 space-y-3">
                      <h3 className="font-bold text-slate-900 text-sm flex items-center gap-2 border-b border-slate-100 pb-2">
                        <Trophy className="w-4 h-4 text-amber-500" />
                        Optimizer Skor Tablosu
                      </h3>

                      <div className="space-y-3 text-xs">
                        {Object.entries(raceData.optimizers).map(([optKey, opt]) => {
                          const isWinner = optKey === raceData.winner;
                          return (
                            <div
                              key={optKey}
                              className={`p-3 rounded-xl border transition ${
                                isWinner
                                  ? 'bg-amber-50/50 border-amber-300 shadow-sm ring-2 ring-amber-400/20'
                                  : 'bg-slate-50/70 border-slate-200'
                              }`}
                            >
                              <div className="flex items-center justify-between mb-1.5">
                                <span className="font-bold text-slate-900 flex items-center gap-1.5">
                                  <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: opt.color }} />
                                  {opt.name}
                                </span>
                                {isWinner && (
                                  <span className="px-2 py-0.5 rounded-full bg-amber-100 text-amber-900 font-bold text-[10px] border border-amber-300">
                                    ★ KAZANAN
                                  </span>
                                )}
                              </div>

                              <div className="grid grid-cols-2 gap-2 text-[11px] pt-1">
                                <div>
                                  <span className="text-slate-400 block">Final Kayıp:</span>
                                  <span className="font-mono font-bold text-slate-900">{opt.final_loss}</span>
                                </div>
                                <div>
                                  <span className="text-slate-400 block">Kayıp Azalması:</span>
                                  <span className="font-mono font-bold text-emerald-700">
                                    %{opt.loss_reduction_pct}
                                  </span>
                                </div>
                                <div>
                                  <span className="text-slate-400 block">Optimum Mesafesi:</span>
                                  <span className="font-mono font-semibold text-slate-700">
                                    {opt.distance_to_optimum}
                                  </span>
                                </div>
                                <div>
                                  <span className="text-slate-400 block">Toplam Yol:</span>
                                  <span className="font-mono font-semibold text-slate-700">{opt.total_distance}</span>
                                </div>
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* ================================================================== */}
        {/* TAB 3: ACTIVATION DYNAMICS & SATURATION EXPLORER                   */}
        {/* ================================================================== */}
        {activeTab === 'activations' && (
          <div className="space-y-6">
            <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200 space-y-4">
              <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 border-b border-slate-100 pb-3">
                <div>
                  <h2 className="text-base font-bold text-slate-900 flex items-center gap-2">
                    <Activity className="w-5 h-5 text-emerald-600" />
                    Aktivasyon Fonksiyonları &amp; Doygunluk (Saturation) Analizi
                  </h2>
                  <p className="text-xs text-slate-500 mt-0.5">
                    Aktivasyon fonksiyonlarının f(x) ve türev f'(x) eğrilerini inceleyerek kaybolan gradiyen (vanishing
                    gradient) bölgelerini analiz edin.
                  </p>
                </div>

                {/* Probe slider */}
                <div className="flex items-center gap-3">
                  <label htmlFor="input-probe-x" className="text-xs font-bold text-slate-700 whitespace-nowrap">
                    Test Noktası (x): {probeX}
                  </label>
                  <input
                    id="input-probe-x"
                    type="range"
                    min="-3.5"
                    max="3.5"
                    step="0.1"
                    value={probeX}
                    onChange={(e) => setProbeX(parseFloat(e.target.value))}
                    className="w-36 accent-emerald-600"
                  />
                </div>
              </div>

              {/* Grid of Activations */}
              {actData?.activations && (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 pt-2">
                  {Object.entries(actData.activations).map(([key, act]) => {
                    // Approximate probe value from curve data
                    const xIdx = Math.max(
                      0,
                      Math.min(
                        actData.x_vals.length - 1,
                        Math.round(((probeX + 4.0) / 8.0) * (actData.x_vals.length - 1))
                      )
                    );
                    const curFx = act.fx[xIdx];
                    const curDfx = act.dfx[xIdx];
                    const isVanishing = Math.abs(curDfx) < 0.05;

                    return (
                      <div
                        key={key}
                        className="p-5 rounded-2xl bg-white border border-slate-200 shadow-sm space-y-3 flex flex-col justify-between"
                      >
                        <div className="space-y-1">
                          <div className="flex items-center justify-between">
                            <span className="font-bold text-slate-900 font-mono text-sm">{act.name}</span>
                            <span
                              className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                                isVanishing
                                  ? 'bg-rose-100 text-rose-800 border border-rose-300'
                                  : 'bg-emerald-100 text-emerald-800 border border-emerald-300'
                              }`}
                            >
                              {isVanishing ? 'Gradiyen Doyumu!' : 'Aktif Gradiyen'}
                            </span>
                          </div>
                          <span className="text-[11px] font-mono text-slate-400 block">{act.derivative_formula}</span>
                        </div>

                        {/* Mini SVG Plot of f(x) and df(x) */}
                        <div className="h-32 w-full bg-slate-950 rounded-xl p-2 relative overflow-hidden">
                          <svg viewBox="0 0 200 100" className="w-full h-full">
                            {/* Zero line */}
                            <line x1="0" y1="50" x2="200" y2="50" stroke="#334155" strokeWidth="1" strokeDasharray="2 2" />

                            {/* f(x) curve - Indigo */}
                            <path
                              d={act.fx
                                .map((v, i) => {
                                  const px = (i / (act.fx.length - 1)) * 200;
                                  const py = 50 - v * 15;
                                  return `${i === 0 ? 'M' : 'L'} ${px.toFixed(1)} ${py.toFixed(1)}`;
                                })
                                .join(' ')}
                              fill="none"
                              stroke="#818cf8"
                              strokeWidth="2"
                            />

                            {/* df(x) curve - Emerald */}
                            <path
                              d={act.dfx
                                .map((v, i) => {
                                  const px = (i / (act.dfx.length - 1)) * 200;
                                  const py = 50 - v * 25;
                                  return `${i === 0 ? 'M' : 'L'} ${px.toFixed(1)} ${py.toFixed(1)}`;
                                })
                                .join(' ')}
                              fill="none"
                              stroke="#34d399"
                              strokeWidth="2"
                              strokeDasharray="2 2"
                            />
                          </svg>

                          <div className="absolute top-1.5 right-2 flex items-center gap-2 text-[9px] font-mono">
                            <span className="text-indigo-400">― f(x)</span>
                            <span className="text-emerald-400">┄ f'(x)</span>
                          </div>
                        </div>

                        {/* Probe values at current x */}
                        <div className="p-2.5 bg-slate-50 rounded-xl border border-slate-200 text-xs space-y-1">
                          <div className="flex justify-between font-mono">
                            <span className="text-slate-500">f({probeX}):</span>
                            <span className="font-bold text-indigo-700">{curFx}</span>
                          </div>
                          <div className="flex justify-between font-mono">
                            <span className="text-slate-500">Türev f'({probeX}):</span>
                            <span className={`font-bold ${isVanishing ? 'text-rose-600' : 'text-emerald-700'}`}>
                              {curDfx}
                            </span>
                          </div>
                        </div>

                        {/* Vanishing notes */}
                        {act.vanishing_notes && act.vanishing_notes.length > 0 && (
                          <div className="text-[10px] text-slate-500 space-y-0.5 border-t border-slate-100 pt-2">
                            <span className="font-bold text-slate-600 block">Doygunluk Bölgeleri:</span>
                            {act.vanishing_notes.map((note, nIdx) => (
                              <div key={nIdx} className="font-mono text-rose-700">
                                • {note}
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
