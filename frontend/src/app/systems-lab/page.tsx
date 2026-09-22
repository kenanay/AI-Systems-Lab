'use client';

import React, { useState, useEffect, useMemo } from 'react';
import Link from 'next/link';
import {
  Cpu,
  Zap,
  HardDrive,
  BarChart3,
  Sliders,
  Play,
  RefreshCw,
  AlertTriangle,
  CheckCircle2,
  AlertCircle,
  Info,
  Layers,
  Sparkles,
  ArrowRight,
  TrendingUp,
  Activity,
  Server,
  HelpCircle,
} from 'lucide-react';
import {
  api,
  HardwarePreset,
  RooflineAnalyzeResponse,
  QuantizationSimulateResponse,
  GPUMemorySimulateResponse,
} from '@/lib/api';

// Predefined Workloads for Roofline Tab
const PREDEFINED_WORKLOADS = [
  {
    name: 'Autoregressive LLM Token Generation',
    intensity: 1.0,
    flops: 2_000_000_000,
    bytes: 2_000_000_000,
    desc: 'Tekil token üretimi; her parametre için 2 byte okunup 2 FLOPs yapılır. Şiddetli derecede Memory-Bound!',
  },
  {
    name: 'Transformer Multi-Head Attention Layer',
    intensity: 42.0,
    flops: 420_000_000_000,
    bytes: 10_000_000_000,
    desc: 'Q*K^T ve Softmax*V matris çarpımları. Donanım bant genişliğine bağlı dengeli/orta yoğunluk bölgesi.',
  },
  {
    name: 'GEMM Dense Matrix Multiply (Compute-Bound)',
    intensity: 250.0,
    flops: 2_500_000_000_000,
    bytes: 10_000_000_000,
    desc: 'Büyük batch veya FFN katmanı matris çarpımı. Yüksek veri yeniden kullanımı sayesinde Tensor Core compute tavanına yaklaşır.',
  },
  {
    name: 'RMSNorm & SwiGLU Activation Elementwise',
    intensity: 0.25,
    flops: 500_000_000,
    bytes: 2_000_000_000,
    desc: 'Eleman bazlı normalizasyon ve aktivasyon. Düşük FLOPs, yüksek bellek aktarımı; tipik Kernel Fusion adayı.',
  },
  {
    name: 'Özel İş Yükü (Custom)',
    intensity: 50.0,
    flops: 500_000_000_000,
    bytes: 10_000_000_000,
    desc: 'Kendi FLOPs ve bellek bayt değerlerinizi tanımlayın.',
  },
];

// Fallback presets if offline
const FALLBACK_PRESETS: HardwarePreset[] = [
  {
    name: 'NVIDIA H100 SXM5',
    category: 'Datacenter GPU',
    peak_tflops: 1000.0,
    peak_bandwidth_gbs: 3350.0,
    vram_gb: 80.0,
    description: 'Hopper mimarisi, 80GB HBM3 bellek ve 4. nesil Tensor Çekirdekleri.',
  },
  {
    name: 'NVIDIA A100 SXM4',
    category: 'Datacenter GPU',
    peak_tflops: 312.0,
    peak_bandwidth_gbs: 2039.0,
    vram_gb: 80.0,
    description: 'Ampere mimarisi, 80GB HBM2e ve 3. nesil Tensor Çekirdekleri.',
  },
  {
    name: 'NVIDIA RTX 4090',
    category: 'Consumer GPU',
    peak_tflops: 330.0,
    peak_bandwidth_gbs: 1008.0,
    vram_gb: 24.0,
    description: 'Ada Lovelace mimarisi, 24GB GDDR6X, yüksek FP16/FP8 tensör gücü.',
  },
  {
    name: 'Apple M3 Max (16-core)',
    category: 'Workstation SoC',
    peak_tflops: 32.0,
    peak_bandwidth_gbs: 400.0,
    vram_gb: 128.0,
    description: 'Birleşik bellek (Unified Memory) mimarisi, düşük gecikmeli çıkarım.',
  },
  {
    name: 'Intel Xeon Platinum 8480+ (AVX-512)',
    category: 'Server CPU',
    peak_tflops: 7.0,
    peak_bandwidth_gbs: 307.0,
    vram_gb: 512.0,
    description: 'Sapphire Rapids sunucu işlemcisi, DDR5 8-kanal bellek kontrolcüsü.',
  },
];

export default function SystemsLabPage() {
  const [activeTab, setActiveTab] = useState<'roofline' | 'quantization' | 'vram'>('roofline');

  // --------------------------------------------------------------------------
  // Tab 1: Roofline Model State
  // --------------------------------------------------------------------------
  const [presets, setPresets] = useState<HardwarePreset[]>(FALLBACK_PRESETS);
  const [selectedPresetName, setSelectedPresetName] = useState<string>('NVIDIA H100 SXM5');
  const [customPeakTflops, setCustomPeakTflops] = useState<number>(1000);
  const [customPeakBandwidth, setCustomPeakBandwidth] = useState<number>(3350);
  const [selectedWorkloadIdx, setSelectedWorkloadIdx] = useState<number>(0);
  const [customIntensity, setCustomIntensity] = useState<number>(1.0);
  const [rooflineLoading, setRooflineLoading] = useState<boolean>(false);
  const [rooflineData, setRooflineData] = useState<RooflineAnalyzeResponse | null>(null);

  // --------------------------------------------------------------------------
  // Tab 2: Quantization Simulator State
  // --------------------------------------------------------------------------
  const [distType, setDistType] = useState<'normal' | 'uniform' | 'outlier'>('outlier');
  const [numElements, setNumElements] = useState<number>(2000);
  const [outlierRatio, setOutlierRatio] = useState<number>(0.01);
  const [outlierMag, setOutlierMag] = useState<number>(6.0);
  const [quantLoading, setQuantLoading] = useState<boolean>(false);
  const [quantData, setQuantData] = useState<QuantizationSimulateResponse | null>(null);
  const [selectedHistView, setSelectedHistView] = useState<'original' | 'int8' | 'int4'>('original');

  // --------------------------------------------------------------------------
  // Tab 3: GPU VRAM State
  // --------------------------------------------------------------------------
  const [vramMode, setVramMode] = useState<'training' | 'inference'>('training');
  const [modelParamB, setModelParamB] = useState<number>(8.0);
  const [precision, setPrecision] = useState<'fp32' | 'fp16' | 'bf16' | 'int8' | 'int4'>('fp16');
  const [batchSize, setBatchSize] = useState<number>(4);
  const [seqLen, setSeqLen] = useState<number>(2048);
  const [optimizer, setOptimizer] = useState<'adamw' | 'adamw_8bit' | 'sgd'>('adamw');
  const [activationCheckpointing, setActivationCheckpointing] = useState<boolean>(true);
  const [gpuVramCap, setGpuVramCap] = useState<number>(80);
  const [vramLoading, setVramLoading] = useState<boolean>(false);
  const [vramData, setVramData] = useState<GPUMemorySimulateResponse | null>(null);

  // Load Presets on Mount
  useEffect(() => {
    let isMounted = true;
    const fetchPresets = async () => {
      try {
        const data = await api.systemsLab.getPresets();
        if (isMounted && data && data.length > 0) {
          setPresets(data);
        }
      } catch (err) {
        console.warn('Could not load online hardware presets, using defaults.', err);
      }
    };
    fetchPresets();
    return () => {
      isMounted = false;
    };
  }, []);

  // Update hardware sliders when preset selection changes
  const currentPreset = useMemo(() => {
    return presets.find((p) => p.name === selectedPresetName) || presets[0] || FALLBACK_PRESETS[0];
  }, [presets, selectedPresetName]);

  useEffect(() => {
    if (currentPreset) {
      setCustomPeakTflops(currentPreset.peak_tflops);
      setCustomPeakBandwidth(currentPreset.peak_bandwidth_gbs);
    }
  }, [currentPreset]);

  // Run Roofline Analysis
  const runRooflineAnalysis = async () => {
    setRooflineLoading(true);
    try {
      const activeWorkload = PREDEFINED_WORKLOADS[selectedWorkloadIdx];
      const intensity =
        selectedWorkloadIdx === PREDEFINED_WORKLOADS.length - 1
          ? customIntensity
          : activeWorkload.intensity;

      const res = await api.systemsLab.analyzeRoofline({
        hardware_name: currentPreset.name,
        peak_tflops: customPeakTflops,
        peak_bandwidth_gbs: customPeakBandwidth,
        workload_name: activeWorkload.name,
        custom_operational_intensity: intensity,
      });
      setRooflineData(res);
    } catch (err) {
      console.error('Roofline analysis failed', err);
    } finally {
      setRooflineLoading(false);
    }
  };

  // Run Quantization Simulation
  const runQuantSimulation = async () => {
    setQuantLoading(true);
    try {
      const res = await api.systemsLab.simulateQuantization({
        distribution_type: distType,
        num_elements: numElements,
        outlier_ratio: outlierRatio,
        outlier_magnitude: outlierMag,
      });
      setQuantData(res);
    } catch (err) {
      console.error('Quantization simulation failed', err);
    } finally {
      setQuantLoading(false);
    }
  };

  // Run VRAM Simulation
  const runVramSimulation = async () => {
    setVramLoading(true);
    try {
      const res = await api.systemsLab.simulateMemory({
        param_count_billions: modelParamB,
        precision,
        mode: vramMode,
        batch_size: batchSize,
        sequence_length: seqLen,
        optimizer,
        activation_checkpointing: activationCheckpointing,
        gpu_vram_gb: gpuVramCap,
      });
      setVramData(res);
    } catch (err) {
      console.error('VRAM simulation failed', err);
    } finally {
      setVramLoading(false);
    }
  };

  // Auto-run initial simulations on mount
  useEffect(() => {
    runRooflineAnalysis();
    runQuantSimulation();
    runVramSimulation();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Recalculate Roofline when inputs change
  useEffect(() => {
    runRooflineAnalysis();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedPresetName, customPeakTflops, customPeakBandwidth, selectedWorkloadIdx, customIntensity]);

  // Recalculate VRAM when inputs change
  useEffect(() => {
    runVramSimulation();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [vramMode, modelParamB, precision, batchSize, seqLen, optimizer, activationCheckpointing, gpuVramCap]);

  // Knee point calculation
  const kneePointIntensity = useMemo(() => {
    if (customPeakBandwidth <= 0) return 0;
    return (customPeakTflops * 1000) / customPeakBandwidth;
  }, [customPeakTflops, customPeakBandwidth]);

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 pb-16">
      {/* Header */}
      <header className="border-b border-slate-200 bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
            <div>
              <div className="flex items-center gap-2 mb-1">
                <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold bg-amber-100 text-amber-800">
                  <Zap className="w-3.5 h-3.5 mr-1 text-amber-600" /> SYSTEMS FOR AI & HARDWARE
                </span>
                <span className="text-xs text-slate-500 font-mono">v1.2.0 • Hardware Co-Design</span>
              </div>
              <h1 className="text-2xl sm:text-3xl font-bold tracking-tight text-slate-900">
                Roofline Modeli, Kuantizasyon & GPU VRAM Laboratuvarı
              </h1>
              <p className="text-sm text-slate-600 mt-1 max-w-3xl">
                Derin öğrenme modellerinin donanım üzerindeki darboğazlarını (Compute vs Memory-Bound) analiz edin, 
                sayısal hassasiyet (FP32/16, INT8/4) kaybını simüle edin ve GPU VRAM belleğini bileşenlerine ayırın.
              </p>
            </div>

            {/* Quick Action Badges */}
            <div className="flex items-center gap-2 self-start md:self-auto">
              <Link
                href="/transformer-lab"
                className="inline-flex items-center px-3 py-1.5 rounded-lg border border-slate-300 text-xs font-medium text-slate-700 hover:bg-slate-50 transition-colors"
              >
                Transformer Lab <ArrowRight className="w-3.5 h-3.5 ml-1" />
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
              id="tab-roofline-btn"
              onClick={() => setActiveTab('roofline')}
              className={`flex items-center gap-2 px-4 py-2.5 text-sm font-medium border-b-2 transition-all ${
                activeTab === 'roofline'
                  ? 'border-indigo-600 text-indigo-600 bg-indigo-50/50 rounded-t-lg'
                  : 'border-transparent text-slate-600 hover:text-slate-900 hover:border-slate-300'
              }`}
            >
              <TrendingUp className="w-4 h-4" />
              Roofline & Memory Wall
            </button>
            <button
              id="tab-quantization-btn"
              onClick={() => setActiveTab('quantization')}
              className={`flex items-center gap-2 px-4 py-2.5 text-sm font-medium border-b-2 transition-all ${
                activeTab === 'quantization'
                  ? 'border-indigo-600 text-indigo-600 bg-indigo-50/50 rounded-t-lg'
                  : 'border-transparent text-slate-600 hover:text-slate-900 hover:border-slate-300'
              }`}
            >
              <Activity className="w-4 h-4" />
              Kuantizasyon Simülatörü
            </button>
            <button
              id="tab-vram-btn"
              onClick={() => setActiveTab('vram')}
              className={`flex items-center gap-2 px-4 py-2.5 text-sm font-medium border-b-2 transition-all ${
                activeTab === 'vram'
                  ? 'border-indigo-600 text-indigo-600 bg-indigo-50/50 rounded-t-lg'
                  : 'border-transparent text-slate-600 hover:text-slate-900 hover:border-slate-300'
              }`}
            >
              <HardDrive className="w-4 h-4" />
              GPU VRAM & Bellek Ayrıştırması
            </button>
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 mt-6">
        {/* ================================================================= */}
        {/* TAB 1: ROOFLINE MODEL & MEMORY WALL                               */}
        {/* ================================================================= */}
        {activeTab === 'roofline' && (
          <div className="space-y-6">
            {/* Top Control Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Hardware Selection Card */}
              <div className="bg-white rounded-xl p-5 border border-slate-200 shadow-sm space-y-4">
                <div className="flex items-center justify-between">
                  <h3 className="text-sm font-semibold text-slate-900 flex items-center gap-2">
                    <Server className="w-4 h-4 text-indigo-600" /> Donanım Mimarisi Seçimi
                  </h3>
                  <span className="text-xs bg-slate-100 text-slate-700 px-2 py-0.5 rounded font-mono">
                    {currentPreset.category}
                  </span>
                </div>

                <div>
                  <label className="block text-xs font-medium text-slate-700 mb-1">
                    Preset Donanım
                  </label>
                  <select
                    id="hardware-preset-select"
                    value={selectedPresetName}
                    onChange={(e) => setSelectedPresetName(e.target.value)}
                    className="w-full bg-slate-50 border border-slate-300 rounded-lg px-3 py-2 text-sm text-slate-900 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                  >
                    {presets.map((preset) => (
                      <option key={preset.name} value={preset.name}>
                        {preset.name} ({preset.peak_tflops} TFLOPs / {preset.peak_bandwidth_gbs} GB/s)
                      </option>
                    ))}
                  </select>
                  <p className="text-xs text-slate-500 mt-1 italic">{currentPreset.description}</p>
                </div>

                <div className="border-t border-slate-100 pt-3 space-y-3">
                  <div>
                    <div className="flex justify-between text-xs text-slate-600 mb-1">
                      <span>Peak Compute (TFLOPs):</span>
                      <span className="font-semibold text-indigo-600 font-mono">{customPeakTflops} TFLOPs</span>
                    </div>
                    <input
                      type="range"
                      min={5}
                      max={2000}
                      step={5}
                      value={customPeakTflops}
                      onChange={(e) => setCustomPeakTflops(Number(e.target.value))}
                      className="w-full accent-indigo-600"
                    />
                  </div>

                  <div>
                    <div className="flex justify-between text-xs text-slate-600 mb-1">
                      <span>Peak HBM/GDDR Bandwidth:</span>
                      <span className="font-semibold text-emerald-600 font-mono">{customPeakBandwidth} GB/s</span>
                    </div>
                    <input
                      type="range"
                      min={50}
                      max={5000}
                      step={25}
                      value={customPeakBandwidth}
                      onChange={(e) => setCustomPeakBandwidth(Number(e.target.value))}
                      className="w-full accent-emerald-600"
                    />
                  </div>

                  <div className="bg-slate-50 rounded-lg p-2.5 border border-slate-100 flex items-center justify-between text-xs">
                    <span className="text-slate-600 font-medium">Knee Point (Denge Noktası I_knee):</span>
                    <span className="font-bold text-slate-900 font-mono">
                      {kneePointIntensity.toFixed(1)} FLOPs/Byte
                    </span>
                  </div>
                </div>
              </div>

              {/* Workload Profile Card */}
              <div className="bg-white rounded-xl p-5 border border-slate-200 shadow-sm space-y-4">
                <div className="flex items-center justify-between">
                  <h3 className="text-sm font-semibold text-slate-900 flex items-center gap-2">
                    <Cpu className="w-4 h-4 text-amber-600" /> Model / İş Yükü Profili
                  </h3>
                  <span className="text-xs bg-amber-50 text-amber-800 px-2 py-0.5 rounded font-mono">
                    Operational Intensity
                  </span>
                </div>

                <div>
                  <label className="block text-xs font-medium text-slate-700 mb-1">
                    Tipik DL İş Yükü
                  </label>
                  <select
                    id="workload-select"
                    value={selectedWorkloadIdx}
                    onChange={(e) => setSelectedWorkloadIdx(Number(e.target.value))}
                    className="w-full bg-slate-50 border border-slate-300 rounded-lg px-3 py-2 text-sm text-slate-900 focus:outline-none focus:ring-2 focus:ring-amber-500"
                  >
                    {PREDEFINED_WORKLOADS.map((wl, idx) => (
                      <option key={wl.name} value={idx}>
                        {wl.name} ({wl.intensity} FLOPs/B)
                      </option>
                    ))}
                  </select>
                  <p className="text-xs text-slate-500 mt-1">
                    {PREDEFINED_WORKLOADS[selectedWorkloadIdx].desc}
                  </p>
                </div>

                <div className="border-t border-slate-100 pt-3 space-y-3">
                  <div>
                    <div className="flex justify-between text-xs text-slate-600 mb-1">
                      <span>Operasyonel Yoğunluk (I = FLOPs / Byte):</span>
                      <span className="font-bold text-amber-600 font-mono">
                        {selectedWorkloadIdx === PREDEFINED_WORKLOADS.length - 1
                          ? customIntensity.toFixed(1)
                          : PREDEFINED_WORKLOADS[selectedWorkloadIdx].intensity.toFixed(1)}{' '}
                        FLOPs/B
                      </span>
                    </div>
                    {selectedWorkloadIdx === PREDEFINED_WORKLOADS.length - 1 && (
                      <input
                        type="range"
                        min={0.1}
                        max={1000}
                        step={0.5}
                        value={customIntensity}
                        onChange={(e) => setCustomIntensity(Number(e.target.value))}
                        className="w-full accent-amber-600 mt-2"
                      />
                    )}
                  </div>

                  <div className="bg-amber-50/60 rounded-lg p-2.5 border border-amber-100 text-xs text-amber-900 space-y-1">
                    <p className="font-medium flex items-center gap-1">
                      <Info className="w-3.5 h-3.5 text-amber-600" /> Operasyonel Yoğunluk Nedir?
                    </p>
                    <p className="text-[11px] leading-relaxed text-amber-800">
                      Bellekten okunan veya yazılan her 1 Bayt veri başına yapılan aritmetik işlem (FLOPs) sayısıdır.
                      Eğer I &lt; I_knee ise çekirdekler veri bekler (Memory-Bound); I &gt; I_knee ise hesaplama tavanına ulaşılır (Compute-Bound).
                    </p>
                  </div>
                </div>
              </div>

              {/* Real-time Roofline Status Card */}
              <div className="bg-white rounded-xl p-5 border border-slate-200 shadow-sm flex flex-col justify-between">
                <div>
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-sm font-semibold text-slate-900 flex items-center gap-2">
                      <BarChart3 className="w-4 h-4 text-emerald-600" /> Analitik Çıktı & Teşhis
                    </h3>
                    {rooflineData && (
                      <span
                        className={`text-xs px-2.5 py-0.5 rounded-full font-bold uppercase tracking-wider ${
                          rooflineData.regime === 'Memory-Bound'
                            ? 'bg-orange-100 text-orange-800 border border-orange-300'
                            : 'bg-indigo-100 text-indigo-800 border border-indigo-300'
                        }`}
                      >
                        {rooflineData.regime}
                      </span>
                    )}
                  </div>

                  {rooflineData ? (
                    <div className="space-y-3">
                      <div className="grid grid-cols-2 gap-2 text-center">
                        <div className="bg-slate-50 p-2.5 rounded-lg border border-slate-100">
                          <p className="text-[11px] text-slate-500 font-medium">Erişilebilir Güç</p>
                          <p className="text-base font-bold text-slate-900 font-mono">
                            {rooflineData.attainable_tflops.toFixed(1)} <span className="text-xs">TFLOPs</span>
                          </p>
                        </div>
                        <div className="bg-slate-50 p-2.5 rounded-lg border border-slate-100">
                          <p className="text-[11px] text-slate-500 font-medium">Donanım Verimi</p>
                          <p className="text-base font-bold text-indigo-600 font-mono">
                            %{rooflineData.efficiency_pct.toFixed(1)}
                          </p>
                        </div>
                      </div>

                      <div className="text-xs text-slate-600 bg-slate-50 p-3 rounded-lg border border-slate-200 space-y-1">
                        <p className="font-semibold text-slate-900">Sistem Mühendisliği Teşhisi:</p>
                        <p className="text-[11px] leading-relaxed">{rooflineData.diagnosis}</p>
                      </div>
                    </div>
                  ) : (
                    <div className="py-8 text-center text-slate-400 text-xs">
                      Analiz hesaplanıyor...
                    </div>
                  )}
                </div>

                <button
                  id="recalculate-roofline-btn"
                  onClick={runRooflineAnalysis}
                  disabled={rooflineLoading}
                  className="w-full mt-4 flex items-center justify-center gap-2 py-2 px-4 rounded-lg bg-slate-900 text-white text-xs font-semibold hover:bg-slate-800 transition-colors shadow-sm disabled:opacity-50"
                >
                  <RefreshCw className={`w-3.5 h-3.5 ${rooflineLoading ? 'animate-spin' : ''}`} />
                  Roofline Analizini Yenile
                </button>
              </div>
            </div>

            {/* Interactive Roofline SVG Visualization */}
            <div className="bg-white rounded-xl p-6 border border-slate-200 shadow-sm space-y-4">
              <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
                <div>
                  <h3 className="text-base font-bold text-slate-900 flex items-center gap-2">
                    <TrendingUp className="w-5 h-5 text-indigo-600" /> Roofline Diyagramı (Log-Log Ölçeği)
                  </h3>
                  <p className="text-xs text-slate-500">
                    Eğimli tavan bellek bant genişliğini (B_peak), yatay tavan tensör işlem gücünü (P_peak) gösterir.
                  </p>
                </div>
                <div className="flex items-center gap-4 text-xs font-medium">
                  <span className="flex items-center gap-1.5 text-emerald-600">
                    <span className="w-3 h-0.5 bg-emerald-500 inline-block"></span> Bellek Tavanı (B_peak)
                  </span>
                  <span className="flex items-center gap-1.5 text-indigo-600">
                    <span className="w-3 h-0.5 bg-indigo-500 inline-block"></span> Hesaplama Tavanı (P_peak)
                  </span>
                  <span className="flex items-center gap-1.5 text-amber-600">
                    <span className="w-2.5 h-2.5 rounded-full bg-amber-500 inline-block"></span> İş Yükü Noktası
                  </span>
                </div>
              </div>

              {/* SVG Canvas */}
              <div className="w-full bg-slate-900 rounded-xl p-4 sm:p-6 overflow-x-auto text-slate-200">
                <div className="min-w-[640px] h-[340px] relative">
                  <svg className="w-full h-full" viewBox="0 0 700 320">
                    <defs>
                      <linearGradient id="roofGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                        <stop offset="0%" stopColor="#10b981" />
                        <stop offset="45%" stopColor="#6366f1" />
                        <stop offset="100%" stopColor="#4f46e5" />
                      </linearGradient>
                      <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
                        <feGaussianBlur stdDeviation="3" result="glow" />
                        <feComposite in="SourceGraphic" in2="glow" operator="over" />
                      </filter>
                    </defs>

                    {/* Grid Lines */}
                    {[50, 100, 150, 200, 250].map((y) => (
                      <line
                        key={y}
                        x1="50"
                        y1={y}
                        x2="680"
                        y2={y}
                        stroke="#334155"
                        strokeDasharray="3 3"
                        strokeWidth="0.8"
                      />
                    ))}
                    {[120, 230, 340, 450, 560, 670].map((x) => (
                      <line
                        key={x}
                        x1={x}
                        y1="20"
                        x2={x}
                        y2="280"
                        stroke="#334155"
                        strokeDasharray="3 3"
                        strokeWidth="0.8"
                      />
                    ))}

                    {/* Axes */}
                    <line x1="50" y1="280" x2="680" y2="280" stroke="#94a3b8" strokeWidth="1.5" />
                    <line x1="50" y1="20" x2="50" y2="280" stroke="#94a3b8" strokeWidth="1.5" />

                    {/* Axis Labels */}
                    <text x="360" y="310" fill="#94a3b8" fontSize="11" textAnchor="middle" fontWeight="500">
                      Operasyonel Yoğunluk (FLOPs / Byte - Log Ölçeği)
                    </text>
                    <text
                      x="-150"
                      y="18"
                      fill="#94a3b8"
                      fontSize="11"
                      textAnchor="middle"
                      transform="rotate(-90)"
                      fontWeight="500"
                    >
                      Erişilebilir Performans (TFLOPs)
                    </text>

                    {/* X-axis tick labels */}
                    <text x="120" y="295" fill="#64748b" fontSize="10" textAnchor="middle">0.1</text>
                    <text x="230" y="295" fill="#64748b" fontSize="10" textAnchor="middle">1.0</text>
                    <text x="340" y="295" fill="#64748b" fontSize="10" textAnchor="middle">10</text>
                    <text x="450" y="295" fill="#64748b" fontSize="10" textAnchor="middle">100</text>
                    <text x="560" y="295" fill="#64748b" fontSize="10" textAnchor="middle">1,000</text>
                    <text x="670" y="295" fill="#64748b" fontSize="10" textAnchor="middle">10,000</text>

                    {/* Y-axis tick labels */}
                    <text x="42" y="283" fill="#64748b" fontSize="10" textAnchor="end">0.1</text>
                    <text x="42" y="203" fill="#64748b" fontSize="10" textAnchor="end">10</text>
                    <text x="42" y="123" fill="#64748b" fontSize="10" textAnchor="end">100</text>
                    <text x="42" y="53" fill="#64748b" fontSize="10" textAnchor="end">1,000</text>

                    {/* Knee point calculation for SVG coordinates */}
                    {(() => {
                      const mapX = (val: number) => {
                        const safeVal = Math.max(0.05, Math.min(val, 20000));
                        const logVal = Math.log10(safeVal);
                        const norm = (logVal - (-1)) / (4 - (-1));
                        return 120 + Math.max(0, Math.min(norm, 1)) * (670 - 120);
                      };

                      const mapY = (val: number) => {
                        const safeVal = Math.max(0.1, Math.min(val, 2000));
                        const logVal = Math.log10(safeVal);
                        const norm = (logVal - (-1)) / (3.3 - (-1));
                        return 280 - Math.max(0, Math.min(norm, 1)) * (280 - 45);
                      };

                      const kneeX = mapX(kneePointIntensity);
                      const kneeY = mapY(customPeakTflops);
                      const startX = mapX(0.05);
                      const startY = mapY((0.05 * customPeakBandwidth) / 1000);

                      const curIntensity =
                        rooflineData?.operational_intensity ||
                        (selectedWorkloadIdx === PREDEFINED_WORKLOADS.length - 1
                          ? customIntensity
                          : PREDEFINED_WORKLOADS[selectedWorkloadIdx].intensity);
                      const curTflops = rooflineData?.attainable_tflops || Math.min(customPeakTflops, (curIntensity * customPeakBandwidth) / 1000);

                      const opX = mapX(curIntensity);
                      const opY = mapY(curTflops);

                      return (
                        <g>
                          {/* Knee vertical guideline */}
                          <line
                            x1={kneeX}
                            y1={kneeY}
                            x2={kneeX}
                            y2="280"
                            stroke="#f59e0b"
                            strokeDasharray="4 4"
                            strokeWidth="1.2"
                          />
                          <text
                            x={kneeX}
                            y="38"
                            fill="#f59e0b"
                            fontSize="10"
                            textAnchor="middle"
                            fontWeight="600"
                          >
                            I_knee = {kneePointIntensity.toFixed(1)} FLOPs/B
                          </text>

                          {/* Roofline Slanted Bandwidth Ceiling */}
                          <line
                            x1={startX}
                            y1={startY}
                            x2={kneeX}
                            y2={kneeY}
                            stroke="#10b981"
                            strokeWidth="3.5"
                            strokeLinecap="round"
                          />

                          {/* Roofline Flat Compute Ceiling */}
                          <line
                            x1={kneeX}
                            y1={kneeY}
                            x2="670"
                            y2={kneeY}
                            stroke="#6366f1"
                            strokeWidth="3.5"
                            strokeLinecap="round"
                          />

                          {/* Operating Point Shadow / Pulse */}
                          <circle
                            cx={opX}
                            cy={opY}
                            r="10"
                            fill="#f59e0b"
                            opacity="0.3"
                            className="animate-ping"
                          />
                          {/* Operating Point Main */}
                          <circle
                            cx={opX}
                            cy={opY}
                            r="6"
                            fill="#f59e0b"
                            stroke="#ffffff"
                            strokeWidth="2"
                            filter="url(#glow)"
                          />

                          {/* Operating Point Callout */}
                          <rect
                            x={Math.min(opX + 10, 520)}
                            y={Math.max(opY - 35, 30)}
                            width="145"
                            height="38"
                            rx="5"
                            fill="#1e293b"
                            stroke="#475569"
                            strokeWidth="1"
                          />
                          <text
                            x={Math.min(opX + 16, 526)}
                            y={Math.max(opY - 20, 45)}
                            fill="#f8fafc"
                            fontSize="9"
                            fontWeight="bold"
                          >
                            {rooflineData?.regime || 'Aktif İş Yükü'}
                          </text>
                          <text
                            x={Math.min(opX + 16, 526)}
                            y={Math.max(opY - 8, 57)}
                            fill="#94a3b8"
                            fontSize="8"
                          >
                            {curTflops.toFixed(1)} TFLOPs ({curIntensity.toFixed(1)} F/B)
                          </text>
                        </g>
                      );
                    })()}
                  </svg>
                </div>
              </div>

              {/* Explanatory Cards */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-2">
                <div className="p-4 rounded-lg bg-orange-50/70 border border-orange-200">
                  <h4 className="text-xs font-bold text-orange-900 uppercase tracking-wider mb-1 flex items-center gap-1.5">
                    <AlertTriangle className="w-4 h-4 text-orange-600" /> Sol Bölge: Memory-Bound (Bellek Duvarı)
                  </h4>
                  <p className="text-xs text-orange-800 leading-relaxed">
                    Operasyonel yoğunluk I &lt; I_knee durumundadır. Tensör çekirdekleri yüksek hızda veri beslenemediği için boşa bekler (idle cycles).
                    Performansı artırmanın yolu <strong>Kernel Fusion</strong> (operatörleri birleştirme), <strong>FlashAttention</strong> (HBM I/O azaltma) 
                    veya daha büyük Batch Size kullanmaktır.
                  </p>
                </div>

                <div className="p-4 rounded-lg bg-indigo-50/70 border border-indigo-200">
                  <h4 className="text-xs font-bold text-indigo-900 uppercase tracking-wider mb-1 flex items-center gap-1.5">
                    <CheckCircle2 className="w-4 h-4 text-indigo-600" /> Sağ Bölge: Compute-Bound (İşlem Tavanı)
                  </h4>
                  <p className="text-xs text-indigo-800 leading-relaxed">
                    Operasyonel yoğunluk I &ge; I_knee durumundadır. Çekirdekler bellek bant genişliğine takılmadan tam kapasite çalışır.
                    Performansı daha da yukarı çekmek için <strong>Kuantizasyon (FP16 ➔ INT8 / INT4)</strong> ile daha yüksek TFLOPs kapasitesine sahip çekirdek modlarına geçilmelidir.
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ================================================================= */}
        {/* TAB 2: QUANTIZATION SIMULATOR (FP32 -> FP16 -> INT8 -> INT4)       */}
        {/* ================================================================= */}
        {activeTab === 'quantization' && (
          <div className="space-y-6">
            {/* Control & Parameters Card */}
            <div className="bg-white rounded-xl p-5 border border-slate-200 shadow-sm space-y-4">
              <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
                <div>
                  <h3 className="text-base font-bold text-slate-900 flex items-center gap-2">
                    <Activity className="w-5 h-5 text-indigo-600" /> Tensör Dağılımı ve Kuantizasyon Parametreleri
                  </h3>
                  <p className="text-xs text-slate-500">
                    Ağırlık ve aktivasyon tensörlerinde sayısal hassasiyet dönüşümlerinin (FP32 ➔ FP16 ➔ INT8 ➔ INT4) hata oranlarını inceleyin.
                  </p>
                </div>
                <button
                  id="simulate-quant-btn"
                  onClick={runQuantSimulation}
                  disabled={quantLoading}
                  className="flex items-center gap-2 py-2 px-5 rounded-lg bg-indigo-600 text-white text-xs font-semibold hover:bg-indigo-700 transition-colors shadow-sm disabled:opacity-50"
                >
                  <RefreshCw className={`w-3.5 h-3.5 ${quantLoading ? 'animate-spin' : ''}`} />
                  Kuantizasyonu Simüle Et
                </button>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-4 gap-4 pt-2">
                <div>
                  <label className="block text-xs font-medium text-slate-700 mb-1">
                    Veri Dağılım Tipi
                  </label>
                  <select
                    id="distribution-select"
                    value={distType}
                    onChange={(e) => setDistType(e.target.value as any)}
                    className="w-full bg-slate-50 border border-slate-300 rounded-lg px-3 py-2 text-sm text-slate-900 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                  >
                    <option value="outlier">Aykırı Değerli (LLM Outliers - Gerçekçi)</option>
                    <option value="normal">Standart Gauss Normal Dağılım</option>
                    <option value="uniform">Düzgün (Uniform) Dağılım</option>
                  </select>
                </div>

                <div>
                  <div className="flex justify-between text-xs text-slate-600 mb-1">
                    <span>Eleman Sayısı (N):</span>
                    <span className="font-semibold text-slate-900 font-mono">{numElements.toLocaleString()}</span>
                  </div>
                  <input
                    type="range"
                    min={500}
                    max={10000}
                    step={500}
                    value={numElements}
                    onChange={(e) => setNumElements(Number(e.target.value))}
                    className="w-full accent-indigo-600"
                  />
                </div>

                {distType === 'outlier' && (
                  <>
                    <div>
                      <div className="flex justify-between text-xs text-slate-600 mb-1">
                        <span>Aykırı Değer Oranı:</span>
                        <span className="font-semibold text-rose-600 font-mono">{(outlierRatio * 100).toFixed(1)}%</span>
                      </div>
                      <input
                        type="range"
                        min={0.002}
                        max={0.05}
                        step={0.002}
                        value={outlierRatio}
                        onChange={(e) => setOutlierRatio(Number(e.target.value))}
                        className="w-full accent-rose-600"
                      />
                    </div>

                    <div>
                      <div className="flex justify-between text-xs text-slate-600 mb-1">
                        <span>Aykırı Değer Büyüklüğü:</span>
                        <span className="font-semibold text-rose-600 font-mono">{outlierMag.toFixed(1)}x</span>
                      </div>
                      <input
                        type="range"
                        min={2.0}
                        max={12.0}
                        step={0.5}
                        value={outlierMag}
                        onChange={(e) => setOutlierMag(Number(e.target.value))}
                        className="w-full accent-rose-600"
                      />
                    </div>
                  </>
                )}
              </div>
            </div>

            {/* Results Grid */}
            {quantData && (
              <div className="space-y-6">
                {/* Stats Summary & Cards */}
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                  {Object.entries(quantData.formats).map(([formatKey, item]) => {
                    const isFp32 = formatKey === 'fp32';
                    const isInt4 = formatKey === 'int4';
                    const isInt8 = formatKey === 'int8';

                    return (
                      <div
                        key={formatKey}
                        className={`rounded-xl p-5 border shadow-sm transition-all ${
                          isFp32
                            ? 'bg-slate-50 border-slate-300'
                            : isInt8
                            ? 'bg-emerald-50/50 border-emerald-300'
                            : isInt4
                            ? 'bg-amber-50/50 border-amber-300'
                            : 'bg-white border-slate-200'
                        }`}
                      >
                        <div className="flex items-center justify-between mb-3">
                          <span
                            className={`text-xs px-2.5 py-0.5 rounded-full font-bold uppercase ${
                              isFp32
                                ? 'bg-slate-200 text-slate-800'
                                : isInt8
                                ? 'bg-emerald-100 text-emerald-800'
                                : isInt4
                                ? 'bg-amber-100 text-amber-800'
                                : 'bg-blue-100 text-blue-800'
                            }`}
                          >
                            {formatKey.toUpperCase()} ({item.bits}-bit)
                          </span>
                          <span className="text-xs font-mono font-bold text-slate-700">
                            {item.compression_ratio.toFixed(1)}x Sıkıştırma
                          </span>
                        </div>

                        <div className="space-y-2 text-xs">
                          <div className="flex justify-between py-1 border-b border-slate-200/60">
                            <span className="text-slate-500">Bellek Boyutu:</span>
                            <span className="font-mono font-semibold text-slate-900">
                              {item.quantized_memory_kb.toFixed(2)} KB
                            </span>
                          </div>
                          <div className="flex justify-between py-1 border-b border-slate-200/60">
                            <span className="text-slate-500">Ortalama Kare Hata (MSE):</span>
                            <span className="font-mono font-semibold text-slate-900">
                              {item.mse.toExponential(3)}
                            </span>
                          </div>
                          <div className="flex justify-between py-1 border-b border-slate-200/60">
                            <span className="text-slate-500">Ortalama Mutlak Hata (MAE):</span>
                            <span className="font-mono font-semibold text-slate-900">
                              {item.mae.toExponential(3)}
                            </span>
                          </div>
                          <div className="flex justify-between py-1 border-b border-slate-200/60">
                            <span className="text-slate-500">Sinyal/Gürültü (SNR):</span>
                            <span
                              className={`font-mono font-bold ${
                                item.snr_db > 30
                                  ? 'text-emerald-600'
                                  : item.snr_db > 15
                                  ? 'text-amber-600'
                                  : 'text-rose-600'
                              }`}
                            >
                              {item.snr_db === Infinity ? '∞' : `${item.snr_db.toFixed(1)} dB`}
                            </span>
                          </div>
                          {item.scale !== undefined && (
                            <div className="flex justify-between py-1 text-[11px] text-slate-500">
                              <span>Scale (S): {item.scale.toFixed(4)}</span>
                              <span>Zero-Point (Z): {item.zero_point}</span>
                            </div>
                          )}
                        </div>
                      </div>
                    );
                  })}
                </div>

                {/* Histogram Distribution Comparison */}
                <div className="bg-white rounded-xl p-6 border border-slate-200 shadow-sm space-y-4">
                  <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
                    <div>
                      <h4 className="text-sm font-bold text-slate-900 flex items-center gap-2">
                        <BarChart3 className="w-4 h-4 text-indigo-600" /> 20-Kanal Yoğunluk Histogramı (Dağılım Bozulması)
                      </h4>
                      <p className="text-xs text-slate-500">
                        Kuantizasyon basamaklarının orijinal sürekli tensör dağılımını nasıl adımlara (discretize) böldüğünü görün.
                      </p>
                    </div>

                    <div className="flex items-center gap-1 bg-slate-100 p-1 rounded-lg text-xs font-medium">
                      <button
                        onClick={() => setSelectedHistView('original')}
                        className={`px-3 py-1 rounded-md transition-all ${
                          selectedHistView === 'original'
                            ? 'bg-white shadow text-slate-900 font-semibold'
                            : 'text-slate-600 hover:text-slate-900'
                        }`}
                      >
                        Orijinal (FP32)
                      </button>
                      <button
                        onClick={() => setSelectedHistView('int8')}
                        className={`px-3 py-1 rounded-md transition-all ${
                          selectedHistView === 'int8'
                            ? 'bg-white shadow text-emerald-700 font-semibold'
                            : 'text-slate-600 hover:text-slate-900'
                        }`}
                      >
                        INT8 (Affine)
                      </button>
                      <button
                        onClick={() => setSelectedHistView('int4')}
                        className={`px-3 py-1 rounded-md transition-all ${
                          selectedHistView === 'int4'
                            ? 'bg-white shadow text-amber-700 font-semibold'
                            : 'text-slate-600 hover:text-slate-900'
                        }`}
                      >
                        INT4 (Affine)
                      </button>
                    </div>
                  </div>

                  {/* Histogram Chart */}
                  <div className="h-64 flex items-end gap-1.5 pt-8 pb-4 px-2 border-b border-slate-200 bg-slate-50/70 rounded-lg">
                    {(() => {
                      const currentBins =
                        selectedHistView === 'original'
                          ? quantData.histogram_original
                          : selectedHistView === 'int8'
                          ? quantData.histogram_int8
                          : quantData.histogram_int4;

                      const maxDensity = Math.max(...currentBins.map((b) => b.density), 0.001);

                      return currentBins.map((bin, idx) => {
                        const heightPct = Math.max(4, (bin.density / maxDensity) * 100);
                        const barColor =
                          selectedHistView === 'original'
                            ? 'bg-indigo-500 hover:bg-indigo-600'
                            : selectedHistView === 'int8'
                            ? 'bg-emerald-500 hover:bg-emerald-600'
                            : 'bg-amber-500 hover:bg-amber-600';

                        return (
                          <div
                            key={idx}
                            className="flex-1 flex flex-col items-center h-full justify-end group relative"
                          >
                            {/* Tooltip */}
                            <div className="absolute -top-12 hidden group-hover:flex flex-col items-center z-10 pointer-events-none">
                              <div className="bg-slate-900 text-white text-[10px] py-1 px-2 rounded shadow-md whitespace-nowrap">
                                Değer: {bin.bin_center.toFixed(2)} | Yoğunluk: {(bin.density * 100).toFixed(1)}% ({bin.count})
                              </div>
                              <div className="w-1.5 h-1.5 bg-slate-900 rotate-45 -mt-0.5"></div>
                            </div>

                            <div
                              style={{ height: `${heightPct}%` }}
                              className={`w-full rounded-t transition-all ${barColor}`}
                            />
                            <span className="text-[9px] text-slate-400 mt-1 font-mono truncate w-full text-center">
                              {bin.bin_center.toFixed(1)}
                            </span>
                          </div>
                        );
                      });
                    })()}
                  </div>
                </div>

                {/* Practical Engineering Insights */}
                <div className="bg-slate-50 rounded-xl p-5 border border-slate-200 space-y-3">
                  <h4 className="text-xs font-bold text-slate-900 uppercase tracking-wider flex items-center gap-1.5">
                    <Sparkles className="w-4 h-4 text-indigo-600" /> Kuantizasyon & LLM Donanım Çıkarımları
                  </h4>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs text-slate-700">
                    {quantData.insights.map((insight, idx) => (
                      <div key={idx} className="flex items-start gap-2 bg-white p-3 rounded-lg border border-slate-200/80 shadow-2xs">
                        <span className="w-5 h-5 rounded-full bg-indigo-50 text-indigo-600 flex items-center justify-center font-bold text-[10px] shrink-0 mt-0.5">
                          {idx + 1}
                        </span>
                        <p className="leading-relaxed">{insight}</p>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* ================================================================= */}
        {/* TAB 3: GPU VRAM & MEMORY DECOMPOSITION SIMULATOR                   */}
        {/* ================================================================= */}
        {activeTab === 'vram' && (
          <div className="space-y-6">
            {/* Control Panel Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Architecture & Model Parameters */}
              <div className="bg-white rounded-xl p-5 border border-slate-200 shadow-sm space-y-4">
                <div className="flex items-center justify-between">
                  <h3 className="text-sm font-semibold text-slate-900 flex items-center gap-2">
                    <Sliders className="w-4 h-4 text-indigo-600" /> Model Parametreleri
                  </h3>
                  {/* Mode switcher */}
                  <div className="flex items-center gap-1 bg-slate-100 p-0.5 rounded-lg text-xs font-medium">
                    <button
                      id="mode-training-btn"
                      onClick={() => setVramMode('training')}
                      className={`px-2.5 py-1 rounded-md transition-all ${
                        vramMode === 'training'
                          ? 'bg-white shadow text-indigo-600 font-bold'
                          : 'text-slate-600 hover:text-slate-900'
                      }`}
                    >
                      Eğitim (Training)
                    </button>
                    <button
                      id="mode-inference-btn"
                      onClick={() => setVramMode('inference')}
                      className={`px-2.5 py-1 rounded-md transition-all ${
                        vramMode === 'inference'
                          ? 'bg-white shadow text-emerald-600 font-bold'
                          : 'text-slate-600 hover:text-slate-900'
                      }`}
                    >
                      Çıkarım (Inference)
                    </button>
                  </div>
                </div>

                <div className="space-y-3">
                  <div>
                    <div className="flex justify-between text-xs text-slate-600 mb-1">
                      <span>Model Boyutu:</span>
                      <span className="font-bold text-slate-900 font-mono">{modelParamB} Milyar Parametre</span>
                    </div>
                    <input
                      id="model-param-slider"
                      type="range"
                      min={0.1}
                      max={70}
                      step={0.5}
                      value={modelParamB}
                      onChange={(e) => setModelParamB(Number(e.target.value))}
                      className="w-full accent-indigo-600"
                    />
                    <div className="flex justify-between text-[10px] text-slate-400 mt-0.5">
                      <span onClick={() => setModelParamB(0.124)} className="cursor-pointer hover:text-indigo-600">NanoGPT (124M)</span>
                      <span onClick={() => setModelParamB(7.0)} className="cursor-pointer hover:text-indigo-600">7B</span>
                      <span onClick={() => setModelParamB(13.0)} className="cursor-pointer hover:text-indigo-600">13B</span>
                      <span onClick={() => setModelParamB(70.0)} className="cursor-pointer hover:text-indigo-600">LLaMA-70B</span>
                    </div>
                  </div>

                  <div>
                    <label className="block text-xs font-medium text-slate-700 mb-1">
                      Ağırlık Hassasiyeti (Precision)
                    </label>
                    <select
                      id="vram-precision-select"
                      value={precision}
                      onChange={(e) => setPrecision(e.target.value as any)}
                      className="w-full bg-slate-50 border border-slate-300 rounded-lg px-3 py-2 text-sm text-slate-900 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                    >
                      <option value="fp32">FP32 (4 Byte/param) - Tam Hassasiyet</option>
                      <option value="fp16">FP16 / BF16 (2 Byte/param) - Endüstri Standardı</option>
                      <option value="int8">INT8 (1 Byte/param) - 8-bit Kuantizasyon</option>
                      <option value="int4">INT4 (0.5 Byte/param) - QLoRA / AWQ</option>
                    </select>
                  </div>

                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <div className="flex justify-between text-xs text-slate-600 mb-1">
                        <span>Batch Size:</span>
                        <span className="font-bold text-slate-900 font-mono">{batchSize}</span>
                      </div>
                      <input
                        type="range"
                        min={1}
                        max={64}
                        step={1}
                        value={batchSize}
                        onChange={(e) => setBatchSize(Number(e.target.value))}
                        className="w-full accent-indigo-600"
                      />
                    </div>

                    <div>
                      <div className="flex justify-between text-xs text-slate-600 mb-1">
                        <span>Dizi Boyu (Seq):</span>
                        <span className="font-bold text-slate-900 font-mono">{seqLen}</span>
                      </div>
                      <input
                        type="range"
                        min={512}
                        max={16384}
                        step={512}
                        value={seqLen}
                        onChange={(e) => setSeqLen(Number(e.target.value))}
                        className="w-full accent-indigo-600"
                      />
                    </div>
                  </div>
                </div>
              </div>

              {/* Training Optimizations & GPU Selection */}
              <div className="bg-white rounded-xl p-5 border border-slate-200 shadow-sm space-y-4">
                <div className="flex items-center justify-between">
                  <h3 className="text-sm font-semibold text-slate-900 flex items-center gap-2">
                    <Server className="w-4 h-4 text-emerald-600" /> Donanım & Optimizasyon Ayarları
                  </h3>
                </div>

                <div className="space-y-3">
                  <div>
                    <label className="block text-xs font-medium text-slate-700 mb-1">
                      Hedef GPU VRAM Kapasitesi
                    </label>
                    <div className="grid grid-cols-3 gap-1.5">
                      {[
                        { cap: 8, label: '8 GB (RTX 4060)' },
                        { cap: 16, label: '16 GB (T4/V100)' },
                        { cap: 24, label: '24 GB (RTX 4090)' },
                        { cap: 40, label: '40 GB (A100)' },
                        { cap: 80, label: '80 GB (A100/H100)' },
                        { cap: 128, label: '128 GB (M3 Max)' },
                      ].map((item) => (
                        <button
                          key={item.cap}
                          type="button"
                          onClick={() => setGpuVramCap(item.cap)}
                          className={`py-1.5 px-2 rounded-lg text-xs font-medium border text-center transition-all ${
                            gpuVramCap === item.cap
                              ? 'bg-indigo-50 border-indigo-600 text-indigo-700 font-bold'
                              : 'border-slate-200 text-slate-600 hover:bg-slate-50'
                          }`}
                        >
                          {item.label}
                        </button>
                      ))}
                    </div>
                  </div>

                  {vramMode === 'training' ? (
                    <>
                      <div>
                        <label className="block text-xs font-medium text-slate-700 mb-1">
                          Optimizatör Tipi (Durum Belleği)
                        </label>
                        <select
                          id="optimizer-select"
                          value={optimizer}
                          onChange={(e) => setOptimizer(e.target.value as any)}
                          className="w-full bg-slate-50 border border-slate-300 rounded-lg px-3 py-2 text-sm text-slate-900 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                        >
                          <option value="adamw">AdamW (12-16 Byte/param: m, v & master weights)</option>
                          <option value="adamw_8bit">8-bit Adam (bitsandbytes, ~6-8 Byte/param)</option>
                          <option value="sgd">SGD with Momentum (~4 Byte/param)</option>
                        </select>
                      </div>

                      <div className="pt-1">
                        <label className="flex items-center gap-2 cursor-pointer">
                          <input
                            type="checkbox"
                            checked={activationCheckpointing}
                            onChange={(e) => setActivationCheckpointing(e.target.checked)}
                            className="w-4 h-4 rounded text-indigo-600 accent-indigo-600"
                          />
                          <span className="text-xs text-slate-700 font-medium">
                            Aktivasyon Yeniden Hesaplama (Gradient Checkpointing)
                          </span>
                        </label>
                        <p className="text-[11px] text-slate-500 ml-6 mt-0.5">
                          Bellek kullanımını %60-70 azaltır, ~%20-25 FLOPs ekler.
                        </p>
                      </div>
                    </>
                  ) : (
                    <div className="bg-emerald-50 rounded-lg p-3 border border-emerald-200 text-xs text-emerald-800 space-y-1">
                      <p className="font-semibold flex items-center gap-1">
                        <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" /> Çıkarım (Inference) Modu Aktif
                      </p>
                      <p className="text-[11px] leading-relaxed">
                        Gradyanlar ve optimizatör bellekten silindi. Bellek yükünü ağırlıklar ve 
                        dizi boyu uzadıkça büyüyen <strong>KV-Cache</strong> (2 · B · S · H_kv · d_head · L) oluşturur.
                      </p>
                    </div>
                  )}
                </div>
              </div>

              {/* Status & OOM Risk Indicator */}
              <div className="bg-white rounded-xl p-5 border border-slate-200 shadow-sm flex flex-col justify-between">
                <div>
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-sm font-semibold text-slate-900 flex items-center gap-2">
                      <HardDrive className="w-4 h-4 text-indigo-600" /> VRAM Tüketim Durumu
                    </h3>
                    {vramData && (
                      <span
                        className={`text-xs px-2.5 py-0.5 rounded-full font-bold uppercase tracking-wider ${
                          vramData.oom_risk === 'LOW'
                            ? 'bg-emerald-100 text-emerald-800 border border-emerald-300'
                            : vramData.oom_risk === 'WARNING'
                            ? 'bg-amber-100 text-amber-800 border border-amber-300'
                            : 'bg-rose-100 text-rose-800 border border-rose-300'
                        }`}
                      >
                        {vramData.oom_risk === 'LOW'
                          ? 'GÜVENLİ (FIT)'
                          : vramData.oom_risk === 'WARNING'
                          ? 'KRİTİK SINIR'
                          : 'CUDA OOM!'}
                      </span>
                    )}
                  </div>

                  {vramData ? (
                    <div className="space-y-4">
                      <div className="text-center bg-slate-50 p-4 rounded-xl border border-slate-200">
                        <p className="text-xs text-slate-500 font-medium">Toplam Gerekli VRAM</p>
                        <p className="text-3xl font-extrabold text-slate-900 font-mono mt-1">
                          {vramData.total_memory_gb.toFixed(2)}{' '}
                          <span className="text-sm font-semibold text-slate-500">/ {vramData.gpu_vram_gb} GB</span>
                        </p>
                        <div className="w-full bg-slate-200 h-2.5 rounded-full mt-3 overflow-hidden">
                          <div
                            style={{ width: `${Math.min(100, vramData.memory_utilization_pct)}%` }}
                            className={`h-full transition-all ${
                              vramData.oom_risk === 'LOW'
                                ? 'bg-emerald-500'
                                : vramData.oom_risk === 'WARNING'
                                ? 'bg-amber-500'
                                : 'bg-rose-500'
                            }`}
                          />
                        </div>
                        <p className="text-xs font-semibold text-slate-600 mt-1.5 font-mono">
                          Kapasite Kullanımı: %{vramData.memory_utilization_pct.toFixed(1)}
                        </p>
                      </div>

                      {vramData.oom_risk === 'CRITICAL_OOM' && (
                        <div className="p-3 rounded-lg bg-rose-50 border border-rose-200 flex items-start gap-2 text-xs text-rose-800">
                          <AlertCircle className="w-4 h-4 text-rose-600 shrink-0 mt-0.5" />
                          <p>
                            <strong>CUDA Out-of-Memory Hatası!</strong> Model bu GPU'ya sığmıyor. 
                            Parametreleri kuantize edin (INT8/4), batch boyutunu küçültün veya ZeRO-3 / FSDP kullanın.
                          </p>
                        </div>
                      )}
                    </div>
                  ) : (
                    <div className="py-8 text-center text-slate-400 text-xs">Hesaplanıyor...</div>
                  )}
                </div>

                <button
                  id="recalculate-vram-btn"
                  onClick={runVramSimulation}
                  disabled={vramLoading}
                  className="w-full mt-4 flex items-center justify-center gap-2 py-2 px-4 rounded-lg bg-slate-900 text-white text-xs font-semibold hover:bg-slate-800 transition-colors shadow-sm disabled:opacity-50"
                >
                  <RefreshCw className={`w-3.5 h-3.5 ${vramLoading ? 'animate-spin' : ''}`} />
                  VRAM Dağılımını Yenile
                </button>
              </div>
            </div>

            {/* Stacked Memory Breakdown Visualization */}
            {vramData && (
              <div className="bg-white rounded-xl p-6 border border-slate-200 shadow-sm space-y-5">
                <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
                  <div>
                    <h3 className="text-base font-bold text-slate-900 flex items-center gap-2">
                      <Layers className="w-5 h-5 text-indigo-600" /> VRAM Bileşen Ayrıştırması (Memory Decomposition)
                    </h3>
                    <p className="text-xs text-slate-500">
                      GPU belleğinin her bir alt bileşen tarafından tüketilen net miktarı (GB).
                    </p>
                  </div>
                </div>

                {/* Stacked Bar Visualizer */}
                <div className="space-y-2">
                  <div className="w-full h-8 bg-slate-100 rounded-lg overflow-hidden flex shadow-inner">
                    {vramData.components.weights_gb > 0 && (
                      <div
                        style={{ width: `${(vramData.components.weights_gb / vramData.total_memory_gb) * 100}%` }}
                        className="bg-indigo-600 h-full transition-all group relative hover:opacity-90"
                        title={`Ağırlıklar: ${vramData.components.weights_gb.toFixed(2)} GB`}
                      />
                    )}
                    {vramData.components.gradients_gb > 0 && (
                      <div
                        style={{ width: `${(vramData.components.gradients_gb / vramData.total_memory_gb) * 100}%` }}
                        className="bg-rose-500 h-full transition-all group relative hover:opacity-90"
                        title={`Gradyanlar: ${vramData.components.gradients_gb.toFixed(2)} GB`}
                      />
                    )}
                    {vramData.components.optimizer_gb > 0 && (
                      <div
                        style={{ width: `${(vramData.components.optimizer_gb / vramData.total_memory_gb) * 100}%` }}
                        className="bg-amber-500 h-full transition-all group relative hover:opacity-90"
                        title={`Optimizatör: ${vramData.components.optimizer_gb.toFixed(2)} GB`}
                      />
                    )}
                    {vramData.components.activations_gb > 0 && (
                      <div
                        style={{ width: `${(vramData.components.activations_gb / vramData.total_memory_gb) * 100}%` }}
                        className="bg-cyan-500 h-full transition-all group relative hover:opacity-90"
                        title={`Aktivasyonlar: ${vramData.components.activations_gb.toFixed(2)} GB`}
                      />
                    )}
                    {vramData.components.kv_cache_gb > 0 && (
                      <div
                        style={{ width: `${(vramData.components.kv_cache_gb / vramData.total_memory_gb) * 100}%` }}
                        className="bg-emerald-500 h-full transition-all group relative hover:opacity-90"
                        title={`KV-Cache: ${vramData.components.kv_cache_gb.toFixed(2)} GB`}
                      />
                    )}
                    {vramData.components.cuda_overhead_gb > 0 && (
                      <div
                        style={{ width: `${(vramData.components.cuda_overhead_gb / vramData.total_memory_gb) * 100}%` }}
                        className="bg-slate-400 h-full transition-all group relative hover:opacity-90"
                        title={`CUDA Context: ${vramData.components.cuda_overhead_gb.toFixed(2)} GB`}
                      />
                    )}
                  </div>

                  {/* Legend Grid */}
                  <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3 pt-3">
                    <div className="bg-slate-50 p-3 rounded-lg border border-slate-200">
                      <div className="flex items-center gap-1.5 text-xs text-indigo-700 font-semibold mb-1">
                        <span className="w-2.5 h-2.5 rounded-full bg-indigo-600"></span> Ağırlıklar (Weights)
                      </div>
                      <p className="text-sm font-bold text-slate-900 font-mono">
                        {vramData.components.weights_gb.toFixed(2)} GB
                      </p>
                      <p className="text-[10px] text-slate-500">
                        %{((vramData.components.weights_gb / vramData.total_memory_gb) * 100).toFixed(1)}
                      </p>
                    </div>

                    <div className="bg-slate-50 p-3 rounded-lg border border-slate-200">
                      <div className="flex items-center gap-1.5 text-xs text-rose-700 font-semibold mb-1">
                        <span className="w-2.5 h-2.5 rounded-full bg-rose-500"></span> Gradyanlar (Gradients)
                      </div>
                      <p className="text-sm font-bold text-slate-900 font-mono">
                        {vramData.components.gradients_gb.toFixed(2)} GB
                      </p>
                      <p className="text-[10px] text-slate-500">
                        %{((vramData.components.gradients_gb / vramData.total_memory_gb) * 100).toFixed(1)}
                      </p>
                    </div>

                    <div className="bg-slate-50 p-3 rounded-lg border border-slate-200">
                      <div className="flex items-center gap-1.5 text-xs text-amber-700 font-semibold mb-1">
                        <span className="w-2.5 h-2.5 rounded-full bg-amber-500"></span> Optimizatör Durumu
                      </div>
                      <p className="text-sm font-bold text-slate-900 font-mono">
                        {vramData.components.optimizer_gb.toFixed(2)} GB
                      </p>
                      <p className="text-[10px] text-slate-500">
                        %{((vramData.components.optimizer_gb / vramData.total_memory_gb) * 100).toFixed(1)}
                      </p>
                    </div>

                    <div className="bg-slate-50 p-3 rounded-lg border border-slate-200">
                      <div className="flex items-center gap-1.5 text-xs text-cyan-700 font-semibold mb-1">
                        <span className="w-2.5 h-2.5 rounded-full bg-cyan-500"></span> Aktivasyonlar
                      </div>
                      <p className="text-sm font-bold text-slate-900 font-mono">
                        {vramData.components.activations_gb.toFixed(2)} GB
                      </p>
                      <p className="text-[10px] text-slate-500">
                        %{((vramData.components.activations_gb / vramData.total_memory_gb) * 100).toFixed(1)}
                      </p>
                    </div>

                    <div className="bg-slate-50 p-3 rounded-lg border border-slate-200">
                      <div className="flex items-center gap-1.5 text-xs text-emerald-700 font-semibold mb-1">
                        <span className="w-2.5 h-2.5 rounded-full bg-emerald-500"></span> KV-Cache
                      </div>
                      <p className="text-sm font-bold text-slate-900 font-mono">
                        {vramData.components.kv_cache_gb.toFixed(2)} GB
                      </p>
                      <p className="text-[10px] text-slate-500">
                        %{((vramData.components.kv_cache_gb / vramData.total_memory_gb) * 100).toFixed(1)}
                      </p>
                    </div>

                    <div className="bg-slate-50 p-3 rounded-lg border border-slate-200">
                      <div className="flex items-center gap-1.5 text-xs text-slate-700 font-semibold mb-1">
                        <span className="w-2.5 h-2.5 rounded-full bg-slate-400"></span> CUDA Context
                      </div>
                      <p className="text-sm font-bold text-slate-900 font-mono">
                        {vramData.components.cuda_overhead_gb.toFixed(2)} GB
                      </p>
                      <p className="text-[10px] text-slate-500">
                        %{((vramData.components.cuda_overhead_gb / vramData.total_memory_gb) * 100).toFixed(1)}
                      </p>
                    </div>
                  </div>
                </div>

                {/* Recommendations */}
                <div className="bg-slate-50 rounded-xl p-5 border border-slate-200 space-y-3">
                  <h4 className="text-xs font-bold text-slate-900 uppercase tracking-wider flex items-center gap-1.5">
                    <Sparkles className="w-4 h-4 text-indigo-600" /> Donanım Optimizasyonu ve Dağıtık Eğitim Önerileri
                  </h4>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs text-slate-700">
                    {vramData.recommendations.map((rec, idx) => (
                      <div key={idx} className="flex items-start gap-2 bg-white p-3 rounded-lg border border-slate-200/80 shadow-2xs">
                        <span className="w-5 h-5 rounded-full bg-indigo-50 text-indigo-600 flex items-center justify-center font-bold text-[10px] shrink-0 mt-0.5">
                          {idx + 1}
                        </span>
                        <p className="leading-relaxed">{rec}</p>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}
