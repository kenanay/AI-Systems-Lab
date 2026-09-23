'use client';

import React, { useState, useEffect } from 'react';
import Link from 'next/link';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend
} from 'recharts';
import {
  Play,
  Square,
  RefreshCw,
  Cpu,
  Layers,
  Database,
  Sliders,
  Award,
  CheckCircle2,
  AlertCircle,
  Clock
} from 'lucide-react';
import { api, TrainingJobResponse, StartTrainingRequest, listTokenizers } from '@/lib/api';

export default function TrainingPage() {
  const queryClient = useQueryClient();

  // Form State
  const [jobName, setJobName] = useState('Türkçe GPT Eğitimi #1');
  const [modelName, setModelName] = useState('gpt-turkish-tiny');
  const [jobType, setJobType] = useState<'PRETRAIN' | 'FULL_SFT' | 'LORA_SFT'>('PRETRAIN');
  const [baseModel, setBaseModel] = useState('');
  const [baseVersion, setBaseVersion] = useState('');
  const [maxSeqLen, setMaxSeqLen] = useState(128);
  const [device, setDevice] = useState('cpu');
  const [datasetId, setDatasetId] = useState('');
  const [tokenizerId, setTokenizerId] = useState('');
  const [epochs, setEpochs] = useState(3);
  const [batchSize, setBatchSize] = useState(4);
  const [lr, setLr] = useState(0.001);
  const [dModel, setDModel] = useState(128);
  const [nLayers, setNLayers] = useState(4);
  const [nHeads, setNHeads] = useState(4);
  const [loraR, setLoraR] = useState(8);
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);

  // Queries
  const { data: datasets } = useQuery({
    queryKey: ['dataset-versions'],
    queryFn: () => api.datasetCompiler.listVersions(),
  });

  const { data: tokenizers } = useQuery({
    queryKey: ['tokenizers-list'],
    queryFn: () => listTokenizers(),
  });

  const { data: jobs, refetch: refetchJobs } = useQuery({
    queryKey: ['training-jobs'],
    queryFn: () => api.training.listJobs(20),
    refetchInterval: 3000,
  });

  const { data: activeJob } = useQuery({
    queryKey: ['training-job', selectedJobId],
    queryFn: () => (selectedJobId ? api.training.getJob(selectedJobId) : null),
    enabled: !!selectedJobId,
    refetchInterval: 2000,
  });

  // Auto-select latest running or completed job if not selected
  useEffect(() => {
    if (jobs && jobs.length > 0 && !selectedJobId) {
      setSelectedJobId(jobs[0].job_id);
    }
  }, [jobs, selectedJobId]);

  // Set default dataset/tokenizer if available
  useEffect(() => {
    if (datasets && datasets.length > 0 && !datasetId) {
      setDatasetId(datasets[0].dataset_id);
    }
    if (tokenizers && tokenizers.length > 0 && !tokenizerId) {
      setTokenizerId(tokenizers[0].tokenizer_id);
    }
  }, [datasets, tokenizers, datasetId, tokenizerId]);

  // Read URL query parameters (e.g. redirected from Dataset Compiler)
  useEffect(() => {
    if (typeof window !== 'undefined') {
      const params = new URLSearchParams(window.location.search);
      const urlDatasetId = params.get('dataset_id');
      const urlDatasetName = params.get('dataset_name');
      const urlJobType = params.get('job_type');
      const urlModelName = params.get('model_name');

      if (urlDatasetId) {
        setDatasetId(urlDatasetId);
      }
      if (urlDatasetName) {
        setJobName(`Eğitim - ${urlDatasetName}`);
        const safeModel = urlDatasetName.toLowerCase().replace(/[^a-z0-9_-]/g, '_');
        setModelName(`${safeModel}-gpt`);
      }
      if (urlJobType === 'LORA_SFT' || urlJobType === 'PRETRAIN') {
        setJobType(urlJobType);
      }
      if (urlModelName) {
        setModelName(urlModelName);
      }
    }
  }, []);

  // Mutations
  const startMutation = useMutation({
    mutationFn: (data: StartTrainingRequest) => api.training.start(data),
    onSuccess: (newJob) => {
      setSelectedJobId(newJob.job_id);
      queryClient.invalidateQueries({ queryKey: ['training-jobs'] });
    },
  });

  const cancelMutation = useMutation({
    mutationFn: (jobId: string) => api.training.cancelJob(jobId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['training-jobs'] });
      queryClient.invalidateQueries({ queryKey: ['training-job', selectedJobId] });
    },
  });

  const handleStartTraining = (e: React.FormEvent) => {
    e.preventDefault();
    startMutation.mutate({
      job_name: jobName,
      model_name: modelName,
      job_type: jobType,
      dataset_id: datasetId || undefined,
      tokenizer_id: tokenizerId || undefined,
      epochs,
      batch_size: batchSize,
      learning_rate: lr,
      d_model: dModel,
      n_layers: nLayers,
      n_heads: nHeads,
      max_seq_len: maxSeqLen,
      device,
      base_model: baseModel || undefined,
      base_version: baseVersion || undefined,
      lora_r: jobType === 'LORA_SFT' ? loraR : undefined,
    });
  };

  const chartData = activeJob?.metrics || [];

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 pb-16">
      {/* Top Navigation */}
      <header className="bg-white border-b border-slate-200 px-6 py-4 flex items-center justify-between sticky top-0 z-30 shadow-sm">
        <div className="flex items-center space-x-3">
          <Link href="/" className="text-xl font-bold text-indigo-600 hover:opacity-80">
            🤖 Local AI Research Lab
          </Link>
          <span className="text-slate-300">/</span>
          <h1 className="text-lg font-semibold text-slate-800">Training Studio (Eğitim Stüdyosu)</h1>
        </div>
        <div className="flex items-center space-x-3">
          <Link
            href="/playground"
            className="px-4 py-2 bg-indigo-50 text-indigo-700 hover:bg-indigo-100 rounded-lg text-sm font-medium transition"
          >
            💬 Model Playground'a Git →
          </Link>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 py-8">
        {/* Educational Banner */}
        <div className="bg-gradient-to-r from-indigo-900 to-indigo-700 text-white rounded-2xl p-6 mb-8 shadow-lg">
          <div className="flex items-start space-x-4">
            <div className="p-3 bg-white/10 rounded-xl">
              <Cpu className="w-8 h-8 text-indigo-200" />
            </div>
            <div>
              <h2 className="text-2xl font-bold mb-2">
                🎓 Açıklama-Önce İlkesi: Model Eğitimi Nasıl Çalışır?
              </h2>
              <p className="text-indigo-100 text-sm leading-relaxed max-w-3xl">
                Dil modeli eğitimi, girdi metinlerindeki tokenlerin ardışık olasılık dağılımını optimize eder.
                <strong> Pretraining</strong> aşamasında model bir sonraki kelimeyi tahmin etmeyi öğrenir (Causal Language Modeling).
                <strong> SFT &amp; LoRA</strong> aşamasında ise taban model dondurulur ve düşük rankli adaptör matrisleri ($W_0 + \frac{'{'}\alpha{'}'}{'{'}r{'}'}BA$) ile modele soru-cevap davranışı kazandırılır.
              </p>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          {/* Sol Kolon: Eğitim Başlatma Formu */}
          <div className="lg:col-span-5 space-y-6">
            <div className="bg-white rounded-xl p-6 border border-slate-200 shadow-sm">
              <h3 className="text-lg font-bold text-slate-900 mb-4 flex items-center space-x-2">
                <Sliders className="w-5 h-5 text-indigo-600" />
                <span>Eğitim Yapılandırması</span>
              </h3>

              <form onSubmit={handleStartTraining} className="space-y-4">
                <div>
                  <label className="block text-xs font-semibold text-slate-600 uppercase mb-1">
                    Eğitim Adı
                  </label>
                  <input
                    type="text"
                    value={jobName}
                    onChange={(e) => setJobName(e.target.value)}
                    required
                    className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm focus:ring-2 focus:ring-indigo-500 focus:outline-none"
                  />
                </div>

                <div>
                  <label className="block text-xs font-semibold text-slate-600 uppercase mb-1">
                    Model Kayıt Adı
                  </label>
                  <input
                    type="text"
                    value={modelName}
                    onChange={(e) => setModelName(e.target.value)}
                    required
                    className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm focus:ring-2 focus:ring-indigo-500 focus:outline-none"
                  />
                </div>

                {/* Eğitim Tipi */}
                <div>
                  <label className="block text-xs font-semibold text-slate-600 uppercase mb-1">
                    Eğitim Yöntemi
                  </label>
                  <div className="grid grid-cols-3 gap-2">
                    <button
                      type="button"
                      onClick={() => setJobType('PRETRAIN')}
                      className={`py-2 px-3 text-xs font-medium rounded-lg border transition ${
                        jobType === 'PRETRAIN'
                          ? 'bg-indigo-600 text-white border-indigo-600 shadow'
                          : 'bg-slate-50 text-slate-700 border-slate-200 hover:bg-slate-100'
                      }`}
                    >
                      Pre-training (Sıfırdan)
                    </button>
                    <button type="button" className="border rounded p-2" onClick={() => setJobType('FULL_SFT')}>Full SFT {jobType === 'FULL_SFT' ? '✓' : ''}</button>
                    <button
                      type="button"
                      onClick={() => setJobType('LORA_SFT')}
                      className={`py-2 px-3 text-xs font-medium rounded-lg border transition ${
                        jobType === 'LORA_SFT'
                          ? 'bg-indigo-600 text-white border-indigo-600 shadow'
                          : 'bg-slate-50 text-slate-700 border-slate-200 hover:bg-slate-100'
                      }`}
                    >
                      LoRA &amp; SFT (İnce Ayar)
                    </button>
                  </div>
                </div>

                {jobType !== 'PRETRAIN' && <div className="space-y-2">
                  <label>Temel model <input required className="border rounded p-2 w-full" value={baseModel} onChange={e => setBaseModel(e.target.value)} /></label>
                  <label>Temel model sürümü <input required className="border rounded p-2 w-full" value={baseVersion} onChange={e => setBaseVersion(e.target.value)} /></label>
                </div>}
                <label>Bağlam uzunluğu <input type="number" min={4} max={8192} value={maxSeqLen} onChange={e => setMaxSeqLen(Number(e.target.value))} className="border rounded p-2 w-full" /></label>
                <label>Cihaz <select value={device} onChange={e => setDevice(e.target.value)} className="border rounded p-2 w-full"><option value="cpu">CPU</option><option value="cuda">CUDA</option><option value="mps">Apple MPS</option></select></label>
                {/* Dataset Seçici */}
                <div>
                  <label className="block text-xs font-semibold text-slate-600 uppercase mb-1">
                    Derlenmiş Dataset
                  </label>
                  <select
                    value={datasetId}
                    onChange={(e) => setDatasetId(e.target.value)}
                    className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm focus:ring-2 focus:ring-indigo-500 bg-white"
                  >
                    <option value="">Varsayılan / Belgeler Veritabanı</option>
                    {datasets?.map((ds: any) => (
                      <option key={ds.dataset_id} value={ds.dataset_id}>
                        {ds.name} (v{ds.version} - {ds.num_documents} belge)
                      </option>
                    ))}
                  </select>
                </div>

                {/* Tokenizer Seçici */}
                <div>
                  <label className="block text-xs font-semibold text-slate-600 uppercase mb-1">
                    Tokenizer
                  </label>
                  <select
                    value={tokenizerId}
                    onChange={(e) => setTokenizerId(e.target.value)}
                    className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm focus:ring-2 focus:ring-indigo-500 bg-white"
                  >
                    <option value="">Tokenizer seçin</option>
                    {tokenizers?.map((tok: any) => (
                      <option key={tok.tokenizer_id} value={tok.tokenizer_id}>
                        {tok.name} ({tok.tokenizer_type} - {tok.vocab_size} vocab)
                      </option>
                    ))}
                  </select>
                </div>

                {/* Hyperparameters Grid */}
                <div className="grid grid-cols-2 gap-3 pt-2">
                  <div>
                    <label className="block text-xs font-medium text-slate-600 mb-1">Epochs</label>
                    <input
                      type="number"
                      value={epochs}
                      min={1}
                      max={50}
                      onChange={(e) => setEpochs(parseInt(e.target.value) || 1)}
                      className="w-full px-3 py-1.5 border border-slate-300 rounded text-sm"
                    />
                  </div>
                  <div>
                    <label className="block text-xs font-medium text-slate-600 mb-1">Batch Size</label>
                    <input
                      type="number"
                      value={batchSize}
                      min={1}
                      max={64}
                      onChange={(e) => setBatchSize(parseInt(e.target.value) || 1)}
                      className="w-full px-3 py-1.5 border border-slate-300 rounded text-sm"
                    />
                  </div>
                  <div>
                    <label className="block text-xs font-medium text-slate-600 mb-1">Learning Rate</label>
                    <input
                      type="number"
                      step="0.0001"
                      value={lr}
                      onChange={(e) => setLr(parseFloat(e.target.value) || 0.001)}
                      className="w-full px-3 py-1.5 border border-slate-300 rounded text-sm"
                    />
                  </div>
                  <div>
                    <label className="block text-xs font-medium text-slate-600 mb-1">Embedding (d_model)</label>
                    <select
                      value={dModel}
                      onChange={(e) => setDModel(parseInt(e.target.value))}
                      className="w-full px-3 py-1.5 border border-slate-300 rounded text-sm bg-white"
                    >
                      <option value={64}>64 (Ultra-Tiny)</option>
                      <option value={128}>128 (Tiny ~800K)</option>
                      <option value={256}>256 (Small ~3M)</option>
                    </select>
                  </div>
                </div>

                {jobType === 'LORA_SFT' && (
                  <div className="p-3 bg-amber-50 rounded-lg border border-amber-200 mt-2">
                    <label className="block text-xs font-bold text-amber-900 uppercase mb-1">
                      LoRA Rank (r)
                    </label>
                    <input
                      type="number"
                      value={loraR}
                      min={2}
                      max={64}
                      onChange={(e) => setLoraR(parseInt(e.target.value) || 8)}
                      className="w-full px-3 py-1.5 border border-amber-300 rounded text-sm bg-white"
                    />
                    <p className="text-[11px] text-amber-700 mt-1">
                      Rank ne kadar küçükse parametre sayısı ve bellek kullanımı o kadar az olur.
                    </p>
                  </div>
                )}

                <button
                  type="submit"
                  disabled={startMutation.isPending}
                  className="w-full py-3 px-4 bg-indigo-600 hover:bg-indigo-700 text-white font-semibold rounded-lg shadow-md transition flex items-center justify-center space-x-2 disabled:opacity-50"
                >
                  <Play className="w-4 h-4 fill-white" />
                  <span>{startMutation.isPending ? 'Başlatılıyor...' : 'Eğitimi Başlat'}</span>
                </button>
              </form>
            </div>

            {/* Geçmiş Eğitim İşleri */}
            <div className="bg-white rounded-xl p-6 border border-slate-200 shadow-sm">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-sm font-bold text-slate-800 uppercase tracking-wider">
                  Eğitim İşleri ({jobs?.length || 0})
                </h3>
                <button
                  onClick={() => refetchJobs()}
                  className="text-slate-400 hover:text-slate-600"
                >
                  <RefreshCw className="w-4 h-4" />
                </button>
              </div>

              <div className="space-y-2 max-h-60 overflow-y-auto pr-1">
                {jobs && jobs.length > 0 ? (
                  jobs.map((j) => (
                    <div
                      key={j.job_id}
                      onClick={() => setSelectedJobId(j.job_id)}
                      className={`p-3 rounded-lg border text-xs cursor-pointer transition flex items-center justify-between ${
                        selectedJobId === j.job_id
                          ? 'border-indigo-500 bg-indigo-50/50'
                          : 'border-slate-200 hover:bg-slate-50'
                      }`}
                    >
                      <div>
                        <p className="font-semibold text-slate-800">{j.job_name}</p>
                        <p className="text-slate-500 text-[11px]">
                          {j.model_name} • {j.job_type}
                        </p>
                      </div>
                      <span
                        className={`px-2 py-0.5 rounded-full font-medium text-[10px] ${
                          j.status === 'COMPLETED'
                            ? 'bg-emerald-100 text-emerald-700'
                            : j.status === 'RUNNING'
                            ? 'bg-blue-100 text-blue-700 animate-pulse'
                            : j.status === 'FAILED'
                            ? 'bg-rose-100 text-rose-700'
                            : 'bg-slate-100 text-slate-600'
                        }`}
                      >
                        {j.status}
                      </span>
                    </div>
                  ))
                ) : (
                  <p className="text-xs text-slate-400 py-4 text-center">Henüz bir eğitim işi yok.</p>
                )}
              </div>
            </div>
          </div>

          {/* Sağ Kolon: Canlı İzleme ve Grafik */}
          <div className="lg:col-span-7 space-y-6">
            {activeJob ? (
              <div className="bg-white rounded-xl p-6 border border-slate-200 shadow-sm space-y-6">
                <div className="flex items-center justify-between border-b border-slate-100 pb-4">
                  <div>
                    <span className="text-xs text-indigo-600 font-bold uppercase tracking-wider">
                      {activeJob.job_id}
                    </span>
                    <h2 className="text-xl font-bold text-slate-900">{activeJob.job_name}</h2>
                    <p className="text-xs text-slate-500 mt-0.5">
                      Model: <strong>{activeJob.model_name}</strong> | Mod: {activeJob.job_type}
                    </p>
                  </div>
                  <div className="flex items-center space-x-2">
                    {['CANCELLED', 'INTERRUPTED', 'FAILED'].includes(activeJob.status) && <button onClick={async () => { await api.training.resumeJob(activeJob.job_id); await refetchJobs(); }} className="border rounded p-2">Checkpoint’ten devam et</button>}
                    {activeJob.error && <p role="alert">{activeJob.error}</p>}
                    {activeJob.status === 'RUNNING' && (
                      <button
                        onClick={() => cancelMutation.mutate(activeJob.job_id)}
                        disabled={cancelMutation.isPending}
                        className="px-3 py-1.5 bg-rose-50 hover:bg-rose-100 text-rose-700 rounded-lg text-xs font-semibold flex items-center space-x-1 border border-rose-200 transition"
                      >
                        <Square className="w-3.5 h-3.5 fill-rose-700" />
                        <span>Durdur</span>
                      </button>
                    )}
                    <span
                      className={`px-3 py-1 rounded-full text-xs font-bold ${
                        activeJob.status === 'COMPLETED'
                          ? 'bg-emerald-100 text-emerald-800'
                          : activeJob.status === 'RUNNING'
                          ? 'bg-blue-100 text-blue-800 animate-pulse'
                          : activeJob.status === 'FAILED'
                          ? 'bg-rose-100 text-rose-800'
                          : 'bg-slate-100 text-slate-700'
                      }`}
                    >
                      {activeJob.status}
                    </span>
                  </div>
                </div>

                {/* Progress Bar */}
                <div>
                  <div className="flex items-center justify-between text-xs font-semibold text-slate-600 mb-1.5">
                    <span>İlerleme ({Math.round(activeJob.progress * 100)}%)</span>
                    <span>
                      Adım: {activeJob.current_step} / {activeJob.total_steps} (Epoch {activeJob.current_epoch}/{activeJob.total_epochs})
                    </span>
                  </div>
                  <div className="w-full bg-slate-100 rounded-full h-3 overflow-hidden">
                    <div
                      className="bg-indigo-600 h-full rounded-full transition-all duration-300"
                      style={{ width: `${Math.min(100, Math.max(0, activeJob.progress * 100))}%` }}
                    />
                  </div>
                </div>

                {/* Live Metric Cards */}
                <div className="grid grid-cols-3 gap-4">
                  <div className="p-4 bg-slate-50 rounded-xl border border-slate-200 text-center">
                    <p className="text-xs text-slate-500 font-medium">Son Kayıp (Loss)</p>
                    <p className="text-2xl font-bold text-slate-900 mt-1">
                      {activeJob.metrics?.length
                        ? activeJob.metrics[activeJob.metrics.length - 1].loss
                        : '-'}
                    </p>
                  </div>
                  <div className="p-4 bg-slate-50 rounded-xl border border-slate-200 text-center">
                    <p className="text-xs text-slate-500 font-medium">Perplexity</p>
                    <p className="text-2xl font-bold text-indigo-600 mt-1">
                      {activeJob.metrics?.length
                        ? activeJob.metrics[activeJob.metrics.length - 1].perplexity || '-'
                        : '-'}
                    </p>
                  </div>
                  <div className="p-4 bg-slate-50 rounded-xl border border-slate-200 text-center">
                    <p className="text-xs text-slate-500 font-medium">Öğrenme Oranı (LR)</p>
                    <p className="text-2xl font-bold text-emerald-600 mt-1">
                      {activeJob.metrics?.length
                        ? activeJob.metrics[activeJob.metrics.length - 1].lr || '-'
                        : '-'}
                    </p>
                  </div>
                </div>

                {/* Canlı Recharts Grafiği */}
                <div className="bg-slate-50 rounded-xl p-4 border border-slate-200">
                  <h4 className="text-xs font-bold text-slate-700 uppercase tracking-wider mb-4">
                    {chartData.length ? 'GERÇEK DENEY — Eğitim kaybı' : 'VERİ BEKLENİYOR — Henüz ölçüm yok'}
                  </h4>
                  <div className="h-64 w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={chartData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                        <XAxis dataKey="step" label={{ value: 'Adım (Step)', position: 'insideBottom', offset: -5 }} />
                        <YAxis label={{ value: 'Loss', angle: -90, position: 'insideLeft' }} />
                        <Tooltip />
                        <Legend />
                        <Line
                          type="monotone"
                          dataKey="loss"
                          stroke="#4f46e5"
                          strokeWidth={2.5}
                          dot={false}
                          name="Loss (Kayıp)"
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                {/* Checkpoint / Kayıt Bilgisi */}
                {activeJob.status === 'COMPLETED' && (
                  <div className="p-4 bg-emerald-50 border border-emerald-200 rounded-xl flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
                    <div className="flex items-center space-x-3">
                      <CheckCircle2 className="w-6 h-6 text-emerald-600 shrink-0" />
                      <div>
                        <p className="text-sm font-bold text-emerald-900">Model Eğitimi Tamamlandı!</p>
                        <p className="text-xs text-emerald-700">
                          Model Registry&apos;ye kaydedildi ({activeJob.model_name}).
                        </p>
                      </div>
                    </div>
                    <div className="flex flex-wrap items-center gap-2">
                      <Link
                        id="test-in-playground-btn"
                        href={`/playground?model=${encodeURIComponent(activeJob.model_name)}&version=${encodeURIComponent(activeJob.config?.model_version || "")}&experiment_id=${activeJob.job_id}&dataset_id=${activeJob.dataset_id || ""}&tokenizer_id=${activeJob.tokenizer_id || ""}`}
                        className="px-3.5 py-2 bg-emerald-600 text-white rounded-lg text-xs font-semibold hover:bg-emerald-700 transition shadow-sm inline-flex items-center gap-1.5"
                      >
                        🎮 Playground&apos;da Test Et ➔
                      </Link>
                      <Link
                        id="evaluate-model-btn"
                        href={`/evaluation?model=${encodeURIComponent(activeJob.model_name)}&version=${encodeURIComponent(activeJob.config?.model_version || "")}&experiment_id=${activeJob.job_id}&dataset_id=${activeJob.dataset_id || ""}&tokenizer_id=${activeJob.tokenizer_id || ""}`}
                        className="px-3.5 py-2 bg-indigo-600 text-white rounded-lg text-xs font-semibold hover:bg-indigo-700 transition shadow-sm inline-flex items-center gap-1.5"
                      >
                        📊 Benchmark &amp; Değerlendir ➔
                      </Link>
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="bg-white rounded-xl p-12 border border-slate-200 text-center shadow-sm">
                <Clock className="w-12 h-12 text-slate-300 mx-auto mb-3" />
                <h3 className="text-base font-bold text-slate-800">Aktif İş Seçilmedi</h3>
                <p className="text-xs text-slate-500 mt-1">
                  Soldaki formdan yeni bir eğitim başlatın veya listeden bir iş seçin.
                </p>
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}
