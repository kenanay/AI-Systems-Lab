'use client';

import React, { useState, useEffect, useMemo, useRef } from 'react';
import Link from 'next/link';
import { useQuery, useMutation } from '@tanstack/react-query';
import {
  Target,
  Compass,
  Layers,
  Sparkles,
  ArrowRight,
  RefreshCw,
  Plus,
  X,
  Sliders,
  Maximize2,
  Minimize2,
  Cpu,
  BarChart3,
  Split,
  ChevronRight,
  ShieldCheck,
  RotateCcw,
  Zap,
  Tag
} from 'lucide-react';
import {
  api,
  EmbeddingProjectRequest,
  EmbeddingProjectResponse,
  ProjectedPoint,
  SimilarityRequest,
  SimilarityResponse,
  AnalogyRequest,
  AnalogyResponse
} from '@/lib/api';

const PRESET_CLUSTERS = [
  {
    name: 'Zıtlıklar & Doğa',
    words: ['sıcak', 'soğuk', 'güneş', 'kar', 'ateş', 'buz', 'yaz', 'kış'],
    color: '#3b82f6',
    desc: 'Zıt kavramlar ve doğa olaylarının vektör ayrışımı'
  },
  {
    name: 'Yapay Zeka & Teknoloji',
    words: ['bilgisayar', 'yazılım', 'veri', 'model', 'algoritma', 'kod', 'robot', 'zeka'],
    color: '#8b5cf6',
    desc: 'Teknolojik ve bilişim terimlerinin yakın kümelenmesi'
  },
  {
    name: 'Duygular',
    words: ['mutluluk', 'neşe', 'hüzün', 'keder', 'öfke', 'sevgi', 'korku', 'umut'],
    color: '#ec4899',
    desc: 'Pozitif ve negatif duygu kutupları'
  },
  {
    name: 'Analoji & Rol',
    words: ['kral', 'kraliçe', 'adam', 'kadın', 'prens', 'prenses', 'lider', 'yönetici'],
    color: '#10b981',
    desc: 'Cinsiyet ve makam eksenlerinde vektör paralellikleri'
  }
];

export default function EmbeddingLabPage() {
  // State for words
  const [selectedWords, setSelectedWords] = useState<string[]>(PRESET_CLUSTERS[0].words);
  const [newWordInput, setNewWordInput] = useState('');
  const [dimensions, setDimensions] = useState<2 | 3>(2);

  // 3D rotation angles for isometric projection
  const [rotX, setRotX] = useState<number>(25); // degrees
  const [rotY, setRotY] = useState<number>(35); // degrees

  // Selected points for distance comparison
  const [selectedPointA, setSelectedPointA] = useState<ProjectedPoint | null>(null);
  const [selectedPointB, setSelectedPointB] = useState<ProjectedPoint | null>(null);
  const [hoveredPoint, setHoveredPoint] = useState<ProjectedPoint | null>(null);

  // Similarity tool state
  const [simWordA, setSimWordA] = useState('sıcak');
  const [simWordB, setSimWordB] = useState('soğuk');

  // Analogy tool state
  const [analogyA, setAnalogyA] = useState('kral');
  const [analogyB, setAnalogyB] = useState('adam');
  const [analogyC, setAnalogyC] = useState('kadın');

  // Projection Mutation
  const projectMutation = useMutation({
    mutationFn: (req: EmbeddingProjectRequest) => api.embeddings.project(req),
  });

  // Similarity Mutation
  const similarityMutation = useMutation({
    mutationFn: (req: SimilarityRequest) => api.embeddings.similarity(req),
  });

  // Analogy Mutation
  const analogyMutation = useMutation({
    mutationFn: (req: AnalogyRequest) => api.embeddings.analogy(req),
  });

  // Execute initial projection on mount
  useEffect(() => {
    projectMutation.mutate({
      words: selectedWords,
      dimensions: dimensions,
      normalize: true,
    });
    similarityMutation.mutate({
      word_a: simWordA,
      word_b: simWordB,
    });
    analogyMutation.mutate({
      word_a: analogyA,
      word_b: analogyB,
      word_c: analogyC,
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleUpdateWords = (words: string[], dim: 2 | 3 = dimensions) => {
    setSelectedWords(words);
    setSelectedPointA(null);
    setSelectedPointB(null);
    projectMutation.mutate({
      words,
      dimensions: dim,
      normalize: true,
    });
  };

  const handleAddWord = (e: React.FormEvent) => {
    e.preventDefault();
    const clean = newWordInput.trim();
    if (!clean || selectedWords.includes(clean)) return;
    const updated = [...selectedWords, clean];
    setNewWordInput('');
    handleUpdateWords(updated);
  };

  const handleRemoveWord = (word: string) => {
    if (selectedWords.length <= 2) return;
    const updated = selectedWords.filter((w) => w !== word);
    handleUpdateWords(updated);
  };

  const handleDimensionToggle = (d: 2 | 3) => {
    setDimensions(d);
    projectMutation.mutate({
      words: selectedWords,
      dimensions: d,
      normalize: true,
    });
  };

  const handlePointClick = (pt: ProjectedPoint) => {
    if (!selectedPointA) {
      setSelectedPointA(pt);
    } else if (!selectedPointB && selectedPointA.text !== pt.text) {
      setSelectedPointB(pt);
      // Run similarity between A and B
      setSimWordA(selectedPointA.text);
      setSimWordB(pt.text);
      similarityMutation.mutate({
        word_a: selectedPointA.text,
        word_b: pt.text,
      });
    } else {
      setSelectedPointA(pt);
      setSelectedPointB(null);
    }
  };

  const handleRunSimilarity = (e: React.FormEvent) => {
    e.preventDefault();
    if (!simWordA.trim() || !simWordB.trim()) return;
    similarityMutation.mutate({
      word_a: simWordA.trim(),
      word_b: simWordB.trim(),
    });
  };

  const handleRunAnalogy = (e: React.FormEvent) => {
    e.preventDefault();
    if (!analogyA.trim() || !analogyB.trim() || !analogyC.trim()) return;
    analogyMutation.mutate({
      word_a: analogyA.trim(),
      word_b: analogyB.trim(),
      word_c: analogyC.trim(),
    });
  };

  // SVG Projection Coordinates mapping
  const points = projectMutation.data?.points || [];
  const explainedVariance = projectMutation.data?.explained_variance_ratio || [];
  const dModel = projectMutation.data?.d_model || 64;
  const modelName = projectMutation.data?.model_name || 'GPT-Model';

  // SVG canvas center and scale
  const SVG_WIDTH = 640;
  const SVG_HEIGHT = 440;
  const CX = SVG_WIDTH / 2;
  const CY = SVG_HEIGHT / 2;

  // Calculate 2D or 3D projected screen coordinates
  const screenPoints = useMemo(() => {
    const radX = (rotX * Math.PI) / 180;
    const radY = (rotY * Math.PI) / 180;

    return points.map((p) => {
      let screenX = 0;
      let screenY = 0;

      if (dimensions === 2 || p.z === null || p.z === undefined) {
        // Direct 2D scale: mapped from [-100, 100] to canvas
        screenX = CX + (p.x / 100) * (SVG_WIDTH * 0.38);
        screenY = CY - (p.y / 100) * (SVG_HEIGHT * 0.38);
      } else {
        // 3D Isometric projection
        const x3 = p.x;
        const y3 = p.y;
        const z3 = p.z;

        // Rotate around Y axis
        const xRotY = x3 * Math.cos(radY) + z3 * Math.sin(radY);
        const zRotY = -x3 * Math.sin(radY) + z3 * Math.cos(radY);

        // Rotate around X axis
        const yRotX = y3 * Math.cos(radX) - zRotY * Math.sin(radX);
        const zFinal = y3 * Math.sin(radX) + zRotY * Math.cos(radX);

        screenX = CX + (xRotY / 100) * (SVG_WIDTH * 0.32);
        screenY = CY - (yRotX / 100) * (SVG_HEIGHT * 0.32);
      }

      return {
        ...p,
        screenX,
        screenY,
      };
    });
  }, [points, dimensions, rotX, rotY, CX, CY]);

  // Between Point A and B on screen
  const screenPointA = screenPoints.find((p) => p.text === selectedPointA?.text);
  const screenPointB = screenPoints.find((p) => p.text === selectedPointB?.text);

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100 pb-20">
      {/* Top Sticky Header */}
      <div className="border-b border-slate-800 bg-slate-950/80 backdrop-blur-md sticky top-16 z-40">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-5">
          <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
            <div>
              <div className="flex items-center space-x-3">
                <span className="p-2.5 rounded-xl bg-cyan-500/20 text-cyan-400 border border-cyan-500/30 text-2xl">
                  🎯
                </span>
                <div>
                  <div className="flex items-center space-x-2.5">
                    <h1 className="text-2xl font-black tracking-tight text-white">
                      Embedding Lab
                    </h1>
                    <span className="px-2 py-0.5 rounded-full text-xs font-semibold bg-cyan-500/20 text-cyan-300 border border-cyan-500/40">
                      PCA &amp; Vektör Geometrisi
                    </span>
                  </div>
                  <p className="text-sm text-slate-400 mt-0.5">
                    Transformer token embedding uzayını 2D/3D SVD projeksiyonu, Cosine benzerliği ve vektör cebiriyle keşfet.
                  </p>
                </div>
              </div>
            </div>

            {/* Model & Dimension Tag */}
            <div className="flex items-center space-x-2.5">
              <div className="px-3 py-1.5 rounded-lg bg-slate-800/80 border border-slate-700 text-xs font-mono text-slate-300 flex items-center space-x-2">
                <Cpu className="w-4 h-4 text-cyan-400" />
                <span>Model: <strong className="text-cyan-300">{modelName}</strong></span>
              </div>
              <div className="px-3 py-1.5 rounded-lg bg-slate-800/80 border border-slate-700 text-xs font-mono text-slate-300">
                <span>d_model: <strong className="text-white">{dModel}D</strong></span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-8 space-y-8">
        {/* Preset Clusters & Custom Word Tag Cloud */}
        <div className="bg-slate-800/60 border border-slate-700/80 rounded-2xl p-6 shadow-xl backdrop-blur-sm space-y-4">
          {/* Preset Buttons */}
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-xs font-bold text-slate-400 uppercase tracking-wider flex items-center gap-1.5 mr-2">
              <Tag className="w-3.5 h-3.5 text-cyan-400" />
              Hazır Kümeler:
            </span>
            {PRESET_CLUSTERS.map((preset, idx) => (
              <button
                key={idx}
                type="button"
                onClick={() => handleUpdateWords(preset.words)}
                className="text-xs px-3.5 py-1.5 rounded-lg bg-slate-900/80 border border-slate-700 hover:border-cyan-500/60 text-slate-300 transition-all hover:bg-slate-800 flex items-center space-x-1.5"
              >
                <span className="w-2 h-2 rounded-full" style={{ backgroundColor: preset.color }} />
                <span className="font-semibold text-white">{preset.name}</span>
                <span className="text-[10px] text-slate-500">({preset.words.length})</span>
              </button>
            ))}
          </div>

          {/* Active Words Tag Cloud & Add New Word */}
          <div className="pt-2 flex flex-wrap items-center gap-2">
            <span className="text-xs text-slate-400 font-medium mr-1">Aktif Kelimeler:</span>
            {selectedWords.map((word) => (
              <span
                key={word}
                className="inline-flex items-center space-x-1 px-3 py-1 rounded-full text-xs font-mono bg-cyan-500/10 text-cyan-300 border border-cyan-500/30"
              >
                <span>{word}</span>
                {selectedWords.length > 2 && (
                  <button
                    type="button"
                    onClick={() => handleRemoveWord(word)}
                    className="hover:text-red-400 transition-colors ml-1"
                    title="Kelimeyi çıkar"
                  >
                    <X className="w-3 h-3" />
                  </button>
                )}
              </span>
            ))}

            {/* Inline Input to Add Word */}
            <form onSubmit={handleAddWord} className="inline-flex items-center space-x-1.5">
              <input
                type="text"
                value={newWordInput}
                onChange={(e) => setNewWordInput(e.target.value)}
                placeholder="+ Kelime ekle..."
                className="px-3 py-1 text-xs rounded-full bg-slate-900/90 border border-slate-700 text-white placeholder-slate-500 focus:outline-none focus:ring-1 focus:ring-cyan-500 w-32"
              />
              <button
                type="submit"
                disabled={!newWordInput.trim()}
                className="p-1 rounded-full bg-cyan-600 hover:bg-cyan-500 text-white disabled:opacity-40 transition-colors"
                title="Ekle"
              >
                <Plus className="w-3.5 h-3.5" />
              </button>
            </form>
          </div>
        </div>

        {/* Main Workspace: 2D/3D Scatter Plot & Side Inspector */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          {/* Scatter Plot Stage */}
          <div className="lg:col-span-8 bg-slate-800/80 border border-slate-700 rounded-2xl p-6 shadow-2xl space-y-4">
            {/* Canvas Toolbar */}
            <div className="flex flex-wrap items-center justify-between gap-4 pb-3 border-b border-slate-700">
              <div className="flex items-center space-x-2">
                <h2 className="text-base font-bold text-white flex items-center space-x-2">
                  <Compass className="w-4 h-4 text-cyan-400" />
                  <span>PCA Projeksiyon Uzayı</span>
                </h2>
                <span className="text-xs px-2 py-0.5 rounded bg-slate-700 text-slate-300 font-mono">
                  {dimensions}D İzdüşüm
                </span>
              </div>

              {/* 2D / 3D Dimension Selector */}
              <div className="flex items-center space-x-2">
                <div className="flex items-center bg-slate-900/90 p-1 rounded-lg border border-slate-800 text-xs">
                  <button
                    type="button"
                    onClick={() => handleDimensionToggle(2)}
                    className={`px-3 py-1 rounded font-medium transition-all ${
                      dimensions === 2
                        ? 'bg-cyan-600 text-white shadow-sm'
                        : 'text-slate-400 hover:text-white'
                    }`}
                  >
                    2D Düzlem (X, Y)
                  </button>
                  <button
                    type="button"
                    onClick={() => handleDimensionToggle(3)}
                    className={`px-3 py-1 rounded font-medium transition-all ${
                      dimensions === 3
                        ? 'bg-cyan-600 text-white shadow-sm'
                        : 'text-slate-400 hover:text-white'
                    }`}
                  >
                    3D İzometrik (X, Y, Z)
                  </button>
                </div>

                <button
                  type="button"
                  onClick={() => {
                    setSelectedPointA(null);
                    setSelectedPointB(null);
                  }}
                  className="px-2.5 py-1.5 rounded-lg bg-slate-900/80 border border-slate-700 text-slate-400 hover:text-white text-xs flex items-center gap-1"
                  title="Seçimleri Sıfırla"
                >
                  <RotateCcw className="w-3.5 h-3.5" />
                  <span>Sıfırla</span>
                </button>
              </div>
            </div>

            {/* 3D Rotation Sliders if 3D is active */}
            {dimensions === 3 && (
              <div className="flex items-center space-x-6 px-4 py-2 rounded-xl bg-slate-900/80 border border-slate-800 text-xs text-slate-400">
                <div className="flex items-center space-x-2 flex-1">
                  <span>Döndür X:</span>
                  <input
                    type="range"
                    min="0"
                    max="90"
                    value={rotX}
                    onChange={(e) => setRotX(Number(e.target.value))}
                    className="w-full accent-cyan-500"
                  />
                  <span className="font-mono text-cyan-300 w-8">{rotX}°</span>
                </div>
                <div className="flex items-center space-x-2 flex-1">
                  <span>Döndür Y:</span>
                  <input
                    type="range"
                    min="0"
                    max="360"
                    value={rotY}
                    onChange={(e) => setRotY(Number(e.target.value))}
                    className="w-full accent-cyan-500"
                  />
                  <span className="font-mono text-cyan-300 w-8">{rotY}°</span>
                </div>
              </div>
            )}

            {/* SVG Visualizer Canvas */}
            <div className="relative w-full aspect-[16/11] bg-slate-950/90 rounded-xl border border-slate-800 overflow-hidden flex items-center justify-center select-none">
              {projectMutation.isPending ? (
                <div className="flex flex-col items-center space-y-3 text-slate-400 text-xs">
                  <RefreshCw className="w-6 h-6 animate-spin text-cyan-400" />
                  <span>Embedding vektörleri SVD ile izdüşürülüyor...</span>
                </div>
              ) : (
                <svg
                  viewBox={`0 0 ${SVG_WIDTH} ${SVG_HEIGHT}`}
                  className="w-full h-full"
                >
                  {/* Grid Lines & Origin Axes */}
                  <line
                    x1="0"
                    y1={CY}
                    x2={SVG_WIDTH}
                    y2={CY}
                    stroke="#334155"
                    strokeDasharray="4 4"
                    strokeWidth="1"
                  />
                  <line
                    x1={CX}
                    y1="0"
                    x2={CX}
                    y2={SVG_HEIGHT}
                    stroke="#334155"
                    strokeDasharray="4 4"
                    strokeWidth="1"
                  />
                  <text x={SVG_WIDTH - 30} y={CY - 8} fill="#64748b" fontSize="10" fontFamily="monospace">
                    PC1
                  </text>
                  <text x={CX + 8} y="20" fill="#64748b" fontSize="10" fontFamily="monospace">
                    PC2
                  </text>

                  {/* Connecting line between Point A and Point B */}
                  {screenPointA && screenPointB && (
                    <g>
                      <line
                        x1={screenPointA.screenX}
                        y1={screenPointA.screenY}
                        x2={screenPointB.screenX}
                        y2={screenPointB.screenY}
                        stroke="#06b6d4"
                        strokeWidth="2.5"
                        strokeDasharray="5 3"
                      />
                      {/* Midpoint Distance Badge */}
                      <circle
                        cx={(screenPointA.screenX + screenPointB.screenX) / 2}
                        cy={(screenPointA.screenY + screenPointB.screenY) / 2}
                        r="12"
                        fill="#0e7490"
                      />
                      <text
                        x={(screenPointA.screenX + screenPointB.screenX) / 2}
                        y={(screenPointA.screenY + screenPointB.screenY) / 2 + 3}
                        textAnchor="middle"
                        fill="#ffffff"
                        fontSize="8"
                        fontWeight="bold"
                        fontFamily="monospace"
                      >
                        {similarityMutation.data ? `${similarityMutation.data.angle_degrees}°` : 'Δ'}
                      </text>
                    </g>
                  )}

                  {/* Render Points */}
                  {screenPoints.map((pt, i) => {
                    const isA = selectedPointA?.text === pt.text;
                    const isB = selectedPointB?.text === pt.text;
                    const isHovered = hoveredPoint?.text === pt.text;

                    // Color palette
                    const pointColor = isA
                      ? '#38bdf8'
                      : isB
                      ? '#f43f5e'
                      : isHovered
                      ? '#a855f7'
                      : '#06b6d4';

                    return (
                      <g
                        key={pt.text}
                        className="cursor-pointer transition-transform duration-200"
                        onClick={() => handlePointClick(pt)}
                        onMouseEnter={() => setHoveredPoint(pt)}
                        onMouseLeave={() => setHoveredPoint(null)}
                      >
                        {/* Glow halo */}
                        {(isA || isB || isHovered) && (
                          <circle
                            cx={pt.screenX}
                            cy={pt.screenY}
                            r="16"
                            fill={pointColor}
                            opacity="0.25"
                            className="animate-pulse"
                          />
                        )}

                        {/* Point Circle */}
                        <circle
                          cx={pt.screenX}
                          cy={pt.screenY}
                          r={isA || isB ? 8 : 6}
                          fill={pointColor}
                          stroke="#0f172a"
                          strokeWidth="2"
                        />

                        {/* Label Badge */}
                        <rect
                          x={pt.screenX + 10}
                          y={pt.screenY - 10}
                          width={pt.text.length * 7.2 + 14}
                          height="18"
                          rx="4"
                          fill="#0f172a"
                          fillOpacity="0.85"
                          stroke={isA || isB ? pointColor : '#334155'}
                          strokeWidth="1"
                        />
                        <text
                          x={pt.screenX + 17}
                          y={pt.screenY + 3}
                          fill={isA || isB ? '#ffffff' : '#cbd5e1'}
                          fontSize="11"
                          fontWeight={isA || isB ? 'bold' : 'normal'}
                          fontFamily="sans-serif"
                        >
                          {pt.text}
                        </text>
                      </g>
                    );
                  })}
                </svg>
              )}
            </div>

            {/* Explained Variance Ratio Bar */}
            <div className="p-4 rounded-xl bg-slate-900/80 border border-slate-800 space-y-2">
              <div className="flex items-center justify-between text-xs">
                <span className="font-bold text-slate-300 flex items-center gap-1.5">
                  <BarChart3 className="w-4 h-4 text-cyan-400" />
                  Açıklanan Varyans (PCA Kalitesi):
                </span>
                <span className="font-mono text-cyan-300">
                  Toplam Korunan Varyans: %
                  {(
                    explainedVariance.reduce((acc, v) => acc + v, 0) * 100
                  ).toFixed(1)}
                </span>
              </div>
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 pt-1">
                {explainedVariance.map((ratio, idx) => (
                  <div key={idx} className="space-y-1">
                    <div className="flex justify-between text-[11px] text-slate-400">
                      <span>Bileşen {idx + 1} (PC{idx + 1}):</span>
                      <strong className="text-white font-mono">%{(ratio * 100).toFixed(1)}</strong>
                    </div>
                    <div className="w-full h-1.5 rounded-full bg-slate-800 overflow-hidden">
                      <div
                        className="h-full bg-gradient-to-r from-cyan-500 to-blue-500 rounded-full"
                        style={{ width: `${ratio * 100}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Right Column: Point Details & Cosine Similarity Calculator */}
          <div className="lg:col-span-4 space-y-6">
            {/* Point Inspector Card */}
            <div className="bg-slate-800/80 border border-slate-700 rounded-2xl p-6 shadow-xl space-y-4">
              <h3 className="text-base font-bold text-white flex items-center space-x-2">
                <Target className="w-4 h-4 text-cyan-400" />
                <span>Nokta &amp; Mesafe İnceleyici</span>
              </h3>

              {selectedPointA ? (
                <div className="space-y-3">
                  <div className="p-3 rounded-lg bg-sky-500/10 border border-sky-500/30 text-xs">
                    <div className="flex justify-between items-center mb-1">
                      <span className="text-sky-300 font-bold">Nokta A (Mavi):</span>
                      <span className="text-white font-mono font-bold text-sm">
                        &quot;{selectedPointA.text}&quot;
                      </span>
                    </div>
                    <div className="font-mono text-[11px] text-slate-400">
                      Koordinat: ({selectedPointA.x}, {selectedPointA.y}
                      {selectedPointA.z !== null ? `, ${selectedPointA.z}` : ''}) • Norm: {selectedPointA.norm}
                    </div>
                  </div>

                  {selectedPointB ? (
                    <div className="p-3 rounded-lg bg-rose-500/10 border border-rose-500/30 text-xs">
                      <div className="flex justify-between items-center mb-1">
                        <span className="text-rose-300 font-bold">Nokta B (Kırmızı):</span>
                        <span className="text-white font-mono font-bold text-sm">
                          &quot;{selectedPointB.text}&quot;
                        </span>
                      </div>
                      <div className="font-mono text-[11px] text-slate-400">
                        Koordinat: ({selectedPointB.x}, {selectedPointB.y}
                        {selectedPointB.z !== null ? `, ${selectedPointB.z}` : ''}) • Norm: {selectedPointB.norm}
                      </div>
                    </div>
                  ) : (
                    <p className="text-xs text-slate-500 italic">
                      Karşılaştırmak için grafikte ikinci bir noktaya tıklayın.
                    </p>
                  )}

                  {/* Direct Computed Similarity between A and B */}
                  {similarityMutation.data && selectedPointB && (
                    <div className="p-4 rounded-xl bg-slate-900 border border-cyan-500/30 space-y-2 pt-3">
                      <div className="flex justify-between items-center text-xs">
                        <span className="text-slate-400">Cosine Benzerliği:</span>
                        <span className="text-cyan-300 font-bold font-mono text-sm">
                          {similarityMutation.data.cosine_similarity.toFixed(4)}
                        </span>
                      </div>
                      <div className="flex justify-between items-center text-xs">
                        <span className="text-slate-400">Açısal Fark (θ):</span>
                        <span className="text-amber-300 font-bold font-mono">
                          {similarityMutation.data.angle_degrees}°
                        </span>
                      </div>
                      <div className="flex justify-between items-center text-xs">
                        <span className="text-slate-400">Öklid Mesafesi:</span>
                        <span className="text-slate-300 font-mono">
                          {similarityMutation.data.euclidean_distance.toFixed(4)}
                        </span>
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <div className="text-center py-8 text-slate-500 text-xs space-y-2">
                  <Compass className="w-8 h-8 mx-auto text-slate-600 opacity-60" />
                  <p>Vektör detaylarını ve koordinatlarını görmek için grafikteki herhangi bir noktaya tıklayın.</p>
                </div>
              )}
            </div>

            {/* Quick Cosine Similarity Tool */}
            <div className="bg-slate-800/80 border border-slate-700 rounded-2xl p-6 shadow-xl space-y-4">
              <h3 className="text-base font-bold text-white flex items-center space-x-2">
                <Zap className="w-4 h-4 text-amber-400" />
                <span>Cosine Benzerlik Ölçer</span>
              </h3>

              <form onSubmit={handleRunSimilarity} className="space-y-3">
                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <label className="text-[11px] text-slate-400 block mb-1">Kelime A</label>
                    <input
                      type="text"
                      value={simWordA}
                      onChange={(e) => setSimWordA(e.target.value)}
                      className="w-full px-3 py-1.5 text-xs rounded-lg bg-slate-900 border border-slate-700 text-white focus:ring-1 focus:ring-amber-500"
                    />
                  </div>
                  <div>
                    <label className="text-[11px] text-slate-400 block mb-1">Kelime B</label>
                    <input
                      type="text"
                      value={simWordB}
                      onChange={(e) => setSimWordB(e.target.value)}
                      className="w-full px-3 py-1.5 text-xs rounded-lg bg-slate-900 border border-slate-700 text-white focus:ring-1 focus:ring-amber-500"
                    />
                  </div>
                </div>

                <button
                  type="submit"
                  disabled={similarityMutation.isPending}
                  className="w-full py-2 rounded-lg bg-amber-600 hover:bg-amber-500 text-white text-xs font-semibold flex items-center justify-center space-x-1.5 transition-colors"
                >
                  {similarityMutation.isPending ? (
                    <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                  ) : (
                    <span>Benzerliği Hesapla</span>
                  )}
                </button>
              </form>

              {similarityMutation.data && (
                <div className="pt-2 border-t border-slate-700/60 space-y-2">
                  <div className="flex justify-between items-center text-xs">
                    <span className="text-slate-400">Benzerlik:</span>
                    <strong className="text-amber-400 font-mono text-base">
                      {(similarityMutation.data.cosine_similarity * 100).toFixed(1)}%
                    </strong>
                  </div>
                  {/* Gauge Bar */}
                  <div className="w-full h-2.5 rounded-full bg-slate-900 overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-red-500 via-amber-500 to-emerald-500 rounded-full transition-all duration-500"
                      style={{
                        width: `${Math.max(
                          0,
                          Math.min(100, (similarityMutation.data.cosine_similarity + 1) * 50)
                        )}%`,
                      }}
                    />
                  </div>
                  <div className="flex justify-between text-[10px] text-slate-500 font-mono">
                    <span>-1.0 (Zıt)</span>
                    <span>0.0 (Dik/Alakasız)</span>
                    <span>+1.0 (Özdeş)</span>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Vector Analogy Section (Semantic Arithmetic) */}
        <div className="bg-slate-800/80 border border-slate-700 rounded-2xl p-6 shadow-2xl space-y-6">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-4 border-b border-slate-700">
            <div>
              <h3 className="text-lg font-bold text-white flex items-center space-x-2">
                <Sparkles className="w-5 h-5 text-emerald-400" />
                <span>Vektör Aritmetiği &amp; Anlamsal Analoji</span>
              </h3>
              <p className="text-xs text-slate-400 mt-0.5">
                Vektör uzayında anlamsal çıkarım: v_A - v_B + v_C ≈ v_hedef
              </p>
            </div>
            <div className="text-xs text-emerald-400 font-mono bg-emerald-500/10 px-3 py-1.5 rounded-lg border border-emerald-500/20">
              kral - adam + kadın = ?
            </div>
          </div>

          <form onSubmit={handleRunAnalogy} className="grid grid-cols-1 sm:grid-cols-4 gap-3 items-end">
            <div>
              <label className="text-xs text-slate-400 block mb-1">Kelime A (Pozitif)</label>
              <input
                type="text"
                value={analogyA}
                onChange={(e) => setAnalogyA(e.target.value)}
                placeholder="kral"
                className="w-full px-3 py-2 text-sm rounded-xl bg-slate-900 border border-slate-700 text-white"
              />
            </div>
            <div>
              <label className="text-xs text-slate-400 block mb-1">Kelime B (Çıkarılan)</label>
              <input
                type="text"
                value={analogyB}
                onChange={(e) => setAnalogyB(e.target.value)}
                placeholder="adam"
                className="w-full px-3 py-2 text-sm rounded-xl bg-slate-900 border border-slate-700 text-white"
              />
            </div>
            <div>
              <label className="text-xs text-slate-400 block mb-1">Kelime C (Eklenen)</label>
              <input
                type="text"
                value={analogyC}
                onChange={(e) => setAnalogyC(e.target.value)}
                placeholder="kadın"
                className="w-full px-3 py-2 text-sm rounded-xl bg-slate-900 border border-slate-700 text-white"
              />
            </div>
            <div>
              <button
                type="submit"
                disabled={analogyMutation.isPending}
                className="w-full py-2.5 rounded-xl bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white font-semibold text-sm flex items-center justify-center space-x-2 transition-all shadow-lg shadow-emerald-600/20"
              >
                {analogyMutation.isPending ? (
                  <RefreshCw className="w-4 h-4 animate-spin" />
                ) : (
                  <>
                    <Sparkles className="w-4 h-4" />
                    <span>Hedefi Bul</span>
                  </>
                )}
              </button>
            </div>
          </form>

          {/* Analogy Results */}
          {analogyMutation.data && (
            <div className="space-y-3 pt-2">
              <span className="text-xs font-bold text-slate-400 uppercase tracking-wider">
                En Yakın Vektör Eşleşmeleri:
              </span>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
                {analogyMutation.data.top_matches.map((item, idx) => (
                  <div
                    key={idx}
                    className="p-3.5 rounded-xl bg-slate-900 border border-slate-800 flex items-center justify-between"
                  >
                    <div className="flex items-center space-x-2.5">
                      <span className="w-6 h-6 rounded-full bg-emerald-500/20 text-emerald-400 flex items-center justify-center text-xs font-bold font-mono">
                        {idx + 1}
                      </span>
                      <strong className="text-white text-sm font-mono">&quot;{item.word}&quot;</strong>
                    </div>
                    <div className="text-right">
                      <span className="text-xs font-mono text-emerald-400 font-bold">
                        {(item.similarity * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Pedagogical Explanation-First Cards */}
        <div className="pt-8 border-t border-slate-800">
          <div className="text-center mb-8">
            <h3 className="text-2xl font-bold text-white flex items-center justify-center space-x-2">
              <span>🎓 Açıklama-Önce (Explanation-First): Embedding &amp; Vektör Uzayı Rehberi</span>
            </h3>
            <p className="text-sm text-slate-400 mt-2 max-w-2xl mx-auto">
              Kelimelerin bilgisayarlar tarafından anlaşılabilir geometrik vektörlere dönüşümü ve uzaysal analoji:
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {/* Card 1: Embedding Nedir? */}
            <div className="bg-slate-800/80 border border-slate-700 rounded-2xl p-6 space-y-4 hover:border-cyan-500/50 transition-all">
              <div className="w-10 h-10 rounded-xl bg-cyan-500/20 text-cyan-400 flex items-center justify-center font-bold">
                1
              </div>
              <h4 className="text-base font-bold text-white">Embedding Nedir?</h4>
              <div className="p-3 bg-slate-900 rounded-xl border border-slate-800 font-mono text-xs text-cyan-300 text-center">
                W: [vocab_size, d_model]
              </div>
              <p className="text-xs text-slate-300 leading-relaxed">
                Kelimeleri seyrek (sparse) one-hot vektörler yerine d-boyutlu sürekli bir uzayda temsil eder. Benzer anlama sahip kelimeler bu uzayda birbirine yakın noktalara yerleşir.
              </p>
            </div>

            {/* Card 2: Cosine Benzerliği */}
            <div className="bg-slate-800/80 border border-slate-700 rounded-2xl p-6 space-y-4 hover:border-amber-500/50 transition-all">
              <div className="w-10 h-10 rounded-xl bg-amber-500/20 text-amber-400 flex items-center justify-center font-bold">
                2
              </div>
              <h4 className="text-base font-bold text-white">Cosine Benzerliği</h4>
              <div className="p-3 bg-slate-900 rounded-xl border border-slate-800 font-mono text-xs text-amber-300 text-center">
                cos(θ) = (u · v) / (||u|| ||v||)
              </div>
              <p className="text-xs text-slate-300 leading-relaxed">
                Vektörlerin mutlak uzunluğuna değil, aralarındaki açıya odaklanır. 1.0 tam paralellik ve aynı anlamı, 0.0 diklik ve bağımsızlığı, -1.0 ise zıt yönelimleri ifade eder.
              </p>
            </div>

            {/* Card 3: PCA ile Boyut İndirgeme */}
            <div className="bg-slate-800/80 border border-slate-700 rounded-2xl p-6 space-y-4 hover:border-purple-500/50 transition-all">
              <div className="w-10 h-10 rounded-xl bg-purple-500/20 text-purple-400 flex items-center justify-center font-bold">
                3
              </div>
              <h4 className="text-base font-bold text-white">PCA (SVD) İzdüşümü</h4>
              <div className="p-3 bg-slate-900 rounded-xl border border-slate-800 font-mono text-xs text-purple-300 text-center">
                X_centered = U · Σ · Vᵀ
              </div>
              <p className="text-xs text-slate-300 leading-relaxed">
                Modelin 64 veya 256 boyutlu uzayındaki bilgiyi en yüksek varyansı koruyan ilk 2 veya 3 ana eksene (Principal Components) izdüşürerek insan gözünün algılayabileceği hale getirir.
              </p>
            </div>

            {/* Card 4: Vektör Aritmetiği */}
            <div className="bg-slate-800/80 border border-slate-700 rounded-2xl p-6 space-y-4 hover:border-emerald-500/50 transition-all">
              <div className="w-10 h-10 rounded-xl bg-emerald-500/20 text-emerald-400 flex items-center justify-center font-bold">
                4
              </div>
              <h4 className="text-base font-bold text-white">Anlamsal Analoji</h4>
              <div className="p-3 bg-slate-900 rounded-xl border border-slate-800 font-mono text-xs text-emerald-300 text-center">
                v_kral - v_adam + v_kadın ≈ v_kraliçe
              </div>
              <p className="text-xs text-slate-300 leading-relaxed">
                Embedding uzayındaki yönler anlamsal ilişkileri (cinsiyet, zaman, çoğulluk) kodlar. Vektör çıkarma ve toplama işlemleriyle anlamsal çıkarım ve mantık yürütme yapılabilir.
              </p>
            </div>
          </div>
        </div>

        {/* Developer Credit */}
        <div className="pt-8 text-center border-t border-slate-800/80">
          <p className="text-xs text-slate-500">
            Local AI Research Lab • Düzenleyen ve Geliştiren: <strong className="text-slate-300">Kenan AY</strong>
          </p>
        </div>
      </div>
    </div>
  );
}
