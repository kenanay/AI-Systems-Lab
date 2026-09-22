'use client';

import React, { useState, useEffect, useMemo } from 'react';
import Link from 'next/link';
import { useQuery, useMutation } from '@tanstack/react-query';
import {
  Eye,
  Layers,
  Sparkles,
  HelpCircle,
  ArrowRight,
  Info,
  ShieldCheck,
  Split,
  ChevronRight,
  Cpu,
  BarChart3,
  Sliders,
  CheckCircle2,
  RefreshCw,
  Maximize2
} from 'lucide-react';
import { api, AttentionInspectRequest, AttentionInspectResponse } from '@/lib/api';

const SAMPLE_PRESETS = [
  {
    label: 'Yapay Zeka',
    text: 'Yapay zeka sistemleri öğrenir ve üretir.',
    desc: 'Temel kavramlar ve fiil-nesne dikkat ilişkisi'
  },
  {
    label: 'Transformer & Attention',
    text: 'Transformer mimarisinde attention mekanizması kilit rol oynar.',
    desc: 'Teknik terimler arası yüksek korelasyon'
  },
  {
    label: 'Bağlamsal Zamir',
    text: 'Öğrenci kütüphanede ders çalışıyordu çünkü sınavı vardı.',
    desc: 'Zamir ve bağlaçların önceki isimlerle ilişkisi'
  },
  {
    label: 'Derin Öğrenme',
    text: 'Derin öğrenme modelleri dil verisi üzerinde eğitilir.',
    desc: 'Sıfat tamlamaları ve edat öbekleri'
  }
];

export default function AttentionLabPage() {
  const [inputText, setInputText] = useState('Transformer mimarisinde attention mekanizması kilit rol oynar.');
  const [selectedLayer, setSelectedLayer] = useState<number | null>(null); // null = avg
  const [selectedHead, setSelectedHead] = useState<number | null>(null); // null = avg
  const [focusedQueryIdx, setFocusedQueryIdx] = useState<number | null>(null);
  const [hoveredCell, setHoveredCell] = useState<{ qIdx: number; kIdx: number; val: number } | null>(null);
  const [viewMode, setViewMode] = useState<'single' | 'multi'>('single');
  const [colorPalette, setColorPalette] = useState<'indigo' | 'emerald' | 'amber'>('indigo');

  // Mutation for fetching attention weights
  const inspectMutation = useMutation({
    mutationFn: (req: AttentionInspectRequest) => api.inference.attention(req),
  });

  // Execute initial inspection on mount
  useEffect(() => {
    inspectMutation.mutate({
      text: inputText,
      layer_idx: selectedLayer ?? undefined,
      head_idx: selectedHead ?? undefined,
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // When layer or head changes and we already have data, re-fetch
  const handleLayerSelect = (l: number | null) => {
    setSelectedLayer(l);
    inspectMutation.mutate({
      text: inputText,
      layer_idx: l ?? undefined,
      head_idx: selectedHead ?? undefined,
    });
  };

  const handleHeadSelect = (h: number | null) => {
    setSelectedHead(h);
    inspectMutation.mutate({
      text: inputText,
      layer_idx: selectedLayer ?? undefined,
      head_idx: h ?? undefined,
    });
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputText.trim()) return;
    setFocusedQueryIdx(null);
    setHoveredCell(null);
    inspectMutation.mutate({
      text: inputText,
      layer_idx: selectedLayer ?? undefined,
      head_idx: selectedHead ?? undefined,
    });
  };

  const handleApplyPreset = (text: string) => {
    setInputText(text);
    setFocusedQueryIdx(null);
    setHoveredCell(null);
    inspectMutation.mutate({
      text,
      layer_idx: selectedLayer ?? undefined,
      head_idx: selectedHead ?? undefined,
    });
  };

  const data = inspectMutation.data;
  const isLoading = inspectMutation.isPending;

  // Active matrix: data.matrix is a 2D array [seq_len, seq_len]
  const matrix = data?.matrix || [];
  const tokens = data?.tokens || [];
  const numLayers = data?.num_layers || 4;
  const numHeads = data?.num_heads || 4;
  const allHeadsMatrix = data?.all_heads_matrix || {};

  // Color generator based on weight and palette
  const getCellColor = (val: number, isMasked: boolean) => {
    if (isMasked) {
      return 'bg-gray-100 text-gray-300 border-gray-200/50';
    }
    const pct = Math.min(100, Math.max(0, val * 100));

    if (colorPalette === 'indigo') {
      if (pct > 60) return 'bg-indigo-600 text-white font-bold';
      if (pct > 35) return 'bg-indigo-500 text-white font-medium';
      if (pct > 20) return 'bg-indigo-400 text-white';
      if (pct > 10) return 'bg-indigo-200 text-indigo-950';
      if (pct > 3) return 'bg-indigo-100 text-indigo-900';
      return 'bg-indigo-50/70 text-indigo-800';
    } else if (colorPalette === 'emerald') {
      if (pct > 60) return 'bg-emerald-600 text-white font-bold';
      if (pct > 35) return 'bg-emerald-500 text-white font-medium';
      if (pct > 20) return 'bg-emerald-400 text-white';
      if (pct > 10) return 'bg-emerald-200 text-emerald-950';
      if (pct > 3) return 'bg-emerald-100 text-emerald-900';
      return 'bg-emerald-50/70 text-emerald-800';
    } else {
      if (pct > 60) return 'bg-amber-600 text-white font-bold';
      if (pct > 35) return 'bg-amber-500 text-white font-medium';
      if (pct > 20) return 'bg-amber-400 text-white';
      if (pct > 10) return 'bg-amber-200 text-amber-950';
      if (pct > 3) return 'bg-amber-100 text-amber-900';
      return 'bg-amber-50/70 text-amber-800';
    }
  };

  // Distribution for focused query
  const focusedDistribution = useMemo(() => {
    if (focusedQueryIdx === null || !matrix[focusedQueryIdx]) return null;
    const row = matrix[focusedQueryIdx];
    return row.map((val, idx) => ({
      token: tokens[idx] || `[${idx}]`,
      index: idx,
      weight: val,
      isMasked: idx > focusedQueryIdx,
    }));
  }, [focusedQueryIdx, matrix, tokens]);

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100 pb-20">
      {/* Top Banner / Header */}
      <div className="border-b border-slate-800 bg-slate-950/80 backdrop-blur-md sticky top-16 z-40">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-5">
          <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
            <div>
              <div className="flex items-center space-x-3">
                <span className="p-2.5 rounded-xl bg-indigo-500/20 text-indigo-400 border border-indigo-500/30 text-2xl">
                  👁️
                </span>
                <div>
                  <div className="flex items-center space-x-2.5">
                    <h1 className="text-2xl font-black tracking-tight text-white">
                      Attention Lab
                    </h1>
                    <span className="px-2 py-0.5 rounded-full text-xs font-semibold bg-indigo-500/20 text-indigo-300 border border-indigo-500/40">
                      Transformer İç Mimarisi
                    </span>
                  </div>
                  <p className="text-sm text-slate-400 mt-0.5">
                    Self-Attention & Multi-Head dinamiklerini interaktif ısı haritası ve nedensel maskeleme ile keşfet.
                  </p>
                </div>
              </div>
            </div>

            {/* Model & Architecture Badge */}
            <div className="flex items-center space-x-2.5">
              <div className="px-3 py-1.5 rounded-lg bg-slate-800/80 border border-slate-700 text-xs font-mono text-slate-300 flex items-center space-x-2">
                <Cpu className="w-4 h-4 text-indigo-400" />
                <span>Model: <strong className="text-indigo-300">{data?.model_name || 'GPT-Model'}</strong></span>
              </div>
              <div className="px-3 py-1.5 rounded-lg bg-slate-800/80 border border-slate-700 text-xs font-mono text-slate-300">
                <span>{numLayers} Katman • {numHeads} Head</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-8 space-y-8">
        {/* Input & Presets Section */}
        <div className="bg-slate-800/60 border border-slate-700/80 rounded-2xl p-6 shadow-xl backdrop-blur-sm">
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-3">
              <div className="relative flex-1">
                <input
                  type="text"
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)}
                  placeholder="İncelemek istediğin metni gir..."
                  className="w-full px-4 py-3.5 rounded-xl bg-slate-900/90 border border-slate-700 text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-indigo-500 text-base"
                />
              </div>
              <button
                type="submit"
                disabled={isLoading || !inputText.trim()}
                className="px-6 py-3.5 rounded-xl bg-gradient-to-r from-indigo-600 to-violet-600 hover:from-indigo-500 hover:to-violet-500 text-white font-semibold flex items-center justify-center space-x-2 transition-all shadow-lg shadow-indigo-600/25 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isLoading ? (
                  <>
                    <RefreshCw className="w-5 h-5 animate-spin" />
                    <span>Hesaplanıyor...</span>
                  </>
                ) : (
                  <>
                    <Eye className="w-5 h-5" />
                    <span>Attention Hesapla</span>
                  </>
                )}
              </button>
            </div>

            {/* Quick Sample Presets */}
            <div className="flex flex-wrap items-center gap-2 pt-1">
              <span className="text-xs text-slate-400 font-medium mr-1">Örnek Cümleler:</span>
              {SAMPLE_PRESETS.map((p, idx) => (
                <button
                  key={idx}
                  type="button"
                  onClick={() => handleApplyPreset(p.text)}
                  className="text-xs px-3 py-1.5 rounded-lg bg-slate-900/70 border border-slate-700/80 hover:border-indigo-500/60 hover:bg-slate-800 text-slate-300 transition-colors flex items-center space-x-1"
                >
                  <span className="font-semibold text-indigo-300">{p.label}:</span>
                  <span className="text-slate-400 truncate max-w-[200px]">{p.text}</span>
                </button>
              ))}
            </div>
          </form>
        </div>

        {/* Toolbar: Katman, Head, Renk ve Görünüm Seçimi */}
        <div className="bg-slate-800/40 border border-slate-700/60 rounded-xl p-4 flex flex-wrap items-center justify-between gap-4">
          {/* Layer Selector */}
          <div className="flex items-center space-x-2">
            <span className="text-xs font-bold text-slate-400 uppercase tracking-wider flex items-center gap-1">
              <Layers className="w-3.5 h-3.5 text-indigo-400" />
              Katman:
            </span>
            <div className="flex items-center space-x-1 bg-slate-900/80 p-1 rounded-lg border border-slate-800">
              <button
                type="button"
                onClick={() => handleLayerSelect(null)}
                className={`px-2.5 py-1 rounded text-xs font-medium transition-all ${
                  selectedLayer === null
                    ? 'bg-indigo-600 text-white shadow-sm'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                Ortalama (Tümü)
              </button>
              {Array.from({ length: numLayers }).map((_, i) => (
                <button
                  key={i}
                  type="button"
                  onClick={() => handleLayerSelect(i)}
                  className={`px-2.5 py-1 rounded text-xs font-medium transition-all ${
                    selectedLayer === i
                      ? 'bg-indigo-600 text-white shadow-sm'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  Katman {i}
                </button>
              ))}
            </div>
          </div>

          {/* Head Selector */}
          <div className="flex items-center space-x-2">
            <span className="text-xs font-bold text-slate-400 uppercase tracking-wider flex items-center gap-1">
              <Split className="w-3.5 h-3.5 text-violet-400" />
              Attention Head:
            </span>
            <div className="flex items-center space-x-1 bg-slate-900/80 p-1 rounded-lg border border-slate-800">
              <button
                type="button"
                onClick={() => handleHeadSelect(null)}
                className={`px-2.5 py-1 rounded text-xs font-medium transition-all ${
                  selectedHead === null
                    ? 'bg-violet-600 text-white shadow-sm'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                Ortalama (Tümü)
              </button>
              {Array.from({ length: numHeads }).map((_, h) => (
                <button
                  key={h}
                  type="button"
                  onClick={() => handleHeadSelect(h)}
                  className={`px-2.5 py-1 rounded text-xs font-medium transition-all ${
                    selectedHead === h
                      ? 'bg-violet-600 text-white shadow-sm'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  Head {h}
                </button>
              ))}
            </div>
          </div>

          {/* View Mode & Color Palette */}
          <div className="flex items-center space-x-3">
            {/* View Mode */}
            <div className="flex items-center bg-slate-900/80 p-1 rounded-lg border border-slate-800">
              <button
                type="button"
                onClick={() => setViewMode('single')}
                className={`px-3 py-1 rounded text-xs font-medium transition-all ${
                  viewMode === 'single'
                    ? 'bg-slate-700 text-white shadow-sm'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                Tekil Isı Haritası
              </button>
              <button
                type="button"
                onClick={() => setViewMode('multi')}
                className={`px-3 py-1 rounded text-xs font-medium transition-all ${
                  viewMode === 'multi'
                    ? 'bg-slate-700 text-white shadow-sm'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                Tüm Head'ler ({numHeads})
              </button>
            </div>

            {/* Color Palette */}
            <div className="flex items-center space-x-1 bg-slate-900/80 p-1 rounded-lg border border-slate-800">
              <button
                type="button"
                onClick={() => setColorPalette('indigo')}
                className={`w-5 h-5 rounded-full bg-indigo-500 transition-all ${
                  colorPalette === 'indigo' ? 'ring-2 ring-white scale-110' : 'opacity-60'
                }`}
                title="İndigo Teması"
              />
              <button
                type="button"
                onClick={() => setColorPalette('emerald')}
                className={`w-5 h-5 rounded-full bg-emerald-500 transition-all ${
                  colorPalette === 'emerald' ? 'ring-2 ring-white scale-110' : 'opacity-60'
                }`}
                title="Zümrüt Teması"
              />
              <button
                type="button"
                onClick={() => setColorPalette('amber')}
                className={`w-5 h-5 rounded-full bg-amber-500 transition-all ${
                  colorPalette === 'amber' ? 'ring-2 ring-white scale-110' : 'opacity-60'
                }`}
                title="Kehribar Teması"
              />
            </div>
          </div>
        </div>

        {/* Main Interactive Stage */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          {/* Left / Center: Interactive Matrix */}
          <div className={`${viewMode === 'single' ? 'lg:col-span-8' : 'lg:col-span-12'}`}>
            <div className="bg-slate-800/80 border border-slate-700 rounded-2xl p-6 shadow-2xl">
              <div className="flex items-center justify-between mb-6 pb-4 border-b border-slate-700">
                <div>
                  <h2 className="text-lg font-bold text-white flex items-center space-x-2">
                    <span>Self-Attention Isı Haritası (Heatmap)</span>
                    <span className="text-xs px-2 py-0.5 rounded bg-slate-700 text-slate-300 font-mono">
                      {selectedLayer === null ? 'Tüm Katmanlar' : `Katman ${selectedLayer}`} • {selectedHead === null ? 'Tüm Headler' : `Head ${selectedHead}`}
                    </span>
                  </h2>
                  <p className="text-xs text-slate-400 mt-1">
                    Satırlar: <strong className="text-indigo-400">Sorgu / Query (Q)</strong> token&apos;ları • Sütunlar: <strong className="text-violet-400">Anahtar / Key (K)</strong> token&apos;ları
                  </p>
                </div>

                {/* Legend */}
                <div className="flex items-center space-x-2 text-xs text-slate-400">
                  <span>Düşük</span>
                  <div className="flex items-center space-x-0.5">
                    <span className="w-3.5 h-3.5 rounded-sm bg-indigo-50/70 border border-slate-600" />
                    <span className="w-3.5 h-3.5 rounded-sm bg-indigo-200" />
                    <span className="w-3.5 h-3.5 rounded-sm bg-indigo-400" />
                    <span className="w-3.5 h-3.5 rounded-sm bg-indigo-600" />
                  </div>
                  <span>Yüksek</span>
                  <span className="ml-3 flex items-center space-x-1 text-slate-500">
                    <span className="w-3.5 h-3.5 rounded-sm bg-gray-900 border border-dashed border-gray-600" />
                    <span>Causal Masked</span>
                  </span>
                </div>
              </div>

              {viewMode === 'single' ? (
                /* Single Matrix View */
                <div className="overflow-x-auto pb-4">
                  {matrix.length > 0 && tokens.length > 0 ? (
                    <div className="inline-block min-w-full">
                      {/* Column Headers (Keys) */}
                      <div className="flex items-end pl-32 mb-2">
                        <div className="text-[11px] font-bold text-violet-400 tracking-wider uppercase mb-1 mr-4">
                          Keys (K) →
                        </div>
                        {tokens.map((token, kIdx) => {
                          const isHoveredCol = hoveredCell?.kIdx === kIdx;
                          return (
                            <div
                              key={kIdx}
                              className={`w-12 text-center text-xs font-mono px-0.5 truncate transition-all ${
                                isHoveredCol
                                  ? 'text-violet-300 font-bold scale-110 underline decoration-violet-400'
                                  : 'text-slate-400'
                              }`}
                              title={`Token ${kIdx}: "${token}"`}
                            >
                              <div className="text-[10px] text-slate-500 font-mono">k{kIdx}</div>
                              {token}
                            </div>
                          );
                        })}
                      </div>

                      {/* Rows (Queries) */}
                      <div className="space-y-1">
                        {tokens.map((qToken, qIdx) => {
                          const isHoveredRow = hoveredCell?.qIdx === qIdx;
                          const isFocused = focusedQueryIdx === qIdx;

                          return (
                            <div key={qIdx} className="flex items-center">
                              {/* Row Label (Query Token) */}
                              <div
                                onClick={() => setFocusedQueryIdx(isFocused ? null : qIdx)}
                                className={`w-32 pr-4 text-right text-xs font-mono truncate cursor-pointer transition-all flex items-center justify-end space-x-1.5 ${
                                  isFocused
                                    ? 'text-indigo-300 font-bold bg-indigo-500/10 py-1 rounded'
                                    : isHoveredRow
                                    ? 'text-indigo-400 font-bold'
                                    : 'text-slate-300 hover:text-indigo-300'
                                }`}
                                title={`Sorgu Token'ı ${qIdx}: "${qToken}" (Detay için tıkla)`}
                              >
                                <span className="text-[10px] text-slate-500 font-mono">q{qIdx}</span>
                                <span className="truncate">{qToken}</span>
                                {isFocused && <ChevronRight className="w-3.5 h-3.5 text-indigo-400" />}
                              </div>

                              {/* Matrix Row Cells */}
                              <div className="flex items-center space-x-1">
                                {tokens.map((kToken, kIdx) => {
                                  const val = matrix[qIdx]?.[kIdx] ?? 0;
                                  const isMasked = kIdx > qIdx; // Causal mask
                                  const isCurrentHover = hoveredCell?.qIdx === qIdx && hoveredCell?.kIdx === kIdx;

                                  return (
                                    <div
                                      key={kIdx}
                                      onMouseEnter={() => setHoveredCell({ qIdx, kIdx, val })}
                                      onMouseLeave={() => setHoveredCell(null)}
                                      onClick={() => setFocusedQueryIdx(qIdx)}
                                      className={`w-12 h-10 rounded-md flex flex-col items-center justify-center text-[10px] cursor-pointer transition-all border ${
                                        isCurrentHover
                                          ? 'ring-2 ring-white z-10 scale-110 shadow-lg'
                                          : ''
                                      } ${
                                        isMasked
                                          ? 'bg-slate-950/80 border-slate-800 text-slate-700 font-mono'
                                          : getCellColor(val, false) + ' border-transparent'
                                      }`}
                                    >
                                      {isMasked ? (
                                        <span className="text-[9px] opacity-40">0.0</span>
                                      ) : (
                                        <>
                                          <span className="font-mono">{(val * 100).toFixed(0)}%</span>
                                        </>
                                      )}
                                    </div>
                                  );
                                })}
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  ) : (
                    <div className="text-center py-12 text-slate-500 text-sm">
                      {isLoading ? 'Attention ağırlıkları hesaplanıyor...' : 'Görüntülenecek veri bulunamadı.'}
                    </div>
                  )}
                </div>
              ) : (
                /* Multi-Head Grid View */
                <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-6">
                  {Object.entries(allHeadsMatrix).map(([headName, headMat]) => (
                    <div
                      key={headName}
                      className="bg-slate-900/90 border border-slate-700/80 rounded-xl p-4 space-y-3"
                    >
                      <div className="flex items-center justify-between">
                        <span className="text-xs font-bold text-indigo-300 uppercase tracking-wider">
                          {headName}
                        </span>
                        <span className="text-[10px] text-slate-500 font-mono">
                          {tokens.length}×{tokens.length}
                        </span>
                      </div>

                      {/* Mini Heatmap */}
                      <div className="grid gap-0.5" style={{ gridTemplateColumns: `repeat(${tokens.length}, minmax(0, 1fr))` }}>
                        {headMat.map((row, rIdx) =>
                          row.map((val, cIdx) => {
                            const isMasked = cIdx > rIdx;
                            const pct = Math.min(100, Math.max(0, val * 100));
                            return (
                              <div
                                key={`${rIdx}-${cIdx}`}
                                className={`aspect-square rounded-sm ${
                                  isMasked
                                    ? 'bg-slate-950/90'
                                    : pct > 50
                                    ? 'bg-indigo-500'
                                    : pct > 25
                                    ? 'bg-indigo-600'
                                    : pct > 10
                                    ? 'bg-indigo-700'
                                    : 'bg-indigo-950/60'
                                }`}
                                title={`Q: ${tokens[rIdx]} -> K: ${tokens[cIdx]} (${(val * 100).toFixed(1)}%)`}
                              />
                            );
                          })
                        )}
                      </div>

                      <div className="text-[11px] text-slate-400 pt-1 border-t border-slate-800 flex justify-between">
                        <span>Max Dikkat:</span>
                        <strong className="text-slate-200">
                          {(Math.max(...headMat.flat()) * 100).toFixed(1)}%
                        </strong>
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {/* Hover Details Card */}
              {hoveredCell && (
                <div className="mt-6 p-4 rounded-xl bg-slate-900/90 border border-indigo-500/40 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
                  <div className="flex items-center space-x-3">
                    <div className="w-9 h-9 rounded-lg bg-indigo-500/20 text-indigo-400 flex items-center justify-center font-bold text-sm">
                      %{(hoveredCell.val * 100).toFixed(1)}
                    </div>
                    <div>
                      <div className="text-sm font-semibold text-white flex items-center space-x-2">
                        <span>Sorgu (Q): <strong className="text-indigo-300">&quot;{tokens[hoveredCell.qIdx]}&quot;</strong></span>
                        <ArrowRight className="w-4 h-4 text-slate-500" />
                        <span>Anahtar (K): <strong className="text-violet-300">&quot;{tokens[hoveredCell.kIdx]}&quot;</strong></span>
                      </div>
                      <p className="text-xs text-slate-400 mt-0.5">
                        {hoveredCell.kIdx > hoveredCell.qIdx
                          ? 'Causal Autoregressive Masking devrede: Gelecekteki tokenlara bakılamaz.'
                          : `Model bu token için toplam dikkatinin %${(hoveredCell.val * 100).toFixed(2)} kadarını bu anahtara ayırdı.`}
                      </p>
                    </div>
                  </div>
                  <div className="text-right text-xs font-mono text-slate-400">
                    <div>Ham Ağırlık: <span className="text-white">{hoveredCell.val.toFixed(4)}</span></div>
                    <div>Konum: [q{hoveredCell.qIdx}, k{hoveredCell.kIdx}]</div>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Right: Token Focus & Distribution (Active when Single View) */}
          {viewMode === 'single' && (
            <div className="lg:col-span-4 space-y-6">
              {/* Token Inspector Card */}
              <div className="bg-slate-800/80 border border-slate-700 rounded-2xl p-6 shadow-xl">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-base font-bold text-white flex items-center space-x-2">
                    <BarChart3 className="w-4 h-4 text-indigo-400" />
                    <span>Token Dikkat Dağılımı</span>
                  </h3>
                  {focusedQueryIdx !== null && (
                    <button
                      type="button"
                      onClick={() => setFocusedQueryIdx(null)}
                      className="text-xs text-slate-400 hover:text-white"
                    >
                      Temizle
                    </button>
                  )}
                </div>

                {focusedDistribution ? (
                  <div className="space-y-4">
                    <div className="p-3 rounded-lg bg-indigo-500/10 border border-indigo-500/20 text-xs text-indigo-300">
                      Odaklanan Sorgu Token&apos;ı: <strong className="text-white">&quot;{tokens[focusedQueryIdx!]}&quot;</strong>
                    </div>

                    <div className="space-y-2.5 max-h-[400px] overflow-y-auto pr-1">
                      {focusedDistribution.map((item) => (
                        <div key={item.index} className="space-y-1">
                          <div className="flex justify-between text-xs">
                            <span className={`font-mono ${item.isMasked ? 'text-slate-600 line-through' : 'text-slate-300'}`}>
                              k{item.index}: &quot;{item.token}&quot;
                            </span>
                            <span className="font-mono text-slate-400">
                              {item.isMasked ? 'Maskeli' : `%${(item.weight * 100).toFixed(1)}`}
                            </span>
                          </div>
                          <div className="w-full h-2 rounded-full bg-slate-900 overflow-hidden">
                            {!item.isMasked && (
                              <div
                                className="h-full bg-gradient-to-r from-indigo-500 to-violet-500 rounded-full transition-all duration-500"
                                style={{ width: `${item.weight * 100}%` }}
                              />
                            )}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                ) : (
                  <div className="text-center py-10 px-4 text-slate-500 text-xs space-y-2">
                    <Sliders className="w-8 h-8 mx-auto text-slate-600 opacity-60" />
                    <p>
                      Sol taraftaki matriste bir satıra (Sorgu / Query) tıklayarak o token&apos;ın dikkat dağılımını incele.
                    </p>
                  </div>
                )}
              </div>

              {/* Quick Architectural Facts */}
              <div className="bg-slate-800/50 border border-slate-700/60 rounded-2xl p-5 space-y-3">
                <h4 className="text-xs font-bold text-slate-300 uppercase tracking-wider flex items-center space-x-1.5">
                  <ShieldCheck className="w-4 h-4 text-emerald-400" />
                  <span>Matematiksel Doğrulama</span>
                </h4>
                <div className="text-xs text-slate-400 space-y-2">
                  <div className="flex justify-between pb-1.5 border-b border-slate-700/50">
                    <span>Satır Olasılık Toplamı:</span>
                    <strong className="text-emerald-400 font-mono">Σ = 1.0 (Softmax)</strong>
                  </div>
                  <div className="flex justify-between pb-1.5 border-b border-slate-700/50">
                    <span>Causal Maskeleme:</span>
                    <strong className="text-indigo-400 font-mono">j &gt; i → -∞</strong>
                  </div>
                  <div className="flex justify-between">
                    <span>Ölçekleme Faktörü:</span>
                    <strong className="text-violet-400 font-mono">1 / √d_k</strong>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Pedagogical Explanation-First Cards */}
        <div className="pt-8 border-t border-slate-800">
          <div className="text-center mb-8">
            <h3 className="text-2xl font-bold text-white flex items-center justify-center space-x-2">
              <span>🎓 Açıklama-Önce (Explanation-First): Self-Attention Rehberi</span>
            </h3>
            <p className="text-sm text-slate-400 mt-2 max-w-2xl mx-auto">
              Attention mekanizması modern yapay zekanın kalbidir. Formüllerin ve mimari kararların arkasındaki mantık:
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {/* Card 1: Attention Formülü */}
            <div className="bg-slate-800/80 border border-slate-700 rounded-2xl p-6 space-y-4 hover:border-indigo-500/50 transition-all">
              <div className="w-10 h-10 rounded-xl bg-indigo-500/20 text-indigo-400 flex items-center justify-center font-bold">
                1
              </div>
              <h4 className="text-base font-bold text-white">Attention Formülü</h4>
              <div className="p-3 bg-slate-900 rounded-xl border border-slate-800 font-mono text-xs text-indigo-300 text-center">
                Attention(Q,K,V) = softmax((QKᵀ) / √dₖ) V
              </div>
              <p className="text-xs text-slate-300 leading-relaxed">
                Her token diğer tüm token&apos;larla nokta çarpımı (dot-product) yaparak bir alaka skoru üretir. Softmax bu skorları toplamı 1.0 olan olasılıklara dönüştürür.
              </p>
            </div>

            {/* Card 2: Q, K, V Rolleri */}
            <div className="bg-slate-800/80 border border-slate-700 rounded-2xl p-6 space-y-4 hover:border-indigo-500/50 transition-all">
              <div className="w-10 h-10 rounded-xl bg-violet-500/20 text-violet-400 flex items-center justify-center font-bold">
                2
              </div>
              <h4 className="text-base font-bold text-white">Q, K ve V Rolleri</h4>
              <ul className="text-xs text-slate-300 space-y-2">
                <li><strong className="text-indigo-400">Query (Q):</strong> &quot;Bu kelime bağlamda ne tür bir bilgi arıyor?&quot;</li>
                <li><strong className="text-violet-400">Key (K):</strong> &quot;Bu kelime hangi bilgiyi temsil ediyor?&quot;</li>
                <li><strong className="text-emerald-400">Value (V):</strong> &quot;Eşleşme olduğunda aktarılacak gerçek anlamsal içerik.&quot;</li>
              </ul>
            </div>

            {/* Card 3: Neden √d_k ile Bölüyoruz? */}
            <div className="bg-slate-800/80 border border-slate-700 rounded-2xl p-6 space-y-4 hover:border-indigo-500/50 transition-all">
              <div className="w-10 h-10 rounded-xl bg-emerald-500/20 text-emerald-400 flex items-center justify-center font-bold">
                3
              </div>
              <h4 className="text-base font-bold text-white">Neden √dₖ ile Bölünür?</h4>
              <p className="text-xs text-slate-300 leading-relaxed">
                Vektör boyutu (dₖ) büyüdükçe nokta çarpım sonuçları çok büyük değerlere ulaşır. Büyük değerlerde softmax fonksiyonunun gradyanı neredeyse sıfıra iner (vanishing gradient). √dₖ varyansı 1&apos;e çekerek eğitimi stabilize eder.
              </p>
            </div>

            {/* Card 4: Causal Maskeleme */}
            <div className="bg-slate-800/80 border border-slate-700 rounded-2xl p-6 space-y-4 hover:border-indigo-500/50 transition-all">
              <div className="w-10 h-10 rounded-xl bg-amber-500/20 text-amber-400 flex items-center justify-center font-bold">
                4
              </div>
              <h4 className="text-base font-bold text-white">Causal (Nedensel) Maskeleme</h4>
              <p className="text-xs text-slate-300 leading-relaxed">
                Dil modelleri bir sonraki kelimeyi tahmin etmek için otoregresif çalışır. Eğitim sırasında gelecekteki kelimeleri &quot;kopya çekmemesi&quot; için üst üçgen matris $-\infty$ ile maskelenir ve softmax sonucu tam olarak 0.00 olur.
              </p>
            </div>
          </div>
        </div>

        {/* Footer Credit */}
        <div className="pt-8 text-center border-t border-slate-800/80">
          <p className="text-xs text-slate-500">
            Local AI Research Lab • Düzenleyen ve Geliştiren: <strong className="text-slate-300">Kenan AY</strong>
          </p>
        </div>
      </div>
    </div>
  );
}
