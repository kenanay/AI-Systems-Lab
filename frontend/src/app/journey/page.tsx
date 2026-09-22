'use client';

import React, { useState, useEffect, useMemo } from 'react';
import Link from 'next/link';
import {
  Compass,
  GitBranch,
  BookOpen,
  CheckCircle2,
  AlertCircle,
  HelpCircle,
  ArrowRight,
  Sparkles,
  ExternalLink,
  Search,
  Layers,
  ChevronRight,
  Activity,
  Award,
  BookCheck,
  Cpu,
  RefreshCw,
} from 'lucide-react';
import {
  api,
  JourneyStage,
  KnowledgeGraphResponse,
  GlossaryTerm,
  CheckQuestionResponse,
} from '@/lib/api';

export default function JourneyPage() {
  const [activeTab, setActiveTab] = useState<'curriculum' | 'graph' | 'glossary'>('curriculum');

  // Curriculum Data & State
  const [stages, setStages] = useState<JourneyStage[]>([]);
  const [loadingStages, setLoadingStages] = useState<boolean>(true);
  const [stageError, setStageError] = useState<string | null>(null);

  // User Quiz Answers State: { [question_id]: { selected: number, result?: CheckQuestionResponse } }
  const [userAnswers, setUserAnswers] = useState<
    Record<string, { selected: number; checking: boolean; result?: CheckQuestionResponse }>
  >({});

  // Knowledge Graph State
  const [graphData, setGraphData] = useState<KnowledgeGraphResponse>({ nodes: [], edges: [] });
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>('math');
  const [loadingGraph, setLoadingGraph] = useState<boolean>(false);

  // Glossary State
  const [glossaryTerms, setGlossaryTerms] = useState<GlossaryTerm[]>([]);
  const [glossarySearch, setGlossarySearch] = useState<string>('');
  const [glossaryCategory, setGlossaryCategory] = useState<string>('all');
  const [loadingGlossary, setLoadingGlossary] = useState<boolean>(false);

  // Fetch Curriculum on mount
  useEffect(() => {
    async function loadCurriculum() {
      setLoadingStages(true);
      try {
        const data = await api.journey.getCurriculum();
        setStages(data || []);
      } catch (err: any) {
        setStageError(err?.message || 'Müfredat yüklenirken hata oluştu.');
      } finally {
        setLoadingStages(false);
      }
    }
    loadCurriculum();
  }, []);

  // Fetch Knowledge Graph on tab switch
  useEffect(() => {
    if (activeTab === 'graph' && graphData.nodes.length === 0) {
      loadGraph();
    }
  }, [activeTab, graphData.nodes.length]);

  const loadGraph = async () => {
    setLoadingGraph(true);
    try {
      const g = await api.journey.getGraph();
      setGraphData(g);
      if (g.nodes.length > 0 && !selectedNodeId) {
        setSelectedNodeId(g.nodes[0].id);
      }
    } catch (err) {
      console.error('Failed to load graph:', err);
    } finally {
      setLoadingGraph(false);
    }
  };

  // Fetch Glossary on tab switch or search/filter change
  useEffect(() => {
    if (activeTab === 'glossary') {
      loadGlossary();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeTab, glossaryCategory, glossarySearch]);

  const loadGlossary = async () => {
    setLoadingGlossary(true);
    try {
      const list = await api.journey.getGlossary({
        search: glossarySearch.trim() || undefined,
        category: glossaryCategory !== 'all' ? glossaryCategory : undefined,
      });
      setGlossaryTerms(list || []);
    } catch (err) {
      console.error('Failed to load glossary:', err);
    } finally {
      setLoadingGlossary(false);
    }
  };

  // Handle Question Answer Submission
  const handleAnswerSubmit = async (questionId: string) => {
    const entry = userAnswers[questionId];
    if (!entry || entry.selected === undefined) return;

    setUserAnswers((prev) => ({
      ...prev,
      [questionId]: { ...entry, checking: true },
    }));

    try {
      const res = await api.journey.checkQuestion({
        question_id: questionId,
        selected_option: entry.selected,
      });
      setUserAnswers((prev) => ({
        ...prev,
        [questionId]: { selected: entry.selected, checking: false, result: res },
      }));
    } catch (err) {
      console.error('Answer check error:', err);
      setUserAnswers((prev) => ({
        ...prev,
        [questionId]: { ...entry, checking: false },
      }));
    }
  };

  // Calculate Progress Stats
  const totalQuestions = useMemo(() => {
    return stages.reduce((acc, stage) => acc + (stage.questions?.length || 0), 0);
  }, [stages]);

  const solvedCorrectly = useMemo(() => {
    return Object.values(userAnswers).filter((a) => a.result?.is_correct).length;
  }, [userAnswers]);

  const progressPercent = totalQuestions > 0 ? Math.round((solvedCorrectly / totalQuestions) * 100) : 0;

  // Selected Node Details in Graph Tab
  const selectedNode = useMemo(() => {
    return graphData.nodes.find((n) => n.id === selectedNodeId) || null;
  }, [graphData.nodes, selectedNodeId]);

  const selectedNodePrereqs = useMemo(() => {
    if (!selectedNodeId) return [];
    const incomingEdgeSources = graphData.edges
      .filter((e) => e.target === selectedNodeId)
      .map((e) => e.source);
    return graphData.nodes.filter((n) => incomingEdgeSources.includes(n.id));
  }, [graphData, selectedNodeId]);

  const selectedNodeNext = useMemo(() => {
    if (!selectedNodeId) return [];
    const outgoingEdgeTargets = graphData.edges
      .filter((e) => e.source === selectedNodeId)
      .map((e) => e.target);
    return graphData.nodes.filter((n) => outgoingEdgeTargets.includes(n.id));
  }, [graphData, selectedNodeId]);

  // Unique Glossary Categories
  const glossaryCategories = useMemo(() => {
    return ['all', 'Mimari', 'Değerlendirme', 'NLP & Tokenizer', 'Hizalama & Fine-Tuning', 'Çıkarım', 'Veri Kalitesi'];
  }, []);

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 pb-16">
      {/* Top Banner & Header */}
      <div className="bg-gradient-to-r from-slate-950 via-indigo-950 to-slate-900 text-white border-b border-indigo-900/50 shadow-md">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
            <div>
              <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-500/20 text-indigo-300 text-xs font-semibold tracking-wide uppercase mb-3 border border-indigo-400/30">
                <Sparkles className="w-3.5 h-3.5" />
                Bölüm 56 • Öğrenme Rehberliği & Knowledge Map
              </div>
              <h1 className="text-3xl font-extrabold tracking-tight sm:text-4xl text-white">
                Guided Learning Journey & Knowledge Map
              </h1>
              <p className="mt-2 text-base text-slate-300 max-w-3xl leading-relaxed">
                Ham veriden modern Transformer dil modellerine uzanan pedagojik yol haritası.
                Kavramları sırasıyla öğrenin, interaktif sorularla bilginizi pekiştirin ve canlı laboratuvarları deneyimleyin.
              </p>
            </div>

            {/* Quick Progress Badge */}
            <div className="bg-slate-900/90 border border-slate-700/80 rounded-2xl p-4 flex items-center gap-4 shadow-lg">
              <div className="w-12 h-12 rounded-xl bg-indigo-600/30 border border-indigo-400/40 flex items-center justify-center text-indigo-400 font-bold text-lg">
                {progressPercent}%
              </div>
              <div>
                <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider block">
                  Öğrenme İlerlemesi
                </span>
                <span className="text-sm font-bold text-white">
                  {solvedCorrectly} / {totalQuestions} Kontrol Tamamlandı
                </span>
              </div>
            </div>
          </div>

          {/* Navigation Tabs */}
          <div className="flex space-x-1 mt-8 overflow-x-auto border-b border-slate-800 pb-1 scrollbar-none">
            <button
              id="tab-curriculum"
              onClick={() => setActiveTab('curriculum')}
              className={`flex items-center gap-2 px-4 py-2.5 text-sm font-medium rounded-t-xl transition-colors whitespace-nowrap ${
                activeTab === 'curriculum'
                  ? 'bg-slate-50 text-indigo-900 shadow-sm font-semibold'
                  : 'text-slate-300 hover:text-white hover:bg-white/5'
              }`}
            >
              <Compass className="w-4 h-4 text-indigo-600" />
              Rehberli Yolculuk (13 Aşama)
            </button>
            <button
              id="tab-graph"
              onClick={() => setActiveTab('graph')}
              className={`flex items-center gap-2 px-4 py-2.5 text-sm font-medium rounded-t-xl transition-colors whitespace-nowrap ${
                activeTab === 'graph'
                  ? 'bg-slate-50 text-indigo-900 shadow-sm font-semibold'
                  : 'text-slate-300 hover:text-white hover:bg-white/5'
              }`}
            >
              <GitBranch className="w-4 h-4 text-emerald-500" />
              Kavram & Ön Bilgi Haritası (DAG)
            </button>
            <button
              id="tab-glossary"
              onClick={() => setActiveTab('glossary')}
              className={`flex items-center gap-2 px-4 py-2.5 text-sm font-medium rounded-t-xl transition-colors whitespace-nowrap ${
                activeTab === 'glossary'
                  ? 'bg-slate-50 text-indigo-900 shadow-sm font-semibold'
                  : 'text-slate-300 hover:text-white hover:bg-white/5'
              }`}
            >
              <BookOpen className="w-4 h-4 text-amber-500" />
              AI & LLM Terimler Sözlüğü
            </button>
          </div>
        </div>
      </div>

      {/* Main Content Area */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 mt-6">
        {/* ==================================================================== */}
        {/* TAB 1: GUIDED JOURNEY (CURRICULUM)                                    */}
        {/* ==================================================================== */}
        {activeTab === 'curriculum' && (
          <div className="space-y-6">
            {/* Overall Progress Tracker Bar */}
            <div className="bg-white p-5 rounded-2xl shadow-sm border border-slate-200">
              <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 mb-3">
                <div className="flex items-center gap-2">
                  <BookCheck className="w-5 h-5 text-indigo-600" />
                  <span className="font-bold text-slate-900 text-sm">
                    Rehberli Öğrenme İlerleme Durumu
                  </span>
                </div>
                <div className="text-xs font-semibold text-slate-500">
                  {solvedCorrectly} / {totalQuestions} soru doğru çözüldü (%{progressPercent})
                </div>
              </div>
              <div className="w-full bg-slate-100 rounded-full h-2.5 overflow-hidden">
                <div
                  className="bg-indigo-600 h-full rounded-full transition-all duration-500"
                  style={{ width: `${progressPercent}%` }}
                />
              </div>

              {/* Quick Jump Bar */}
              <div className="mt-4 pt-3 border-t border-slate-100 flex items-center gap-1.5 overflow-x-auto scrollbar-none text-xs">
                <span className="text-slate-400 font-semibold uppercase tracking-wider whitespace-nowrap mr-1">
                  Hızlı Atlama:
                </span>
                {stages.map((st) => {
                  return (
                    <a
                      key={st.id}
                      href={`#${st.id}`}
                      className="px-2.5 py-1 rounded-lg bg-slate-50 hover:bg-indigo-50 border border-slate-200 hover:border-indigo-300 text-slate-700 font-medium transition whitespace-nowrap"
                    >
                      Aşama {st.order}
                    </a>
                  );
                })}
              </div>
            </div>

            {loadingStages ? (
              <div className="bg-white p-12 rounded-2xl text-center shadow-sm border border-slate-200 text-slate-400 text-sm flex flex-col items-center gap-2">
                <RefreshCw className="w-6 h-6 animate-spin text-indigo-600" />
                Müfredat aşamaları yükleniyor...
              </div>
            ) : stageError ? (
              <div className="p-4 rounded-xl bg-red-50 border border-red-200 text-red-700 text-sm flex items-center gap-3">
                <AlertCircle className="w-5 h-5" />
                <span>{stageError}</span>
              </div>
            ) : (
              <div className="space-y-6">
                {stages.map((stage) => {
                  return (
                    <div
                      key={stage.id}
                      id={stage.id}
                      className="bg-white rounded-2xl shadow-sm border border-slate-200 overflow-hidden transition-all hover:border-indigo-200 scroll-mt-20"
                    >
                      {/* Stage Header */}
                      <div className="p-6 border-b border-slate-100 bg-gradient-to-r from-slate-50/70 to-white">
                        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
                          <div>
                            <div className="flex items-center gap-2 mb-1.5">
                              <span className="px-2.5 py-0.5 rounded-full bg-indigo-100 text-indigo-800 text-xs font-bold uppercase tracking-wider border border-indigo-200">
                                {stage.category}
                              </span>
                              <span className="text-xs text-slate-400 font-medium">
                                Adım {stage.order + 1} / {stages.length}
                              </span>
                            </div>
                            <h2 className="text-xl font-bold text-slate-900 tracking-tight">
                              {stage.title}
                            </h2>
                            <p className="mt-1 text-sm text-slate-600 leading-relaxed max-w-4xl">
                              {stage.summary}
                            </p>
                          </div>

                          {/* Lab Button */}
                          {stage.lab_url && (
                            <Link
                              href={stage.lab_url}
                              className="inline-flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl bg-indigo-50 hover:bg-indigo-100 text-indigo-700 text-xs font-bold transition border border-indigo-200/80 shadow-2xs whitespace-nowrap self-start md:self-center"
                            >
                              <Layers className="w-4 h-4 text-indigo-600" />
                              <span>{stage.lab_title || 'Canlı Laboratuvarda Dene'}</span>
                              <ArrowRight className="w-3.5 h-3.5" />
                            </Link>
                          )}
                        </div>

                        {/* Concept Badges */}
                        <div className="flex flex-wrap items-center gap-1.5 mt-4">
                          <span className="text-xs font-semibold text-slate-400 mr-1">Temel Kavramlar:</span>
                          {stage.concepts.map((concept, i) => (
                            <span
                              key={i}
                              className="px-2.5 py-0.5 rounded-md bg-white border border-slate-200 text-slate-700 text-xs font-medium"
                            >
                              {concept}
                            </span>
                          ))}
                        </div>
                      </div>

                      {/* Stage Body */}
                      <div className="p-6 grid grid-cols-1 lg:grid-cols-2 gap-6">
                        {/* Left: Objectives */}
                        <div className="space-y-3">
                          <h3 className="text-xs font-bold text-slate-500 uppercase tracking-wider flex items-center gap-2">
                            <Activity className="w-3.5 h-3.5 text-indigo-500" />
                            Bu Aşamada Kazanılacak Yetkinlikler
                          </h3>
                          <ul className="space-y-2">
                            {stage.objectives.map((obj, i) => (
                              <li key={i} className="flex items-start gap-2.5 text-xs text-slate-700 leading-relaxed">
                                <CheckCircle2 className="w-4 h-4 text-emerald-500 flex-shrink-0 mt-0.5" />
                                <span>{obj}</span>
                              </li>
                            ))}
                          </ul>

                          {/* Prerequisites Note */}
                          {stage.prerequisites && stage.prerequisites.length > 0 && (
                            <div className="mt-4 pt-3 border-t border-slate-100 text-[11px] text-slate-400 flex items-center gap-1.5">
                              <span className="font-semibold">Ön Koşul Aşamalar:</span>
                              {stage.prerequisites.map((p) => (
                                <a
                                  key={p}
                                  href={`#${p}`}
                                  className="text-indigo-600 hover:underline font-mono"
                                >
                                  {p}
                                </a>
                              ))}
                            </div>
                          )}
                        </div>

                        {/* Right: Milestone Check Quiz */}
                        <div className="bg-slate-50/80 p-5 rounded-xl border border-slate-200/80 space-y-4">
                          <div className="flex items-center justify-between border-b border-slate-200 pb-2.5">
                            <h3 className="text-xs font-bold text-slate-800 uppercase tracking-wider flex items-center gap-2">
                              <HelpCircle className="w-4 h-4 text-indigo-600" />
                              Aşama Bilgi Kontrolü ({stage.questions.length} Soru)
                            </h3>
                            <span className="text-[11px] text-slate-400 font-medium">Kavram Pekiştirme</span>
                          </div>

                          <div className="space-y-5">
                            {stage.questions.map((q, qIndex) => {
                              const answerState = userAnswers[q.id];
                              const isSubmitted = !!answerState?.result;
                              const isCorrect = answerState?.result?.is_correct;

                              return (
                                <div key={q.id} className="space-y-2.5 text-xs">
                                  <div className="font-semibold text-slate-900 flex items-start gap-2 leading-relaxed">
                                    <span className="w-4 h-4 rounded-full bg-indigo-100 text-indigo-700 flex items-center justify-center flex-shrink-0 text-[10px] font-bold">
                                      {qIndex + 1}
                                    </span>
                                    <span>{q.question}</span>
                                  </div>

                                  {/* Options */}
                                  <div className="space-y-1.5 pl-6">
                                    {q.options.map((opt, optIndex) => {
                                      const isSelected = answerState?.selected === optIndex;
                                      return (
                                        <label
                                          key={optIndex}
                                          className={`flex items-start gap-2.5 p-2 rounded-lg border cursor-pointer transition ${
                                            isSelected
                                              ? 'bg-indigo-50/70 border-indigo-400 text-indigo-950 font-medium'
                                              : 'bg-white border-slate-200/80 hover:bg-slate-100/50 text-slate-700'
                                          }`}
                                        >
                                          <input
                                            type="radio"
                                            name={`q_${q.id}`}
                                            checked={isSelected}
                                            onChange={() => {
                                              setUserAnswers((prev) => ({
                                                ...prev,
                                                [q.id]: {
                                                  selected: optIndex,
                                                  checking: false,
                                                },
                                              }));
                                            }}
                                            className="mt-0.5 accent-indigo-600"
                                          />
                                          <span className="leading-snug">{opt}</span>
                                        </label>
                                      );
                                    })}
                                  </div>

                                  {/* Check Button */}
                                  <div className="pl-6 flex items-center gap-3">
                                    <button
                                      onClick={() => handleAnswerSubmit(q.id)}
                                      disabled={answerState?.selected === undefined || answerState?.checking}
                                      className="px-3 py-1 rounded-md bg-indigo-600 hover:bg-indigo-500 text-white font-semibold text-[11px] transition disabled:opacity-40"
                                    >
                                      {answerState?.checking ? 'Kontrol Ediliyor...' : 'Cevabı Doğrula'}
                                    </button>
                                  </div>

                                  {/* Result & Pedagogical Feedback */}
                                  {isSubmitted && (
                                    <div
                                      className={`mt-2 p-3 rounded-lg border text-xs leading-relaxed space-y-1 ${
                                        isCorrect
                                          ? 'bg-emerald-50 border-emerald-200 text-emerald-900'
                                          : 'bg-red-50 border-red-200 text-red-900'
                                      }`}
                                    >
                                      <div className="font-bold flex items-center gap-1.5">
                                        {isCorrect ? (
                                          <>
                                            <CheckCircle2 className="w-4 h-4 text-emerald-600" />
                                            <span>Harika, Doğru Cevap!</span>
                                          </>
                                        ) : (
                                          <>
                                            <AlertCircle className="w-4 h-4 text-red-600" />
                                            <span>Yanlış Seçenek (Doğrusu: {q.options[q.correct_index]})</span>
                                          </>
                                        )}
                                      </div>
                                      <p className="text-[11px] opacity-90">{answerState.result?.explanation}</p>
                                      {answerState.result?.math_intuition && (
                                        <div className="mt-1 font-mono text-[11px] bg-white/70 p-1.5 rounded border border-black/5">
                                          Matematiksel Sezgi: {answerState.result.math_intuition}
                                        </div>
                                      )}
                                    </div>
                                  )}
                                </div>
                              );
                            })}
                          </div>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        )}

        {/* ==================================================================== */}
        {/* TAB 2: KNOWLEDGE MAP (DAG)                                           */}
        {/* ==================================================================== */}
        {activeTab === 'graph' && (
          <div className="space-y-6">
            <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200">
              <div className="border-b border-slate-100 pb-4 mb-5 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
                <div>
                  <h2 className="text-lg font-bold text-slate-900 flex items-center gap-2">
                    <GitBranch className="w-5 h-5 text-emerald-600" />
                    Kavram & Ön Bilgi Bağımlılık Haritası (Knowledge DAG)
                  </h2>
                  <p className="text-xs text-slate-500 mt-1">
                    Hangi konunun hangi kavrama ön koşul olduğunu gösteren katmanlı yönlü döngüsüz çizge (DAG).
                    İncelemek istediğiniz düğüme tıklayın.
                  </p>
                </div>

                <button
                  onClick={loadGraph}
                  disabled={loadingGraph}
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-100 hover:bg-slate-200 text-slate-700 text-xs font-semibold transition self-start"
                >
                  <RefreshCw className={`w-3.5 h-3.5 ${loadingGraph ? 'animate-spin' : ''}`} />
                  Yenile
                </button>
              </div>

              {loadingGraph ? (
                <div className="p-12 text-center text-slate-400 text-sm">Graf yükleniyor...</div>
              ) : (
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                  {/* Left: Interactive Node Canvas */}
                  <div className="lg:col-span-2 bg-slate-950 p-6 rounded-2xl border border-slate-800 text-white min-h-[460px] flex flex-col justify-between">
                    <div>
                      <span className="text-[11px] uppercase tracking-wider text-slate-400 font-bold block mb-4">
                        Öğrenme Akış Seviyeleri (Level 0 ➔ Level 11)
                      </span>

                      {/* Visual Node Grid */}
                      <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                        {graphData.nodes.map((node) => {
                          const isSelected = selectedNodeId === node.id;
                          return (
                            <button
                              key={node.id}
                              onClick={() => setSelectedNodeId(node.id)}
                              className={`p-3 rounded-xl border text-left transition-all ${
                                isSelected
                                  ? 'bg-indigo-600/40 border-indigo-400 ring-2 ring-indigo-500/50 shadow-lg'
                                  : 'bg-slate-900/80 border-slate-800 hover:border-slate-600 hover:bg-slate-800/60'
                              }`}
                            >
                              <div className="flex items-center justify-between text-[10px] text-slate-400 mb-1">
                                <span className="uppercase font-mono">Seviye {node.level}</span>
                                <span className="px-1.5 py-0.5 rounded bg-slate-800 text-slate-300">
                                  {node.category}
                                </span>
                              </div>
                              <div className="font-bold text-xs text-slate-100 leading-tight">
                                {node.label}
                              </div>
                            </button>
                          );
                        })}
                      </div>
                    </div>

                    <div className="mt-6 pt-4 border-t border-slate-800/80 flex items-center justify-between text-xs text-slate-400 font-mono">
                      <span>{graphData.nodes.length} Temel Düğüm</span>
                      <span>{graphData.edges.length} Ön Koşul Bağlantısı</span>
                    </div>
                  </div>

                  {/* Right: Node Details & Prerequisites Drawer */}
                  <div className="bg-slate-50 p-5 rounded-2xl border border-slate-200 flex flex-col justify-between">
                    {selectedNode ? (
                      <div className="space-y-4">
                        <div>
                          <span className="px-2 py-0.5 rounded bg-indigo-100 text-indigo-800 text-[10px] font-bold uppercase tracking-wider">
                            {selectedNode.category} • Seviye {selectedNode.level}
                          </span>
                          <h3 className="text-base font-bold text-slate-900 mt-2">
                            {selectedNode.label}
                          </h3>
                        </div>

                        {/* Incoming Prerequisites */}
                        <div>
                          <span className="text-xs font-bold text-slate-500 uppercase tracking-wider block mb-1.5">
                            Gerekli Ön Koşullar:
                          </span>
                          {selectedNodePrereqs.length > 0 ? (
                            <div className="space-y-1">
                              {selectedNodePrereqs.map((p) => (
                                <button
                                  key={p.id}
                                  onClick={() => setSelectedNodeId(p.id)}
                                  className="w-full text-left p-2 rounded-lg bg-white border border-slate-200 text-xs font-medium text-slate-700 hover:border-indigo-300 hover:bg-indigo-50/40 flex items-center justify-between"
                                >
                                  <span>{p.label}</span>
                                  <ChevronRight className="w-3.5 h-3.5 text-slate-400" />
                                </button>
                              ))}
                            </div>
                          ) : (
                            <span className="text-xs text-slate-400 italic">
                              Temel başlangıç noktası (Ön koşul yok)
                            </span>
                          )}
                        </div>

                        {/* Outgoing Enables Next */}
                        <div>
                          <span className="text-xs font-bold text-slate-500 uppercase tracking-wider block mb-1.5">
                            Kilidini Açtığı İleri Konular:
                          </span>
                          {selectedNodeNext.length > 0 ? (
                            <div className="space-y-1">
                              {selectedNodeNext.map((n) => (
                                <button
                                  key={n.id}
                                  onClick={() => setSelectedNodeId(n.id)}
                                  className="w-full text-left p-2 rounded-lg bg-white border border-slate-200 text-xs font-medium text-slate-700 hover:border-emerald-300 hover:bg-emerald-50/40 flex items-center justify-between"
                                >
                                  <span>{n.label}</span>
                                  <ChevronRight className="w-3.5 h-3.5 text-slate-400" />
                                </button>
                              ))}
                            </div>
                          ) : (
                            <span className="text-xs text-slate-400 italic">Son seviye uygulama aşaması</span>
                          )}
                        </div>
                      </div>
                    ) : (
                      <div className="text-xs text-slate-400 italic">Lütfen bir düğüm seçiniz</div>
                    )}

                    {/* Action buttons */}
                    {selectedNode && (
                      <div className="mt-6 pt-4 border-t border-slate-200 space-y-2">
                        {selectedNode.lab_url && (
                          <Link
                            href={selectedNode.lab_url}
                            className="w-full inline-flex items-center justify-center gap-2 p-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white font-semibold text-xs transition shadow-sm"
                          >
                            <Layers className="w-4 h-4" />
                            İlgili Laboratuvara Git
                          </Link>
                        )}
                        <button
                          onClick={() => {
                            setActiveTab('curriculum');
                            const el = document.getElementById(selectedNode.stage_id);
                            if (el) el.scrollIntoView({ behavior: 'smooth' });
                          }}
                          className="w-full inline-flex items-center justify-center gap-2 p-2 rounded-xl bg-slate-200 hover:bg-slate-300 text-slate-700 font-semibold text-xs transition"
                        >
                          <BookOpen className="w-4 h-4" />
                          Aşama Dersi ve Testine Git
                        </button>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* ==================================================================== */}
        {/* TAB 3: AI & LLM GLOSSARY                                              */}
        {/* ==================================================================== */}
        {activeTab === 'glossary' && (
          <div className="space-y-6">
            {/* Search & Category Filter Bar */}
            <div className="bg-white p-4 rounded-2xl shadow-sm border border-slate-200 flex flex-col md:flex-row items-center justify-between gap-4">
              <div className="relative w-full md:w-80">
                <input
                  type="text"
                  placeholder="Terimlerde veya açıklamalarda ara..."
                  value={glossarySearch}
                  onChange={(e) => setGlossarySearch(e.target.value)}
                  className="w-full p-2.5 pl-9 rounded-xl border border-slate-200 text-xs bg-slate-50 focus:outline-none focus:ring-2 focus:ring-indigo-500/20"
                />
                <Search className="w-4 h-4 text-slate-400 absolute left-3 top-3" />
              </div>

              {/* Category Pills */}
              <div className="flex items-center gap-1.5 overflow-x-auto w-full md:w-auto scrollbar-none">
                {glossaryCategories.map((cat) => (
                  <button
                    key={cat}
                    onClick={() => setGlossaryCategory(cat)}
                    className={`px-3 py-1.5 rounded-lg text-xs font-semibold whitespace-nowrap transition ${
                      glossaryCategory === cat
                        ? 'bg-indigo-600 text-white'
                        : 'bg-slate-100 hover:bg-slate-200 text-slate-600'
                    }`}
                  >
                    {cat === 'all' ? 'Tüm Terimler' : cat}
                  </button>
                ))}
              </div>
            </div>

            {/* Glossary Grid */}
            {loadingGlossary ? (
              <div className="bg-white p-12 rounded-2xl text-center shadow-sm border border-slate-200 text-slate-400 text-sm">
                Terimler yükleniyor...
              </div>
            ) : glossaryTerms.length === 0 ? (
              <div className="bg-white p-12 rounded-2xl text-center shadow-sm border border-slate-200 text-slate-400 text-sm">
                Aramanızla eşleşen terim bulunamadı.
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {glossaryTerms.map((term, idx) => (
                  <div
                    key={idx}
                    className="bg-white p-5 rounded-2xl shadow-sm border border-slate-200 flex flex-col justify-between hover:border-indigo-300 transition-all group"
                  >
                    <div>
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-[10px] uppercase font-bold tracking-wider px-2 py-0.5 rounded bg-indigo-50 text-indigo-700 border border-indigo-200">
                          {term.category}
                        </span>
                      </div>
                      <h3 className="font-bold text-slate-900 text-base group-hover:text-indigo-600 transition-colors">
                        {term.term}
                      </h3>
                      <p className="text-xs font-medium text-slate-700 mt-2 leading-relaxed">
                        {term.short_def}
                      </p>
                      <p className="text-[11px] text-slate-500 mt-2 leading-normal">
                        {term.detailed_explanation}
                      </p>

                      {/* Formula callout */}
                      {term.formula && (
                        <div className="mt-3 p-2 rounded-lg bg-slate-950 text-slate-200 font-mono text-[11px] overflow-x-auto border border-slate-800">
                          <span className="text-[9px] uppercase tracking-wider text-slate-500 block mb-0.5">
                            Matematiksel Formül:
                          </span>
                          {term.formula}
                        </div>
                      )}
                    </div>

                    {/* Bottom link to Lab */}
                    {term.related_lab_url && (
                      <div className="mt-4 pt-3 border-t border-slate-100 flex items-center justify-end">
                        <Link
                          href={term.related_lab_url}
                          className="inline-flex items-center gap-1.5 text-xs font-semibold text-indigo-600 hover:text-indigo-800 transition"
                        >
                          <span>Laboratuvara Git</span>
                          <ExternalLink className="w-3.5 h-3.5" />
                        </Link>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
