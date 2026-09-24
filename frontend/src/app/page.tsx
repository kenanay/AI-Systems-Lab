/**
 * Home Page
 */

import Link from 'next/link';

export default function HomePage() {
  return (
    <div className="min-h-[calc(100vh-4rem)] bg-gradient-to-b from-slate-50 via-indigo-50/20 to-slate-100/70 pb-16">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12 sm:py-16">
        <div className="text-center mb-12">
          <h1 className="text-4xl sm:text-5xl font-bold text-slate-900 tracking-tight mb-4">
            🤖 Local AI Research Lab
          </h1>
          <p className="text-xl text-slate-600 max-w-2xl mx-auto">
            Local-First AI Systems Research & Learning Platform
          </p>
          <p className="text-sm text-slate-500 mt-2">
            Yapay zeka modellerini öğren, eğit ve deneyle — tamamen kendi bilgisayarında.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 max-w-7xl mx-auto">
          {/* Dataset Explorer */}
          <Link
            href="/dataset-explorer"
            className="bg-white rounded-lg shadow-md hover:shadow-xl transition-shadow p-6 border border-gray-200"
          >
            <div className="text-4xl mb-4">📚</div>
            <h2 className="text-2xl font-semibold text-gray-900 mb-2">
              Dataset Explorer
            </h2>
            <p className="text-gray-600">
              Dosyalarını yükle, parse et, dataset'ini keşfet ve yönet.
            </p>
            <div className="mt-4 text-sm text-blue-600 font-medium">
              Başla →
            </div>
          </Link>

          {/* File Upload */}
          <Link
            href="/upload"
            className="bg-white rounded-lg shadow-md hover:shadow-xl transition-shadow p-6 border border-gray-200"
          >
            <div className="text-4xl mb-4">⬆️</div>
            <h2 className="text-2xl font-semibold text-gray-900 mb-2">
              File Upload
            </h2>
            <p className="text-gray-600">
              TXT, MD, PDF dosyalarını yükle ve parse et.
            </p>
            <div className="mt-4 text-sm text-blue-600 font-medium">
              Dosya Yükle →
            </div>
          </Link>

          {/* Tokenizer Lab - NOW ACTIVE */}
          <Link
            href="/tokenizer"
            className="bg-white rounded-lg shadow-md hover:shadow-xl transition-shadow p-6 border border-gray-200"
          >
            <div className="text-4xl mb-4">🔤</div>
            <h2 className="text-2xl font-semibold text-gray-900 mb-2">
              Tokenizer Lab
            </h2>
            <p className="text-gray-600">
              BPE, WordPiece ve SentencePiece tokenizer'ları öğren ve eğit.
            </p>
            <div className="mt-4 text-sm text-blue-600 font-medium">
              Başla →
            </div>
          </Link>

          {/* Dataset Compiler - NOW ACTIVE */}
          <Link
            href="/dataset-compiler"
            className="bg-white rounded-lg shadow-md hover:shadow-xl transition-shadow p-6 border border-gray-200"
          >
            <div className="text-4xl mb-4">⚙️</div>
            <h2 className="text-2xl font-semibold text-gray-900 mb-2">
              Dataset Compiler
            </h2>
            <p className="text-gray-600">
              Canonical Parquet dataset'leri derle ve optimize et.
            </p>
            <div className="mt-4 text-sm text-blue-600 font-medium">
              Derle →
            </div>
          </Link>

          {/* Training Lab - NOW ACTIVE */}
          <Link
            href="/training"
            className="bg-white rounded-lg shadow-md hover:shadow-xl transition-shadow p-6 border border-gray-200"
          >
            <div className="text-4xl mb-4">🏋️</div>
            <h2 className="text-2xl font-semibold text-gray-900 mb-2">
              Training Lab
            </h2>
            <p className="text-gray-600">
              Kendi modelini eğit: Pretraining, SFT ve LoRA ile ince ayar.
            </p>
            <div className="mt-4 text-sm text-blue-600 font-medium">
              Eğitime Başla →
            </div>
          </Link>

          {/* Model Hub - NOW ACTIVE */}
          <Link
            href="/models"
            className="bg-white rounded-lg shadow-md hover:shadow-xl transition-shadow p-6 border border-gray-200"
          >
            <div className="text-4xl mb-4">📦</div>
            <h2 className="text-2xl font-semibold text-gray-900 mb-2">
              Model Hub
            </h2>
            <p className="text-gray-600">
              Eğitilmiş modelleri incele, sürümleri, parametreleri ve metrikleri yönet.
            </p>
            <div className="mt-4 text-sm text-blue-600 font-medium">
              Modelleri Gör →
            </div>
          </Link>

          {/* Model Playground - NOW ACTIVE */}
          <Link
            href="/playground"
            className="bg-white rounded-lg shadow-md hover:shadow-xl transition-shadow p-6 border border-gray-200"
          >
            <div className="text-4xl mb-4">💬</div>
            <h2 className="text-2xl font-semibold text-gray-900 mb-2">
              Model Playground
            </h2>
            <p className="text-gray-600">
              Eğittiğin modellerle etkileşimli sohbet et ve metin üret.
            </p>
            <div className="mt-4 text-sm text-blue-600 font-medium">
              Test Et →
            </div>
          </Link>

          {/* Embedding Lab - NOW ACTIVE */}
          <Link
            href="/embedding-lab"
            className="bg-white rounded-lg shadow-md hover:shadow-xl transition-shadow p-6 border border-gray-200 group"
          >
            <div className="text-4xl mb-4 transform group-hover:scale-110 transition-transform">🎯</div>
            <h2 className="text-2xl font-semibold text-gray-900 mb-2">
              Embedding Lab
            </h2>
            <p className="text-gray-600">
              Vektör uzayı, PCA 2D/3D projeksiyonu, Cosine benzerliği ve anlamsal cebiri keşfet.
            </p>
            <div className="mt-4 text-sm text-blue-600 font-medium">
              Vektörleri İncele →
            </div>
          </Link>

          {/* Attention Lab - NOW ACTIVE */}
          <Link
            href="/attention-lab"
            className="bg-white rounded-lg shadow-md hover:shadow-xl transition-shadow p-6 border border-gray-200 group"
          >
            <div className="text-4xl mb-4 transform group-hover:scale-110 transition-transform">👁️</div>
            <h2 className="text-2xl font-semibold text-gray-900 mb-2">
              Attention Lab
            </h2>
            <p className="text-gray-600">
              Self-Attention ve Multi-Head Attention matrislerini katman ve head bazında etkileşimli incele.
            </p>
            <div className="mt-4 text-sm text-blue-600 font-medium">
              Matrisleri Keşfet →
            </div>
          </Link>
        </div>

        {/* Info section */}
        <div className="mt-16 max-w-4xl mx-auto bg-white rounded-lg shadow-md p-8 border border-gray-200">
          <h3 className="text-2xl font-semibold text-gray-900 mb-4">
            🎓 Açıklama-Önce (Explanation-First) İlkesi
          </h3>
          <p className="text-gray-700 leading-relaxed mb-4">
            Bu platform yalnızca bir AI aracı değil, aynı zamanda bir öğrenme platformudur. 
            Her işlemin ardındaki matematiksel ve mimari detaylar açıklanır.
          </p>
          <ul className="space-y-2 text-gray-700">
            <li className="flex items-start">
              <span className="mr-2">✓</span>
              <span><strong>Ne yapıyoruz?</strong> Her adım açıklanır.</span>
            </li>
            <li className="flex items-start">
              <span className="mr-2">✓</span>
              <span><strong>Neden yapıyoruz?</strong> Tasarım kararları paylaşılır.</span>
            </li>
            <li className="flex items-start">
              <span className="mr-2">✓</span>
              <span><strong>Matematiksel karşılığı nedir?</strong> Formüller gösterilir.</span>
            </li>
            <li className="flex items-start">
              <span className="mr-2">✓</span>
              <span><strong>Gerçek sistemlerde ne kullanılır?</strong> Production patterns öğretilir.</span>
            </li>
          </ul>
        </div>

        {/* Developer Credit */}
        <div className="mt-8 text-center">
          <p className="text-gray-600 text-sm">
            Düzenleyen ve Geliştiren: <strong className="text-gray-900">Kenan AY</strong>
          </p>
          <p className="text-gray-500 text-xs mt-1">
            © 2026 Local AI Research Lab
          </p>
        </div>
      </div>
    </div>
  );
}
