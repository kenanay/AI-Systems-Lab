/**
 * Home Page
 */

import Link from 'next/link';

export default function HomePage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <div className="container mx-auto px-4 py-16">
        <div className="text-center mb-12">
          <h1 className="text-5xl font-bold text-gray-900 mb-4">
            🤖 Local AI Research Lab
          </h1>
          <p className="text-xl text-gray-600 max-w-2xl mx-auto">
            Local-First AI Systems Research & Learning Platform
          </p>
          <p className="text-sm text-gray-500 mt-2">
            Yapay zeka modellerini öğren, eğit ve deneyle — tamamen kendi bilgisayarında.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 max-w-6xl mx-auto">
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

          {/* Tokenizer Lab (Future) */}
          <div className="bg-gray-100 rounded-lg shadow-md p-6 border border-gray-300 opacity-60 cursor-not-allowed">
            <div className="text-4xl mb-4">🔤</div>
            <h2 className="text-2xl font-semibold text-gray-700 mb-2">
              Tokenizer Lab
            </h2>
            <p className="text-gray-500">
              BPE, WordPiece ve SentencePiece tokenizer'ları öğren.
            </p>
            <div className="mt-4 text-sm text-gray-400 font-medium">
              Yakında →
            </div>
          </div>

          {/* Embedding Lab (Future) */}
          <div className="bg-gray-100 rounded-lg shadow-md p-6 border border-gray-300 opacity-60 cursor-not-allowed">
            <div className="text-4xl mb-4">🎯</div>
            <h2 className="text-2xl font-semibold text-gray-700 mb-2">
              Embedding Lab
            </h2>
            <p className="text-gray-500">
              Word2Vec, GloVe ve transformer embedding'lerini keşfet.
            </p>
            <div className="mt-4 text-sm text-gray-400 font-medium">
              Yakında →
            </div>
          </div>

          {/* Attention Lab (Future) */}
          <div className="bg-gray-100 rounded-lg shadow-md p-6 border border-gray-300 opacity-60 cursor-not-allowed">
            <div className="text-4xl mb-4">👁️</div>
            <h2 className="text-2xl font-semibold text-gray-700 mb-2">
              Attention Lab
            </h2>
            <p className="text-gray-500">
              Self-Attention ve Multi-Head Attention mekanizmasını görselleştir.
            </p>
            <div className="mt-4 text-sm text-gray-400 font-medium">
              Yakında →
            </div>
          </div>

          {/* Training Lab (Future) */}
          <div className="bg-gray-100 rounded-lg shadow-md p-6 border border-gray-300 opacity-60 cursor-not-allowed">
            <div className="text-4xl mb-4">🏋️</div>
            <h2 className="text-2xl font-semibold text-gray-700 mb-2">
              Training Lab
            </h2>
            <p className="text-gray-500">
              Kendi modelini eğit: Pretraining, SFT, LoRA, RLHF.
            </p>
            <div className="mt-4 text-sm text-gray-400 font-medium">
              Yakında →
            </div>
          </div>
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
      </div>
    </div>
  );
}
