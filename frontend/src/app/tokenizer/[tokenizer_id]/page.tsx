'use client';

/**
 * Tokenizer Detail Page
 * 
 * BPE Visualization + Token Inspector
 * 
 * Progressive Disclosure Seviyeleri:
 * - Seviye 1: Token visualization (renk kodlu)
 * - Seviye 2: Token IDs, statistics
 * - Seviye 3: Merge operations, pair frequency
 */

import { useState, useEffect } from 'react';
import { useParams } from 'next/navigation';
import { useTokenizer, useEncodeText } from '@/hooks/useTokenizer';

export default function TokenizerDetailPage() {
  const params = useParams();
  const tokenizerId = params.tokenizer_id as string;
  
  const [inputText, setInputText] = useState('Merhaba dünya! Bu bir tokenizer testidir.');
  const [showLevel, setShowLevel] = useState<1 | 2 | 3>(1);
  
  // Queries
  const { data: tokenizer, isLoading } = useTokenizer(tokenizerId);
  const encodeText = useEncodeText();
  
  // Encoding state
  const [tokens, setTokens] = useState<number[]>([]);
  const [encodedText, setEncodedText] = useState('');
  
  // Auto-encode when input changes
  useEffect(() => {
    if (inputText.trim() && tokenizerId) {
      const timer = setTimeout(() => {
        encodeText.mutate(
          { tokenizerId, text: inputText },
          {
            onSuccess: (data) => {
              setTokens(data.token_ids);
              setEncodedText(data.text);
            },
          }
        );
      }, 500); // Debounce 500ms
      
      return () => clearTimeout(timer);
    }
  }, [inputText, tokenizerId]);
  
  // Token color generator (consistent colors for same token IDs)
  const getTokenColor = (tokenId: number): string => {
    const colors = [
      'bg-blue-100 text-blue-800',
      'bg-green-100 text-green-800',
      'bg-yellow-100 text-yellow-800',
      'bg-red-100 text-red-800',
      'bg-purple-100 text-purple-800',
      'bg-pink-100 text-pink-800',
      'bg-indigo-100 text-indigo-800',
      'bg-teal-100 text-teal-800',
    ];
    return colors[tokenId % colors.length];
  };
  
  if (isLoading) {
    return (
      <div className="container mx-auto p-6">
        <div className="text-gray-500">Yükleniyor...</div>
      </div>
    );
  }
  
  if (!tokenizer) {
    return (
      <div className="container mx-auto p-6">
        <div className="text-red-500">Tokenizer bulunamadı</div>
      </div>
    );
  }
  
  return (
    <div className="container mx-auto p-6">
      {/* Header */}
      <div className="mb-6">
        <div className="flex items-center gap-2 mb-2">
          <a href="/tokenizer" className="text-blue-500 hover:underline">
            ← Tokenizer Lab
          </a>
        </div>
        <h1 className="text-3xl font-bold mb-2">{tokenizer.name}</h1>
        <p className="text-gray-600">{tokenizer.tokenizer_id}</p>
      </div>
      
      {/* Tokenizer Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <div className="border rounded-lg p-4">
          <div className="text-sm text-gray-500">Type</div>
          <div className="text-xl font-semibold">{tokenizer.tokenizer_type}</div>
        </div>
        <div className="border rounded-lg p-4">
          <div className="text-sm text-gray-500">Vocab Size</div>
          <div className="text-xl font-semibold">{tokenizer.vocab_size.toLocaleString()}</div>
        </div>
        <div className="border rounded-lg p-4">
          <div className="text-sm text-gray-500">Merges</div>
          <div className="text-xl font-semibold">
            {tokenizer.num_merges?.toLocaleString() || 'N/A'}
          </div>
        </div>
        <div className="border rounded-lg p-4">
          <div className="text-sm text-gray-500">Training Docs</div>
          <div className="text-xl font-semibold">{tokenizer.num_training_documents}</div>
        </div>
      </div>
      
      {/* Progressive Disclosure Tabs */}
      <div className="flex gap-2 mb-6 border-b">
        <button
          className={`px-4 py-2 font-medium ${
            showLevel === 1
              ? 'border-b-2 border-blue-500 text-blue-600'
              : 'text-gray-500 hover:text-gray-700'
          }`}
          onClick={() => setShowLevel(1)}
        >
          Seviye 1: Token Görselleştirme
        </button>
        <button
          className={`px-4 py-2 font-medium ${
            showLevel === 2
              ? 'border-b-2 border-blue-500 text-blue-600'
              : 'text-gray-500 hover:text-gray-700'
          }`}
          onClick={() => setShowLevel(2)}
        >
          Seviye 2: Token Detayları
        </button>
        <button
          className={`px-4 py-2 font-medium ${
            showLevel === 3
              ? 'border-b-2 border-blue-500 text-blue-600'
              : 'text-gray-500 hover:text-gray-700'
          }`}
          onClick={() => setShowLevel(3)}
        >
          Seviye 3: BPE Teknikleri
        </button>
      </div>
      
      {/* Input Text Area */}
      <div className="mb-6">
        <label className="block text-sm font-medium mb-2">
          Test Metni (Gerçek zamanlı tokenization)
        </label>
        <textarea
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          className="w-full px-4 py-3 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
          rows={4}
          placeholder="Türkçe veya İngilizce text girin..."
        />
      </div>
      
      {/* Seviye 1: Token Visualization */}
      {showLevel === 1 && (
        <div className="space-y-6">
          <div className="border rounded-lg p-6">
            <h2 className="text-xl font-semibold mb-4">Token Görselleştirme</h2>
            <p className="text-sm text-gray-600 mb-4">
              Her token farklı renkte gösteriliyor. Aynı renkteki token'lar aynı ID'ye sahip.
            </p>
            
            {encodeText.isPending && (
              <div className="text-gray-500">Tokenizing...</div>
            )}
            
            {tokens.length > 0 ? (
              <div className="flex flex-wrap gap-2">
                {tokens.map((tokenId, index) => (
                  <span
                    key={index}
                    className={`px-3 py-1 rounded font-mono text-sm ${getTokenColor(tokenId)}`}
                    title={`Token ID: ${tokenId}`}
                  >
                    T{index}
                  </span>
                ))}
              </div>
            ) : (
              <div className="text-gray-400 italic">Token'lar burada görünecek</div>
            )}
            
            <div className="mt-4 pt-4 border-t">
              <div className="text-sm text-gray-600">
                <strong>Toplam Token:</strong> {tokens.length}
              </div>
              <div className="text-sm text-gray-600">
                <strong>Karakter Sayısı:</strong> {inputText.length}
              </div>
              <div className="text-sm text-gray-600">
                <strong>Compression Ratio:</strong>{' '}
                {tokens.length > 0 ? (inputText.length / tokens.length).toFixed(2) : 'N/A'}
              </div>
            </div>
          </div>
          
          {/* Educational Explanation - Seviye 1 */}
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
            <h3 className="font-semibold text-blue-900 mb-2">📚 Tokenization Nedir?</h3>
            <p className="text-sm text-blue-800">
              Tokenization, text'i daha küçük parçalara (token'lara) bölme işlemidir. 
              Büyük dil modelleri (LLM'ler) text'i bu token'lar üzerinden işler. 
              Her token bir ID ile temsil edilir.
            </p>
            <p className="text-sm text-blue-800 mt-2">
              <strong>Örnek:</strong> "Merhaba" kelimesi birden fazla token'a bölünebilir: 
              ["Mer", "ha", "ba"] veya ["Merhaba"] (vocabulary'ye bağlı).
            </p>
          </div>
        </div>
      )}
      
      {/* Seviye 2: Token Details */}
      {showLevel === 2 && (
        <div className="space-y-6">
          <div className="border rounded-lg p-6">
            <h2 className="text-xl font-semibold mb-4">Token Detayları</h2>
            
            {tokens.length > 0 ? (
              <div className="space-y-4">
                {/* Token IDs Table */}
                <div>
                  <h3 className="font-medium mb-2">Token ID Listesi</h3>
                  <div className="bg-gray-50 rounded p-4 max-h-64 overflow-y-auto">
                    <div className="grid grid-cols-8 gap-2">
                      {tokens.map((tokenId, index) => (
                        <div
                          key={index}
                          className="text-center p-2 bg-white border rounded text-sm"
                        >
                          <div className="text-xs text-gray-500">#{index}</div>
                          <div className="font-mono font-semibold">{tokenId}</div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
                
                {/* Token Statistics */}
                <div>
                  <h3 className="font-medium mb-2">İstatistikler</h3>
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                    <div className="bg-gray-50 rounded p-3">
                      <div className="text-xs text-gray-500">Unique Tokens</div>
                      <div className="text-lg font-semibold">
                        {new Set(tokens).size}
                      </div>
                    </div>
                    <div className="bg-gray-50 rounded p-3">
                      <div className="text-xs text-gray-500">Avg Token ID</div>
                      <div className="text-lg font-semibold">
                        {(tokens.reduce((a, b) => a + b, 0) / tokens.length).toFixed(0)}
                      </div>
                    </div>
                    <div className="bg-gray-50 rounded p-3">
                      <div className="text-xs text-gray-500">Chars per Token</div>
                      <div className="text-lg font-semibold">
                        {(inputText.length / tokens.length).toFixed(2)}
                      </div>
                    </div>
                  </div>
                </div>
                
                {/* Token Frequency */}
                <div>
                  <h3 className="font-medium mb-2">Token Frekansı (Top 10)</h3>
                  <div className="bg-gray-50 rounded p-4">
                    {(() => {
                      const freq = tokens.reduce((acc, id) => {
                        acc[id] = (acc[id] || 0) + 1;
                        return acc;
                      }, {} as Record<number, number>);
                      
                      const sorted = Object.entries(freq)
                        .sort(([, a], [, b]) => b - a)
                        .slice(0, 10);
                      
                      return (
                        <div className="space-y-1">
                          {sorted.map(([id, count]) => (
                            <div key={id} className="flex items-center gap-2">
                              <span className="font-mono text-sm w-16">ID {id}</span>
                              <div className="flex-1 bg-blue-100 rounded h-6 relative">
                                <div
                                  className="bg-blue-500 h-full rounded"
                                  style={{
                                    width: `${(count / tokens.length) * 100}%`,
                                  }}
                                />
                                <span className="absolute inset-0 flex items-center justify-center text-xs font-semibold">
                                  {count}x ({((count / tokens.length) * 100).toFixed(1)}%)
                                </span>
                              </div>
                            </div>
                          ))}
                        </div>
                      );
                    })()}
                  </div>
                </div>
              </div>
            ) : (
              <div className="text-gray-400 italic">Detaylar burada görünecek</div>
            )}
          </div>
          
          {/* Educational Explanation - Seviye 2 */}
          <div className="bg-green-50 border border-green-200 rounded-lg p-4">
            <h3 className="font-semibold text-green-900 mb-2">📊 Token ID'leri Ne İşe Yarar?</h3>
            <p className="text-sm text-green-800">
              Her token bir sayısal ID ile temsil edilir (0'dan vocab_size'a kadar). 
              Model bu ID'leri kullanarak embedding'leri (vektör temsillerini) bulur.
            </p>
            <p className="text-sm text-green-800 mt-2">
              <strong>Compression Ratio:</strong> Kaç karakter bir token'a sığdırılıyor. 
              Yüksek ratio = daha verimli tokenization. Türkçe'de genelde 3-4 char/token.
            </p>
          </div>
        </div>
      )}
      
      {/* Seviye 3: BPE Techniques */}
      {showLevel === 3 && (
        <div className="space-y-6">
          <div className="border rounded-lg p-6">
            <h2 className="text-xl font-semibold mb-4">BPE (Byte-Pair Encoding) Nasıl Çalışır?</h2>
            
            <div className="space-y-4">
              {/* Step by step explanation */}
              <div className="bg-gray-50 rounded p-4">
                <h3 className="font-semibold mb-3">Algoritma Adımları:</h3>
                <ol className="list-decimal list-inside space-y-2 text-sm">
                  <li>
                    <strong>Başlangıç:</strong> Text'i karakterlere böl. Her karakter bir token.
                  </li>
                  <li>
                    <strong>Pair Counting:</strong> Ardışık token pair'lerinin frekansını say.
                  </li>
                  <li>
                    <strong>Merge:</strong> En sık görülen pair'i tek bir token'a birleştir.
                  </li>
                  <li>
                    <strong>Tekrarla:</strong> Hedef vocab size'a ulaşana kadar 2-3 adımını tekrarla.
                  </li>
                </ol>
              </div>
              
              {/* Example */}
              <div className="bg-blue-50 rounded p-4">
                <h3 className="font-semibold mb-3 text-blue-900">Örnek:</h3>
                <div className="space-y-2 text-sm text-blue-800 font-mono">
                  <div>
                    <strong>Input:</strong> "aaabdaaabac"
                  </div>
                  <div>
                    <strong>Step 0:</strong> ["a", "a", "a", "b", "d", "a", "a", "a", "b", "a", "c"]
                  </div>
                  <div>
                    <strong>Pair Frequencies:</strong> ("a","a")=4, ("a","b")=2, ("b","a")=1, ...
                  </div>
                  <div>
                    <strong>Merge 1:</strong> ("a","a") → "aa"
                  </div>
                  <div>
                    <strong>Step 1:</strong> ["aa", "a", "b", "d", "aa", "a", "b", "a", "c"]
                  </div>
                  <div className="text-xs text-blue-600 mt-2">
                    ↓ Bu işlem vocab_size'a ulaşana kadar devam eder
                  </div>
                </div>
              </div>
              
              {/* Tokenizer Stats */}
              <div className="grid grid-cols-2 gap-4">
                <div className="border rounded p-4">
                  <h4 className="font-semibold mb-2 text-sm">Bu Tokenizer'ın Merge Sayısı</h4>
                  <div className="text-3xl font-bold text-blue-600">
                    {tokenizer.num_merges?.toLocaleString() || 'N/A'}
                  </div>
                  <p className="text-xs text-gray-600 mt-1">
                    Training sırasında yapılan merge operasyonu sayısı
                  </p>
                </div>
                
                <div className="border rounded p-4">
                  <h4 className="font-semibold mb-2 text-sm">Base Vocab → Final Vocab</h4>
                  <div className="text-lg font-semibold">
                    ~256 → {tokenizer.vocab_size.toLocaleString()}
                  </div>
                  <p className="text-xs text-gray-600 mt-1">
                    Byte-level (256 char) → {tokenizer.vocab_size} tokens
                  </p>
                </div>
              </div>
              
              {/* Special Tokens */}
              <div className="border rounded p-4">
                <h4 className="font-semibold mb-2">Special Tokens</h4>
                <div className="flex flex-wrap gap-2">
                  {tokenizer.special_tokens.map((token, index) => (
                    <span
                      key={index}
                      className="px-3 py-1 bg-purple-100 text-purple-800 rounded font-mono text-sm"
                    >
                      {token}
                    </span>
                  ))}
                </div>
                <p className="text-xs text-gray-600 mt-2">
                  Özel anlamlı token'lar: padding, unknown, sentence başı/sonu
                </p>
              </div>
            </div>
          </div>
          
          {/* Educational Explanation - Seviye 3 */}
          <div className="bg-purple-50 border border-purple-200 rounded-lg p-4">
            <h3 className="font-semibold text-purple-900 mb-2">🎓 Neden BPE?</h3>
            <p className="text-sm text-purple-800">
              BPE, kelime-bazlı ve karakter-bazlı tokenization'ın avantajlarını birleştirir:
            </p>
            <ul className="list-disc list-inside text-sm text-purple-800 mt-2 space-y-1">
              <li><strong>Karakter-bazlı:</strong> Her şeyi temsil edebilir ama çok uzun</li>
              <li><strong>Kelime-bazlı:</strong> Hızlı ama rare word'ler için kötü</li>
              <li><strong>BPE (Subword):</strong> Frequent subword'leri öğrenir, rare word'leri parçalara böler</li>
            </ul>
            <p className="text-sm text-purple-800 mt-2">
              Sonuç: Kompakt vocabulary + güçlü generalization!
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
