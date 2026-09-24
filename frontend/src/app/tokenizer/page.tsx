'use client';

/**
 * Tokenizer Lab Page
 * 
 * BPE tokenizer training interface ve vocabulary explorer.
 */

import { useState } from 'react';
import { useTokenizers, useTokenizerJobs, useCreateTokenizerJob } from '@/hooks/useTokenizer';
import { useFiles } from '@/hooks/useFiles';

export default function TokenizerPage() {
  const [activeTab, setActiveTab] = useState<'train' | 'list'>('list');
  
  // Training form state
  const [jobName, setJobName] = useState('');
  const [selectedFiles, setSelectedFiles] = useState<string[]>([]);
  const [vocabSize, setVocabSize] = useState(8000);
  const [minFrequency, setMinFrequency] = useState(2);

  // Queries
  const { data: tokenizers, isLoading: loadingTokenizers } = useTokenizers(true);
  const { data: jobs, isLoading: loadingJobs } = useTokenizerJobs();
  const { data: files, isLoading: loadingFiles } = useFiles();
  
  // Mutations
  const createJob = useCreateTokenizerJob();

  const handleCreateJob = async () => {
    if (!jobName.trim() || selectedFiles.length === 0) {
      alert('Job ismi ve en az bir dosya seçilmeli');
      return;
    }

    try {
      await createJob.mutateAsync({
        job_name: jobName,
        file_ids: selectedFiles,
        vocab_size: vocabSize,
        min_frequency: minFrequency,
      });

      // Reset form
      setJobName('');
      setSelectedFiles([]);
      alert('Training job oluşturuldu!');
    } catch (error) {
      console.error('Job creation failed:', error);
      alert('Job oluşturulamadı');
    }
  };

  const handleFileSelect = (fileId: string) => {
    if (selectedFiles.includes(fileId)) {
      setSelectedFiles(selectedFiles.filter(id => id !== fileId));
    } else {
      setSelectedFiles([...selectedFiles, fileId]);
    }
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <div className="mb-6">
        <h1 className="text-3xl font-bold mb-2">Tokenizer Lab</h1>
        <p className="text-gray-600">BPE Tokenizer Training ve Vocabulary Explorer</p>
      </div>

      {/* Tabs */}
      <div className="flex gap-4 mb-6 border-b">
        <button
          className={`px-4 py-2 font-medium ${
            activeTab === 'list'
              ? 'border-b-2 border-blue-500 text-blue-600'
              : 'text-gray-500 hover:text-gray-700'
          }`}
          onClick={() => setActiveTab('list')}
        >
          Tokenizer Listesi
        </button>
        <button
          className={`px-4 py-2 font-medium ${
            activeTab === 'train'
              ? 'border-b-2 border-blue-500 text-blue-600'
              : 'text-gray-500 hover:text-gray-700'
          }`}
          onClick={() => setActiveTab('train')}
        >
          Yeni Training
        </button>
      </div>

      {/* Tokenizer List Tab */}
      {activeTab === 'list' && (
        <div>
          <div className="mb-6">
            <h2 className="text-xl font-semibold mb-4">Trained Tokenizer'lar</h2>
            
            {loadingTokenizers ? (
              <div className="text-gray-500">Yükleniyor...</div>
            ) : tokenizers && tokenizers.length > 0 ? (
              <div className="grid gap-4">
                {tokenizers.map((tokenizer) => (
                  <div
                    key={tokenizer.tokenizer_id}
                    className="border rounded-lg p-4 hover:shadow-lg transition-shadow"
                  >
                    <div className="flex justify-between items-start mb-2">
                      <div>
                        <h3 className="text-lg font-semibold">{tokenizer.name}</h3>
                        <p className="text-sm text-gray-500">{tokenizer.tokenizer_id}</p>
                      </div>
                      <span className="px-2 py-1 bg-green-100 text-green-800 rounded text-sm">
                        {tokenizer.tokenizer_type}
                      </span>
                    </div>
                    
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-4">
                      <div>
                        <div className="text-sm text-gray-500">Vocab Size</div>
                        <div className="font-semibold">{tokenizer.vocab_size.toLocaleString()}</div>
                      </div>
                      <div>
                        <div className="text-sm text-gray-500">Merges</div>
                        <div className="font-semibold">{tokenizer.num_merges?.toLocaleString() || 'N/A'}</div>
                      </div>
                      <div>
                        <div className="text-sm text-gray-500">Documents</div>
                        <div className="font-semibold">{tokenizer.num_training_documents}</div>
                      </div>
                      <div>
                        <div className="text-sm text-gray-500">Duration</div>
                        <div className="font-semibold">
                          {tokenizer.training_duration_seconds
                            ? `${tokenizer.training_duration_seconds.toFixed(2)}s`
                            : 'N/A'}
                        </div>
                      </div>
                    </div>

                    <div className="mt-4 flex gap-2">
                      <a
                        href={`/tokenizer/${tokenizer.tokenizer_id}`}
                        className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
                      >
                        Detay
                      </a>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-gray-500 text-center py-8 border rounded-lg">
                Henüz tokenizer yok. Yeni bir training başlatın.
              </div>
            )}
          </div>

          {/* Recent Jobs */}
          <div className="mt-8">
            <h2 className="text-xl font-semibold mb-4">Son Training Job'ları</h2>
            
            {loadingJobs ? (
              <div className="text-gray-500">Yükleniyor...</div>
            ) : jobs && jobs.length > 0 ? (
              <div className="space-y-2">
                {jobs.slice(0, 5).map((job) => (
                  <div
                    key={job.job_id}
                    className="border rounded p-3 flex justify-between items-center"
                  >
                    <div>
                      <div className="font-medium">{job.job_name}</div>
                      <div className="text-sm text-gray-500">{job.job_id}</div>
                    </div>
                    <div className="flex items-center gap-4">
                      <div className="text-sm">
                        <span
                          className={`px-2 py-1 rounded ${
                            job.status === 'COMPLETED'
                              ? 'bg-green-100 text-green-800'
                              : job.status === 'RUNNING'
                              ? 'bg-blue-100 text-blue-800'
                              : job.status === 'FAILED'
                              ? 'bg-red-100 text-red-800'
                              : 'bg-gray-100 text-gray-800'
                          }`}
                        >
                          {job.status}
                        </span>
                      </div>
                      {(job.status === 'RUNNING' || job.status === 'PENDING') && (
                        <div className="text-sm text-gray-600">
                          {Math.round(job.progress * 100)}%
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-gray-500 text-center py-4 border rounded-lg">
                Henüz job yok
              </div>
            )}
          </div>
        </div>
      )}

      {/* Training Tab */}
      {activeTab === 'train' && (
        <div className="max-w-2xl">
          <h2 className="text-xl font-semibold mb-4">Yeni Tokenizer Training</h2>
          
          <div className="space-y-6">
            {/* Job Name */}
            <div>
              <label className="block text-sm font-medium mb-2">Job İsmi</label>
              <input
                type="text"
                value={jobName}
                onChange={(e) => setJobName(e.target.value)}
                className="w-full px-3 py-2 border rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder="Örn: Turkish General BPE v1"
              />
            </div>

            {/* Vocab Size */}
            <div>
              <label className="block text-sm font-medium mb-2">
                Vocabulary Size: {vocabSize.toLocaleString()}
              </label>
              <input
                type="range"
                min="500"
                max="50000"
                step="500"
                value={vocabSize}
                onChange={(e) => setVocabSize(Number(e.target.value))}
                className="w-full"
              />
              <div className="text-xs text-gray-500 mt-1">
                Önerilen: 8000-16000 (genel amaçlı), 32000-50000 (large models)
              </div>
            </div>

            {/* Min Frequency */}
            <div>
              <label className="block text-sm font-medium mb-2">
                Minimum Frequency: {minFrequency}
              </label>
              <input
                type="range"
                min="1"
                max="10"
                value={minFrequency}
                onChange={(e) => setMinFrequency(Number(e.target.value))}
                className="w-full"
              />
              <div className="text-xs text-gray-500 mt-1">
                Merge için gerekli minimum görülme sayısı
              </div>
            </div>

            {/* File Selection */}
            <div>
              <label className="block text-sm font-medium mb-2">
                Training Dosyaları ({selectedFiles.length} seçili)
              </label>
              
              {loadingFiles ? (
                <div className="text-gray-500">Dosyalar yükleniyor...</div>
              ) : files && files.length > 0 ? (
                <div className="border rounded max-h-64 overflow-y-auto">
                  {files.map((file) => (
                    <label
                      key={file.file_id}
                      className="flex items-center p-3 hover:bg-gray-50 cursor-pointer border-b last:border-b-0"
                    >
                      <input
                        type="checkbox"
                        checked={selectedFiles.includes(file.file_id)}
                        onChange={() => handleFileSelect(file.file_id)}
                        className="mr-3"
                      />
                      <div className="flex-1">
                        <div className="font-medium text-sm">{file.original_name}</div>
                        <div className="text-xs text-gray-500">
                          {file.file_id} • {(file.size_bytes / 1024).toFixed(1)} KB
                        </div>
                      </div>
                    </label>
                  ))}
                </div>
              ) : (
                <div className="text-gray-500 text-center py-4 border rounded">
                  Dosya yok. Önce dosya upload edin.
                </div>
              )}
            </div>

            {/* Submit Button */}
            <button
              onClick={handleCreateJob}
              disabled={!jobName.trim() || selectedFiles.length === 0 || createJob.isPending}
              className="w-full px-4 py-3 bg-blue-500 text-white rounded hover:bg-blue-600 disabled:bg-gray-300 disabled:cursor-not-allowed font-medium"
            >
              {createJob.isPending ? 'Oluşturuluyor...' : 'Training Başlat'}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
