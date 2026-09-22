import axios from 'axios';
import { ragApi, embeddingsApi, inferenceApi, evaluationApi, nnLabApi, modelsApi } from '@/lib/api';

// Mock axios instance methods
jest.mock('axios', () => {
  const mAxiosInstance = {
    get: jest.fn(),
    post: jest.fn(),
    put: jest.fn(),
    delete: jest.fn(),
    interceptors: {
      request: { use: jest.fn() },
      response: { use: jest.fn() },
    },
  };
  return {
    create: jest.fn(() => mAxiosInstance),
  };
});

describe('Frontend API Client Layer', () => {
  let axiosInstance: any;

  beforeEach(() => {
    jest.clearAllMocks();
    axiosInstance = (axios.create as jest.Mock)();
  });

  describe('ragApi', () => {
    it('calls /api/v1/rag/chunk with chunk request payload', async () => {
      const mockResponse = { data: { chunks: [], total_chunks: 0, strategy: 'recursive', total_characters: 0 } };
      axiosInstance.post.mockResolvedValueOnce(mockResponse);

      const payload = { text: 'test text', strategy: 'recursive' as const, chunk_size: 200, chunk_overlap: 50 };
      const res = await ragApi.chunk(payload);

      expect(axiosInstance.post).toHaveBeenCalledWith('/api/v1/rag/chunk', payload);
      expect(res).toEqual(mockResponse.data);
    });

    it('calls /api/v1/rag/search with search payload', async () => {
      const mockResponse = { data: { query: 'attention', mode: 'hybrid', results: [], count: 0, latency_ms: 1.2 } };
      axiosInstance.post.mockResolvedValueOnce(mockResponse);

      const payload = { query: 'attention', mode: 'hybrid' as const, top_k: 3 };
      const res = await ragApi.search(payload);

      expect(axiosInstance.post).toHaveBeenCalledWith('/api/v1/rag/search', payload);
      expect(res).toEqual(mockResponse.data);
    });

    it('calls /api/v1/rag/query with RAG query payload', async () => {
      const mockResponse = {
        data: {
          query: 'nedir?',
          answer: 'cevap',
          retrieved_chunks: [],
          citations: [],
          model_name: 'test',
          prompt_used: 'prompt',
          stats: {},
        },
      };
      axiosInstance.post.mockResolvedValueOnce(mockResponse);

      const payload = { question: 'nedir?', top_k: 2 };
      const res = await ragApi.query(payload);

      expect(axiosInstance.post).toHaveBeenCalledWith('/api/v1/rag/query', payload);
      expect(res).toEqual(mockResponse.data);
    });

    it('calls /api/v1/rag/collections', async () => {
      const mockResponse = { data: [{ collection_name: 'default', vector_count: 10, d_model: 64, backend: 'pytorch' }] };
      axiosInstance.get.mockResolvedValueOnce(mockResponse);

      const res = await ragApi.collections();

      expect(axiosInstance.get).toHaveBeenCalledWith('/api/v1/rag/collections');
      expect(res).toEqual(mockResponse.data);
    });
  });

  describe('embeddingsApi', () => {
    it('calls /api/v1/embeddings/project with words array', async () => {
      const mockResponse = { data: { points: [], dimensions: 2, explained_variance_ratio: [0.7, 0.3], d_model: 64, model_name: 'gpt' } };
      axiosInstance.post.mockResolvedValueOnce(mockResponse);

      const payload = { words: ['kedi', 'köpek'], dimensions: 2, normalize: true };
      const res = await embeddingsApi.project(payload);

      expect(axiosInstance.post).toHaveBeenCalledWith('/api/v1/embeddings/project', payload);
      expect(res).toEqual(mockResponse.data);
    });

    it('calls /api/v1/embeddings/similarity with two words', async () => {
      const mockResponse = { data: { word_a: 'kedi', word_b: 'köpek', cosine_similarity: 0.82, angle_degrees: 35.0, euclidean_distance: 0.6, dot_product: 0.82, model_name: 'gpt' } };
      axiosInstance.post.mockResolvedValueOnce(mockResponse);

      const payload = { word_a: 'kedi', word_b: 'köpek' };
      const res = await embeddingsApi.similarity(payload);

      expect(axiosInstance.post).toHaveBeenCalledWith('/api/v1/embeddings/similarity', payload);
      expect(res).toEqual(mockResponse.data);
    });
  });

  describe('inferenceApi', () => {
    it('calls /api/v1/inference/beam-search with beam search payload', async () => {
      const mockResponse = {
        data: {
          prompt: 'test prompt',
          beam_width: 3,
          hypotheses: [
            { hypothesis_id: 1, text: 'test result', score: -1.2, normalized_score: -0.6, length: 2, tokens: ['test', 'result'] }
          ],
          model_name: 'test_model',
          execution_time_ms: 15.4
        }
      };
      axiosInstance.post.mockResolvedValueOnce(mockResponse);

      const payload = { prompt: 'test prompt', beam_width: 3, max_length: 20 };
      const res = await inferenceApi.beamSearch(payload);

      expect(axiosInstance.post).toHaveBeenCalledWith('/api/v1/inference/beam-search', payload);
      expect(res).toEqual(mockResponse.data);
    });

    it('calls /api/v1/inference/next-token-probs with logits inspect payload', async () => {
      const mockResponse = {
        data: {
          prompt: 'yapay zeka',
          temperature: 0.8,
          candidates: [
            { rank: 1, token_id: 10, token_text: ' sistemleri', probability: 0.45, percentage: 45.0, raw_logit: 5.2 }
          ],
          model_name: 'test_model'
        }
      };
      axiosInstance.post.mockResolvedValueOnce(mockResponse);

      const payload = { prompt: 'yapay zeka', top_k: 5, temperature: 0.8 };
      const res = await inferenceApi.nextTokenProbs(payload);

      expect(axiosInstance.post).toHaveBeenCalledWith('/api/v1/inference/next-token-probs', payload);
      expect(res).toEqual(mockResponse.data);
    });
  });

  describe('evaluationApi', () => {
    it('calls /api/v1/evaluation/benchmarks/sample-questions with query param', async () => {
      const mockResponse = {
        data: [
          {
            id: 'gsm8k_01',
            input: 'Test soru',
            target: 'Test cozum #### 42',
            domain: 'aritmetik',
            category: 'math_word_problem',
            difficulty: 'easy',
            numeric_answer: 42,
            steps: ['Adım 1: 20 + 22 = 42'],
            keywords: ['toplama']
          }
        ]
      };
      axiosInstance.get.mockResolvedValueOnce(mockResponse);

      const res = await evaluationApi.getSampleQuestions('gsm8k_cot');

      expect(axiosInstance.get).toHaveBeenCalledWith(
        '/api/v1/evaluation/benchmarks/sample-questions',
        { params: { benchmark_name: 'gsm8k_cot' } }
      );
      expect(res).toEqual(mockResponse.data);
    });

    it('calls /api/v1/evaluation/radar-comparison with model names payload', async () => {
      const mockResponse = {
        data: {
          models: [
            {
              model_name: 'nano-gpt-v1',
              overall_score: 75.5,
              dimensions: [
                { name: 'Akıl Yürütme (GSM8K CoT)', score: 70.0, raw_score: 70.0 },
                { name: 'Türkçe Bilgi & Doğruluk', score: 80.0, raw_score: 80.0 }
              ]
            }
          ],
          dimensions: ['Akıl Yürütme (GSM8K CoT)', 'Türkçe Bilgi & Doğruluk'],
          winner_by_dimension: {
            'Akıl Yürütme (GSM8K CoT)': 'nano-gpt-v1',
            'Türkçe Bilgi & Doğruluk': 'nano-gpt-v1'
          },
          overall_winner: 'nano-gpt-v1'
        }
      };
      axiosInstance.post.mockResolvedValueOnce(mockResponse);

      const payload = { model_names: ['nano-gpt-v1', 'nano-gpt-v2'] };
      const res = await evaluationApi.getRadarComparison(payload);

      expect(axiosInstance.post).toHaveBeenCalledWith('/api/v1/evaluation/radar-comparison', payload);
      expect(res).toEqual(mockResponse.data);
    });

    it('calls /api/v1/evaluation/run with benchmark run payload', async () => {
      const mockResponse = {
        data: {
          benchmark_id: 'b123',
          model_name: 'nano-gpt-v1',
          benchmark_name: 'gsm8k_cot',
          score: 87.5,
          metrics: { accuracy: 87.5 },
          samples_evaluated: 8,
          created_at: '2026-09-22T20:00:00Z'
        }
      };
      axiosInstance.post.mockResolvedValueOnce(mockResponse);

      const payload = {
        model_name: 'nano-gpt-v1',
        benchmark_name: 'gsm8k_cot',
        max_samples: 8,
        batch_size: 4
      };
      const res = await evaluationApi.runBenchmark(payload);

      expect(axiosInstance.post).toHaveBeenCalledWith('/api/v1/evaluation/run', payload);
      expect(res).toEqual(mockResponse.data);
    });

    it('calls /api/v1/evaluation/inspect with candidate and reference text', async () => {
      const mockResponse = {
        data: {
          candidate_tokens: ['bu', 'bir', 'test'],
          reference_tokens: ['bu', 'bir', 'ornek'],
          matched_unigrams: ['bir', 'bu'],
          matched_bigrams: ['bu bir'],
          matched_trigrams: [],
          bleu: { bleu: 66.7, precisions: [0.667, 0.5, 0.0, 0.0], bp: 1.0 },
          brevity_penalty: 1.0,
          candidate_len: 3,
          reference_len: 3,
          rouge: { 'rouge-1': { f: 0.667, p: 0.667, r: 0.667 } }
        }
      };
      axiosInstance.post.mockResolvedValueOnce(mockResponse);

      const payload = { candidate: 'bu bir test', reference: 'bu bir ornek', max_n: 3 };
      const res = await evaluationApi.inspectText(payload);

      expect(axiosInstance.post).toHaveBeenCalledWith('/api/v1/evaluation/inspect-text', payload);
      expect(res).toEqual(mockResponse.data);
    });
  });

  describe('nnLabApi', () => {
    it('calls /api/v1/nn-lab/mlp/simulate with simulation payload', async () => {
      const mockResponse = {
        data: {
          architecture: [2, 4, 1],
          loss: 0.05,
          loss_function: 'mse',
          learning_rate: 0.05,
          inputs: [0.5, -0.2],
          targets: [1.0],
          output: [0.95],
          nodes: [],
          synapses: [],
          layer_health: [],
          dead_neurons: [],
          total_parameters: 17,
          chain_rule_steps: [],
        },
      };
      axiosInstance.post.mockResolvedValueOnce(mockResponse);

      const payload = {
        inputs: [0.5, -0.2],
        targets: [1.0],
        hidden_dims: [4],
        activation: 'gelu',
        loss_function: 'mse',
      };
      const res = await nnLabApi.simulateMLP(payload);

      expect(axiosInstance.post).toHaveBeenCalledWith('/api/v1/nn-lab/mlp/simulate', payload);
      expect(res).toEqual(mockResponse.data);
    });

    it('calls /api/v1/nn-lab/optimizers/race with race payload', async () => {
      const mockResponse = {
        data: {
          landscape: 'saddle',
          initial_point: [-0.1, 1.8],
          steps: 40,
          results: [],
          contours: { x_range: [-3, 3], y_range: [-3, 3], levels: [], grid_z: [] },
        },
      };
      axiosInstance.post.mockResolvedValueOnce(mockResponse);

      const payload = {
        landscape: 'saddle',
        optimizers: ['adamw', 'sgd'],
        steps: 40,
      };
      const res = await nnLabApi.raceOptimizers(payload);

      expect(axiosInstance.post).toHaveBeenCalledWith('/api/v1/nn-lab/optimizers/race', payload);
      expect(res).toEqual(mockResponse.data);
    });

    it('calls /api/v1/nn-lab/landscapes', async () => {
      const mockResponse = {
        data: {
          landscapes: [],
          optimizers: [],
        },
      };
      axiosInstance.get.mockResolvedValueOnce(mockResponse);

      const res = await nnLabApi.getLandscapes();

      expect(axiosInstance.get).toHaveBeenCalledWith('/api/v1/nn-lab/landscapes');
      expect(res).toEqual(mockResponse.data);
    });

    it('calls /api/v1/nn-lab/activations with resolution param', async () => {
      const mockResponse = {
        data: {
          xs: [-3, 0, 3],
          functions: [],
        },
      };
      axiosInstance.get.mockResolvedValueOnce(mockResponse);

      const res = await nnLabApi.getActivations(60);

      expect(axiosInstance.get).toHaveBeenCalledWith('/api/v1/nn-lab/activations', {
        params: { num_points: 60 },
      });
      expect(res).toEqual(mockResponse.data);
    });
  });

  describe('modelsApi (Export & Quantization)', () => {
    it('calls POST /api/v1/models/{name}/export with export payload', async () => {
      const mockExportRecord = {
        success: true,
        model_name: 'test-gpt',
        version: '1.0.0',
        export_format: 'onnx',
        quantization: 'int8',
        file_name: 'test-gpt_1.0.0_int8.onnx',
        file_path: 'models/test-gpt/1.0.0/exports/test-gpt_1.0.0_int8.onnx',
        file_size_mb: 14.2,
        compression_ratio: 3.7,
        sha256: 'abcdef123456',
        export_duration_sec: 1.2,
        download_url: '/api/v1/models/test-gpt/download/test-gpt_1.0.0_int8.onnx',
        deployment_snippet: 'import onnxruntime as ort',
        created_at: '2026-09-22T21:00:00Z',
      };
      axiosInstance.post.mockResolvedValueOnce({ data: mockExportRecord });

      const res = await modelsApi.exportModel('test-gpt', {
        version: '1.0.0',
        export_format: 'onnx',
        quantization: 'int8',
      });

      expect(axiosInstance.post).toHaveBeenCalledWith('/api/v1/models/test-gpt/export', {
        version: '1.0.0',
        export_format: 'onnx',
        quantization: 'int8',
      });
      expect(res).toEqual(mockExportRecord);
    });

    it('calls GET /api/v1/models/{name}/exports with version query param', async () => {
      const mockExports = [
        {
          file_name: 'test-gpt_1.0.0.onnx',
          export_format: 'onnx',
          quantization: 'none',
          file_size_mb: 52.4,
        },
      ];
      axiosInstance.get.mockResolvedValueOnce({ data: mockExports });

      const res = await modelsApi.listExports('test-gpt', '1.0.0');

      expect(axiosInstance.get).toHaveBeenCalledWith('/api/v1/models/test-gpt/exports', {
        params: { version: '1.0.0' },
      });
      expect(res).toEqual(mockExports);
    });

    it('generates correct download URL with getDownloadUrl', () => {
      const url = modelsApi.getDownloadUrl('test-gpt', 'model.onnx', '1.0.0');
      expect(url).toContain('/api/v1/models/test-gpt/download/model.onnx?version=1.0.0');
    });
  });
});


