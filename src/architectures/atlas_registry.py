"""
src/architectures/atlas_registry.py

Architecture Atlas: Canonical AI Systems Registry & Analytical Scaling Engine
=============================================================================
Bu modül, Modern Yapay Zekâ mimarilerinin (Transformer, MoE, Mamba SSM, GQA,
ViT, Diffusion, LSTM, CNN, VAE, MLP) matematiksel temellerini, donanım ve
bellek karmaşıklıklarını içeren merkezi bilgi ve kıyaslama motorudur.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import math


@dataclass
class ArchitectureSpec:
    id: str
    name: str
    category: str
    tag: str
    formula: str
    time_complexity: str
    memory_complexity: str
    inference_cache: str
    summary: str
    strengths: List[str]
    limitations: List[str]
    reference_models: List[str]
    active_in_lab: bool = True


ARCHITECTURE_CATALOG: Dict[str, Dict[str, Any]] = {
    "transformer_dense": {
        "id": "transformer_dense",
        "name": "Dense Transformer (GPT / LLaMA)",
        "category": "Attention-Based",
        "tag": "Standard Foundation",
        "formula": "Attention(Q, K, V) = softmax(Q K^T / √d_k) V",
        "time_complexity": "O(T^2 · D)",
        "memory_complexity": "O(T^2 + T · D)",
        "inference_cache": "O(L · B · T · D) [KV Cache büyür]",
        "summary": "Tüm token'ların çiftler halinde birbirine dikkat ettiği, matris çarpımı ve softmax tabanlı endüstri standardı autoregressive dil modeli mimarisi.",
        "strengths": [
            "Yüksek temsil gücü ve paralel eğitim verimliliği",
            "Milyarlarca parametreye ölçeklenebilirlik",
            "In-context learning yeteneği"
        ],
        "limitations": [
            "Uzun sekanslarda O(T^2) karesel bellek ve hesaplama maliyeti",
            "Çıkarım sırasında şişen KV Cache bellek darboğazı"
        ],
        "canonical_models": ["GPT-4", "LLaMA-3", "Claude", "BERT"]
    },
    "moe_sparse": {
        "id": "moe_sparse",
        "name": "Mixture of Experts (MoE)",
        "category": "Sparse Activation",
        "tag": "Extreme Efficiency",
        "formula": "y = ∑_{i ∈ TopK} Gate(x)_i · Expert_i(x) + L_aux",
        "time_complexity": "O(T · (Attention + K · D_ff))",
        "memory_complexity": "O(Parametre: E · D_ff, Aktif: K · D_ff)",
        "inference_cache": "O(L · B · T · D)",
        "summary": "Model parametre kapasitesini devasa boyutlara çıkarırken, her token için yalnızca en uygun K uzmanı (ör. Top-2) dinamik yönlendirici (Router) ile aktive eden mimari.",
        "strengths": [
            "Devasa parametre kapasitesine rağmen token başına düşük FLOPs",
            "Eğitim maliyetinde %50-%70 tasarruf",
            "Uzman seviyesinde uzmanlaşma"
        ],
        "limitations": [
            "Tüm uzmanların VRAM'de saklanması zorunluluğu (Yüksek bellek)",
            "Uzman yük dengesizliği (Expert load imbalance) riski"
        ],
        "canonical_models": ["Mixtral 8x7B", "DeepSeek-V2/V3", "Grok-1", "Switch Transformer"]
    },
    "mamba_ssm": {
        "id": "mamba_ssm",
        "name": "Mamba / Selective SSM",
        "category": "State Space Models",
        "tag": "Linear Sequence",
        "formula": "h_t = exp(Δ_t A) h_{t-1} + (Δ_t B_t) x_t,  y_t = C_t h_t + D x_t",
        "time_complexity": "O(T · D · d_state) [Lineer Zaman]",
        "memory_complexity": "O(T · D) [Lineer Bellek]",
        "inference_cache": "O(B · D · d_state) [Sabit O(1) Bellek]",
        "summary": "Sürekli durum uzayını seçici parametrelerle (Δ, B, C) birleştiren; Transformer'ın O(T^2) darboğazını O(T) doğrusal hesaplama ve sabit çıkarım belleğiyle çözen yeni nesil sekans modeli.",
        "strengths": [
            "Milyonlarca token'lık devasa bağlam pencerelerinde sabit bellek",
            "O(1) sabit boyutlu çıkarım durumu (KV Cache gerekmez)",
            "Çok hızlı token üretim hızı (Throughput)"
        ],
        "limitations": [
            "In-context kopyalama ve geri çağırma görevlerinde attention kadar keskin olamayabilir",
            "Donanım optimizasyon kütüphaneleri Transformer kadar olgun değil"
        ],
        "canonical_models": ["Mamba-1 / Mamba-2", "Falcon-Mamba", "Jamba (Hibrit Transformer-MoE-SSM)"]
    },
    "gqa_transformer": {
        "id": "gqa_transformer",
        "name": "Grouped-Query Attention (GQA)",
        "category": "Attention Optimization",
        "tag": "Inference Optimized",
        "formula": "KV_group = Q_heads // KV_heads,  Attn(Q_g, KV_shared)",
        "time_complexity": "O(T^2 · D)",
        "memory_complexity": "O(T^2 + T · (D_Q + D_KV))",
        "inference_cache": "O(L · B · T · D / GroupRatio) [%75 KV Tasarrufu]",
        "summary": "Çoklu query başlıklarının az sayıda Key ve Value başlığını paylaştığı; MHA kalitesini MQA çıkarım hızı ve bellek tasarrufuyla birleştiren mimari.",
        "strengths": [
            "KV Cache boyutunu 4x-8x azaltır",
            "MHA ile neredeyse farksız perplexity performansı",
            "Daha büyük batch size ve daha uzun sekans desteği"
        ],
        "limitations": [
            "Standart attention gibi yine sekans uzunluğunda karesel O(T^2) FLOPs gerektirir"
        ],
        "canonical_models": ["LLaMA-3", "Mistral 7B", "Gemma 2", "Qwen 2.5"]
    },
    "vision_transformer": {
        "id": "vision_transformer",
        "name": "Vision Transformer (ViT)",
        "category": "Multimodal / Vision",
        "tag": "Visual Attention",
        "formula": "Tokens = Linear(Flatten(Patch(Image))) + PosEmb",
        "time_complexity": "O((HW/P^2)^2 · D)",
        "memory_complexity": "O((HW/P^2)^2 + N · D)",
        "inference_cache": "N/A (Encoder Mimarisi)",
        "summary": "Görüntüleri 16x16 yamalara (patch) bölerek kelime token'ları gibi Transformer encoder'ına besleyen görsel dikkat mimarisi.",
        "strengths": [
            "Büyük veri kümelerinde CNN'lerin çok ötesinde genelleme yeteneği",
            "Görsel öğeler arası küresel bağlam modelleme"
        ],
        "limitations": [
            "Küçük veri setlerinde tümevarımsal önyargı (inductive bias) eksikliği",
            "Yüksek çözünürlükte yama sayısı arttıkça karesel maliyet"
        ],
        "canonical_models": ["ViT-Base/Large/Huge", "CLIP", "DINOv2", "SigLIP"]
    },
    "diffusion": {
        "id": "diffusion",
        "name": "Diffusion Models (DDPM / DiT)",
        "category": "Generative / Probabilistic",
        "tag": "Continuous Denoising",
        "formula": "x_{t-1} = 1/√α_t (x_t - (1-α_t)/√(1-ᾱ_t) ε_θ(x_t, t)) + σ_t z",
        "time_complexity": "O(T_steps · ModelForward)",
        "memory_complexity": "O(LatentSize · D)",
        "inference_cache": "N/A",
        "summary": "Veriye aşamalı olarak Gaussian gürültüsü ekleyen ve ardından öğrenilmiş bir ağ ile bu gürültüyü adım adım temizleyerek yüksek kaliteli veri üreten üretken model.",
        "strengths": [
            "Mod çökmesi (mode collapse) yaşamayan kararlı eğitim",
            "Son derece yüksek görsel ve ses üretim kalitesi"
        ],
        "limitations": [
            "Çıkarım sırasında 20-50 adım iteratif gürültü temizleme (yavaş üretim)"
        ],
        "canonical_models": ["Stable Diffusion", "Midjourney", "FLUX.1", "Sora (DiT)"]
    },
    "rnn_lstm": {
        "id": "rnn_lstm",
        "name": "Recurrent Networks (LSTM / GRU)",
        "category": "Recurrent",
        "tag": "Classic Sequential",
        "formula": "f_t = σ(W_f x_t + U_f h_{t-1}),  c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t",
        "time_complexity": "O(T · D^2)",
        "memory_complexity": "O(D) [Sabit Durum]",
        "inference_cache": "O(D) [Çok Düşük]",
        "summary": "Sekansı adım adım işleyerek gizli durum (hidden state) ve hücre belleği (cell state) üzerinden bilgi aktaran klasik yinelemeli ağlar.",
        "strengths": [
            "Sabit bellek ayak izi",
            "Düşük donanımlı gömülü sistemlerde hafif çalışma"
        ],
        "limitations": [
            "Sekans boyunca paralel eğitilemez (yavaş eğitim)",
            "Kaybolan/patlayan gradyanlar ve sınırlı uzun bağlam kapasitesi"
        ],
        "canonical_models": ["Vanilla LSTM", "GRU", "ELMo"]
    },
    "cnn": {
        "id": "cnn",
        "name": "Convolutional Neural Network (CNN)",
        "category": "Convolutional",
        "tag": "Spatial Hierarchy",
        "formula": "y[i, j] = ∑_{m, n} x[i+m, j+n] · K[m, n] + b",
        "time_complexity": "O(H · W · K_h · K_w · C_in · C_out)",
        "memory_complexity": "O(H · W · C)",
        "inference_cache": "N/A",
        "summary": "Yerel reseptif alanlar ve paylaşılan ağırlıklar (kernel) ile öteleme simetrisini (translation invariance) kullanan uzamsal hiyerarşik ağ.",
        "strengths": [
            "Güçlü uzamsal tümevarımsal önyargı (inductive bias)",
            "Küçük veri setlerinde bile hızlı ve kararlı öğrenme"
        ],
        "limitations": [
            "Uzak konumlar arasındaki ilişkileri yakalamak için çok sayıda derin katman gerekir"
        ],
        "canonical_models": ["ResNet", "ConvNeXt", "EfficientNet"]
    },
    "autoencoder_vae": {
        "id": "autoencoder_vae",
        "name": "Variational Autoencoder (VAE)",
        "category": "Generative / Representation",
        "tag": "Latent Compression",
        "formula": "L = E_{q(z|x)}[log p(x|z)] - D_{KL}(q(z|x) || p(z))",
        "time_complexity": "O(N_layers · D)",
        "memory_complexity": "O(LatentDim)",
        "inference_cache": "N/A",
        "summary": "Girdiyi sürekli ve olasılıksal bir gizli uzaya (latent space) sıkıştıran ve bu uzaydan örnekleme yaparak yeni veriler üretebilen mimari.",
        "strengths": [
            "Düzenli ve sürekli gizli uzay temsili",
            "Latent Diffusion modellerinde görüntü sıkıştırma omurgası"
        ],
        "limitations": [
            "Bulanık (blurry) rekonstrüksiyon eğilimi"
        ],
        "canonical_models": ["VAE", "VQ-VAE", "Stable Diffusion VAE"]
    },
    "mlp": {
        "id": "mlp",
        "name": "Multilayer Perceptron (MLP)",
        "category": "Dense Feed-Forward",
        "tag": "Universal Approximator",
        "formula": "y = σ(W_2 σ(W_1 x + b_1) + b_2)",
        "time_complexity": "O(D_in · D_hidden + D_hidden · D_out)",
        "memory_complexity": "O(D_hidden)",
        "inference_cache": "N/A",
        "summary": "Her nöronun bir önceki katmandaki tüm nöronlara bağlandığı, lineer dönüşümler ve doğrusal olmayan aktivasyon fonksiyonlarından oluşan temel yapay sinir ağı.",
        "strengths": [
            "Evrensel fonksiyon yaklaşımı (Universal approximation theorem)",
            "Tabüler ve yapısal verilerde doğrudan uygulanabilirlik"
        ],
        "limitations": [
            "Uzamsal veya zamansal sekans ilişkilerini yapısal olarak modelleyemez"
        ],
        "canonical_models": ["PositionWiseFFN", "MLP-Mixer", "TabNet"]
    }
}


def get_all_architectures() -> List[Dict[str, Any]]:
    """Returns a list of all registered architectures with metadata."""
    return list(ARCHITECTURE_CATALOG.values())


get_architecture_catalog = get_all_architectures


def get_architecture_by_id(arch_id: str) -> Optional[Dict[str, Any]]:
    """Lookup architecture metadata by identifier."""
    return ARCHITECTURE_CATALOG.get(arch_id)


def simulate_scaling_curves(
    seq_lengths: List[int],
    d_model: int = 256,
    num_heads: int = 8,
    num_experts: int = 4,
    top_k: int = 2,
    d_state: int = 16,
    num_layers: int = 1
) -> Dict[str, Any]:
    """
    Computes comparative scaling curves (FLOPs and Memory) across sequence lengths.
    
    Demonstrates:
    - Dense Transformer: O(T^2) quadratic attention FLOPs and activation memory.
    - Sparse MoE: O(T^2) attention with reduced O(K) sparse FFN activation.
    - Mamba SSM: O(T) strictly linear scaling across all sequence lengths.
    """
    dense_flops = []
    moe_flops = []
    ssm_flops = []

    dense_memory_kb = []
    moe_memory_kb = []
    ssm_memory_kb = []

    d_ff = 4 * d_model

    for T in seq_lengths:
        # 1. Dense Transformer:
        # Attention: 2 * T^2 * d_model (QK^T + Attn*V) + 2 * T * d_model^2 (projections)
        # FFN: 2 * T * (2 * d_model * d_ff) = 4 * T * d_model * d_ff
        t_attn_flops = (2 * (T ** 2) * d_model) + (4 * T * (d_model ** 2))
        t_ffn_flops = 4 * T * d_model * d_ff
        t_total_flops = (t_attn_flops + t_ffn_flops) * num_layers
        dense_flops.append(int(t_total_flops))

        # Attention matrix memory: B=1, H=num_heads, T * T * 4 bytes
        t_matrix_bytes = num_heads * (T ** 2) * 4
        t_acts_bytes = ((T * d_model * 4) + t_matrix_bytes) * num_layers
        dense_memory_kb.append(round(t_acts_bytes / 1024, 2))

        # 2. Sparse MoE:
        # Attention: same as Dense Transformer
        # FFN: only Top-k experts activated per token: 4 * T * d_model * (d_ff * top_k / num_experts)
        moe_total_flops = (t_attn_flops + (4 * T * d_model * d_ff * (top_k / num_experts))) * num_layers
        moe_flops.append(int(moe_total_flops))
        moe_memory_kb.append(round((t_acts_bytes * 0.75) / 1024, 2))

        # 3. Mamba SSM:
        # Strictly O(T): Linear scan over T with d_inner * d_state operations
        d_inner = 2 * d_model
        ssm_scan_flops = T * (6 * d_model * d_inner + 4 * d_inner * d_state) * num_layers
        ssm_flops.append(int(ssm_scan_flops))
        # Constant recurrent state memory + input tokens
        ssm_acts_bytes = ((T * d_model * 4) + (d_inner * d_state * 4)) * num_layers
        ssm_memory_kb.append(round(ssm_acts_bytes / 1024, 2))

    return {
        "seq_lengths": seq_lengths,
        "dense_transformer": {
            "name": "Dense Transformer (O(T²))",
            "flops": dense_flops,
            "memory_kb": dense_memory_kb,
            "color": "#ef4444"  # Red
        },
        "sparse_moe": {
            "name": f"Sparse MoE (Top-{top_k}/{num_experts})",
            "flops": moe_flops,
            "memory_kb": moe_memory_kb,
            "color": "#3b82f6"  # Blue
        },
        "mamba_ssm": {
            "name": "Mamba SSM (O(T) Linear)",
            "flops": ssm_flops,
            "memory_kb": ssm_memory_kb,
            "color": "#10b981"  # Green
        }
    }
