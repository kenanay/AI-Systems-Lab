# Frontend Mode Badge Entegrasyon Kılavuzu

## Oluşturulan Component

✅ `frontend/src/components/ModeBadge.tsx` oluşturuldu

## Özellikler

- ✅ `ModeBadge`: Kompakt badge component
- ✅ `ModeIndicator`: Detaylı açıklamalı indicator
- ✅ `useOperationMode`: Mode detection hook
- ✅ 4 mod: `real`, `simulation`, `demo`, `pending`
- ✅ Dark mode desteği
- ✅ 3 boyut: `sm`, `md`, `lg`
- ✅ İkon desteği

## Entegrasyon Örnekleri

### 1. Evaluation Lab Entegrasyonu

```tsx
// frontend/src/app/evaluation/page.tsx
import ModeBadge, { ModeIndicator, useOperationMode } from '@/components/ModeBadge';

// Benchmark sonucu gösterilirken:
<div className="flex items-center justify-between mb-4">
  <h3 className="text-lg font-semibold">
    {benchmark.benchmark_name}
  </h3>
  <ModeBadge mode={benchmark.metrics?.mode || 'real'} />
</div>

// Detaylı açıklama için:
<ModeIndicator 
  mode="real"
  title="Gerçek Model Değerlendirmesi"
  description="Bu perplexity değeri gerçek test kümesi üzerinde hesaplanmıştır."
/>
```

### 2. Training Lab Entegrasyonu

```tsx
// frontend/src/app/training/page.tsx
import ModeBadge from '@/components/ModeBadge';

// Loss grafiğinde:
<div className="p-4 border rounded-lg">
  <div className="flex items-center justify-between mb-3">
    <h4 className="font-semibold">Training Loss</h4>
    <ModeBadge 
      mode={activeJob ? 'real' : 'demo'} 
      size="sm"
    />
  </div>
  <ResponsiveContainer width="100%" height={300}>
    {/* Loss grafiği */}
  </ResponsiveContainer>
</div>
```

### 3. RAG Lab Entegrasyonu

```tsx
// frontend/src/app/rag-lab/page.tsx
import ModeBadge, { useOperationMode } from '@/components/ModeBadge';

// RAG yanıtında:
const mode = useOperationMode(ragResponse);

<div className="bg-white dark:bg-gray-800 p-6 rounded-lg border">
  <div className="flex items-center justify-between mb-4">
    <h3 className="text-xl font-semibold">Üretilen Yanıt</h3>
    <ModeBadge mode={mode} />
  </div>
  <p className="text-gray-700 dark:text-gray-300">
    {ragResponse.answer}
  </p>
</div>
```

### 4. Attention Lab Entegrasyonu

```tsx
// frontend/src/app/attention-lab/page.tsx
import ModeBadge from '@/components/ModeBadge';

// Attention heatmap'te:
<div className="space-y-4">
  <ModeIndicator 
    mode="simulation"
    title="Attention Simülasyonu"
    description="Bu attention pattern'i rastgele başlatılmış demo model ile üretilmiştir. Gerçek eğitilmiş model seçmek için Model Registry'yi kullanın."
  />
  
  {/* Heatmap visualization */}
</div>
```

### 5. Systems Lab Entegrasyonu

```tsx
// frontend/src/app/systems-lab/page.tsx
import { ModeIndicator } from '@/components/ModeBadge';

// Sistem simülasyonlarında:
<ModeIndicator 
  mode="simulation"
  title="GPU Bellek Simülasyonu"
  description="Teorik hesaplama - gerçek GPU ölçümleri değil"
  className="mb-4"
/>
```

## Hızlı Entegrasyon Kontrol Listesi

Her laboratuvar için:

- [ ] **Evaluation Lab** (`/evaluation`)
  - [ ] Benchmark sonuçlarına badge ekle
  - [ ] Perplexity/BLEU/ROUGE için mode göster
  
- [ ] **Training Lab** (`/training`)
  - [ ] Loss grafiğine badge ekle
  - [ ] Demo data uyarısı ekle
  
- [ ] **RAG Lab** (`/rag-lab`)
  - [ ] Query sonuçlarına mode badge ekle
  - [ ] Real/Demo ayrımı net göster
  
- [ ] **Attention Lab** (`/attention-lab`)
  - [ ] Heatmap'e simulation badge ekle
  - [ ] Demo model uyarısı ekle
  
- [ ] **Embedding Lab** (`/embedding-lab`)
  - [ ] Vector visualization'a badge ekle
  
- [ ] **Transformer Lab** (`/transformer-lab`)
  - [ ] Forward pass sonuçlarına badge ekle
  
- [ ] **Math Lab** (`/math-lab`)
  - [ ] Tüm hesaplamalara "SİMÜLASYON" badge ekle
  
- [ ] **Tensor Lab** (`/tensor-lab`)
  - [ ] Operasyonlara "SİMÜLASYON" badge ekle
  
- [ ] **NN Lab** (`/nn-lab`)
  - [ ] Network visualization'a badge ekle
  
- [ ] **Systems Lab** (`/systems-lab`)
  - [ ] Tüm simülasyonlara badge ekle

## API Response'larında Mode Bilgisi

Backend API'lar zaten mode bilgisi dönüyor:

```python
# Backend
{
  "stats": {
    "mode": "real",  # veya "simulation", "demo"
    ...
  }
}
```

Frontend'de bu değeri kullanın:

```tsx
const mode = useOperationMode(apiResponse);
<ModeBadge mode={mode} />
```

## Styling

Component otomatik olarak dark mode desteği içerir:

- ✅ Light mode renkleri
- ✅ Dark mode renkleri
- ✅ Border ve background tutarlı
- ✅ Erişilebilir contrast oranları

## Testing

Component test örneği:

```tsx
// __tests__/ModeBadge.test.tsx
import { render, screen } from '@testing-library/react';
import ModeBadge from '@/components/ModeBadge';

test('renders real mode badge', () => {
  render(<ModeBadge mode="real" />);
  expect(screen.getByText('GERÇEK')).toBeInTheDocument();
});

test('shows correct description on hover', () => {
  render(<ModeBadge mode="simulation" />);
  const badge = screen.getByTitle('Eğitim amaçlı matematiksel simülasyon');
  expect(badge).toBeInTheDocument();
});
```

## Tamamlanma

✅ Component oluşturuldu  
⏳ Lab entegrasyonları yapılacak (her lab için 5-10 dakika)  
⏳ Test coverage eklenecek

**Not:** Component hazır, her lab'a manuel olarak entegre edilmesi gerekiyor.
Her lab için yukarıdaki örnekleri referans alarak badge'leri ekleyin.
