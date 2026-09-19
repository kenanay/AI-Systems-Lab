#!/bin/bash
# Geliştirme ortamı kurulum scripti

set -e

echo "🚀 Local AI Research Lab - Geliştirme Ortamı Kurulumu"
echo ""

# Python versiyonu kontrolü
echo "📌 Python versiyonu kontrol ediliyor..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.10"

if [[ $(echo -e "$python_version\n$required_version" | sort -V | head -n1) != "$required_version" ]]; then
    echo "❌ Python $required_version veya üzeri gerekli. Mevcut: $python_version"
    exit 1
fi
echo "✅ Python $python_version"

# Virtual environment oluştur
echo ""
echo "📦 Virtual environment oluşturuluyor..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ Virtual environment oluşturuldu"
else
    echo "ℹ️  Virtual environment zaten mevcut"
fi

# Virtual environment aktive et
echo ""
echo "🔧 Virtual environment aktive ediliyor..."
source venv/bin/activate

# Python bağımlılıklarını yükle
echo ""
echo "📚 Python bağımlılıkları yükleniyor..."
pip install --upgrade pip
pip install -r requirements.txt
pip install -e ".[dev]"
echo "✅ Python bağımlılıkları yüklendi"

# Frontend bağımlılıklarını yükle
echo ""
echo "🎨 Frontend bağımlılıkları yükleniyor..."
if [ -d "frontend" ]; then
    cd frontend
    if command -v npm &> /dev/null; then
        npm install
        echo "✅ Frontend bağımlılıkları yüklendi"
    else
        echo "⚠️  npm bulunamadı. Frontend kurulumu atlandı."
    fi
    cd ..
fi

# Gerekli dizinleri oluştur
echo ""
echo "📁 Dizinler oluşturuluyor..."
mkdir -p logs
mkdir -p datasets/raw
mkdir -p datasets/normalized
mkdir -p models
mkdir -p checkpoints
mkdir -p experiments
echo "✅ Dizinler oluşturuldu"

# .env dosyası oluştur
echo ""
if [ ! -f ".env" ]; then
    echo "📝 .env dosyası oluşturuluyor..."
    cp .env.example .env
    echo "✅ .env dosyası oluşturuldu (.env.example'dan)"
    echo "⚠️  .env dosyasını kendi ayarlarınıza göre düzenleyin"
else
    echo "ℹ️  .env dosyası zaten mevcut"
fi

# Git hooks kurulumu (opsiyonel)
echo ""
echo "🪝 Git hooks kurulumu..."
if [ -d ".git" ]; then
    # Pre-commit hook
    cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
echo "Running pre-commit checks..."

# Python linting
pylint --errors-only src/ || exit 1

# Type checking
mypy src/ || exit 1

echo "✅ Pre-commit checks passed"
EOF
    chmod +x .git/hooks/pre-commit
    echo "✅ Git hooks kuruldu"
else
    echo "ℹ️  Git repository değil, hooks atlandı"
fi

# Test çalıştır
echo ""
echo "🧪 Test kontrolü yapılıyor..."
pytest tests/ -v || echo "⚠️  Henüz test bulunmuyor"

# GPU kontrolü
echo ""
echo "🎮 GPU kontrolü..."
python3 << EOF
import torch
if torch.cuda.is_available():
    print(f"✅ CUDA mevcut: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA version: {torch.version.cuda}")
    print(f"   GPU sayısı: {torch.cuda.device_count()}")
else:
    print("ℹ️  CUDA mevcut değil (CPU modunda çalışacak)")
EOF

echo ""
echo "✨ Kurulum tamamlandı!"
echo ""
echo "📝 Sonraki adımlar:"
echo "   1. Virtual environment'ı aktive edin: source venv/bin/activate"
echo "   2. .env dosyasını düzenleyin"
echo "   3. Backend'i başlatın: python backend/main.py"
echo "   4. Frontend'i başlatın: cd frontend && npm run dev"
echo ""
echo "📚 Dokümantasyon: docs/README.md"
echo "🔧 Kiro yapılandırması: .kiro/steering/"
echo ""
