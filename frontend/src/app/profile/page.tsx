'use client';

/**
 * Kullanıcı Profili ve API Key Yönetim Paneli
 * 
 * RBAC rol görüntüleme, API anahtarı üretme/iptal etme ve Python SDK/cURL entegrasyon örnekleri.
 */

import React, { useState, useEffect } from 'react';
import Link from 'next/link';
import { useAuth } from '@/lib/auth-context';
import { api } from '@/lib/api';
import { APIKeyItem, User } from '@/types/auth';

export default function ProfilePage() {
  const { user, isAuthenticated, isLoading, logout, refreshUser } = useAuth();

  const [activeTab, setActiveTab] = useState<'profile' | 'apikeys' | 'users'>('profile');
  const [apiKeys, setApiKeys] = useState<APIKeyItem[]>([]);
  const [loadingKeys, setLoadingKeys] = useState(false);
  const [allUsers, setAllUsers] = useState<User[]>([]);
  const [loadingUsers, setLoadingUsers] = useState(false);

  // API Key Create Modal
  const [showKeyModal, setShowKeyModal] = useState(false);
  const [keyName, setKeyName] = useState('');
  const [keyDays, setKeyDays] = useState<number>(30);
  const [creatingKey, setCreatingKey] = useState(false);
  const [newlyCreatedKey, setNewlyCreatedKey] = useState<string | null>(null);
  const [copiedKey, setCopiedKey] = useState(false);

  // Load API keys
  useEffect(() => {
    if (isAuthenticated) {
      loadKeys();
      if (user?.role === 'admin') {
        loadUsers();
      }
    }
  }, [isAuthenticated, user?.role]);

  const loadKeys = async () => {
    try {
      setLoadingKeys(true);
      const keys = await api.auth.getApiKeys();
      setApiKeys(keys);
    } catch (err) {
      console.error('API anahtarları yüklenemedi:', err);
    } finally {
      setLoadingKeys(false);
    }
  };

  const loadUsers = async () => {
    try {
      setLoadingUsers(true);
      const res = await api.auth.getUsers();
      setAllUsers(res.users);
    } catch (err) {
      console.error('Kullanıcılar yüklenemedi:', err);
    } finally {
      setLoadingUsers(false);
    }
  };

  const handleCreateKey = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!keyName.trim()) return;

    try {
      setCreatingKey(true);
      const created = await api.auth.createApiKey({
        name: keyName.trim(),
        expires_in_days: keyDays > 0 ? keyDays : undefined,
      });

      if (created.raw_key) {
        setNewlyCreatedKey(created.raw_key);
      }
      setKeyName('');
      loadKeys();
    } catch (err) {
      console.error('API Key oluşturulamadı:', err);
    } finally {
      setCreatingKey(false);
    }
  };

  const handleRevokeKey = async (keyId: string) => {
    if (!confirm('Bu API anahtarını iptal etmek istediğinize emin misiniz?')) return;
    try {
      await api.auth.revokeApiKey(keyId);
      loadKeys();
    } catch (err) {
      console.error('API Key silinemedi:', err);
    }
  };

  const handleChangeRole = async (userId: string, newRole: string) => {
    try {
      await api.auth.updateUserRole(userId, newRole);
      loadUsers();
    } catch (err) {
      alert('Rol güncellenemedi.');
    }
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    setCopiedKey(true);
    setTimeout(() => setCopiedKey(false), 2500);
  };

  if (isLoading) {
    return (
      <div className="min-h-[70vh] flex items-center justify-center">
        <div className="flex items-center gap-3 text-gray-500">
          <div className="w-5 h-5 border-2 border-blue-600 border-t-transparent rounded-full animate-spin" />
          <span>Oturum bilgileri yükleniyor...</span>
        </div>
      </div>
    );
  }

  if (!isAuthenticated || !user) {
    return (
      <div className="min-h-[75vh] flex items-center justify-center p-4">
        <div className="max-w-md w-full bg-white rounded-2xl shadow-xl border border-gray-200 p-8 text-center space-y-4">
          <div className="w-16 h-16 bg-amber-50 text-amber-600 rounded-full flex items-center justify-center mx-auto text-3xl">
            🔒
          </div>
          <h2 className="text-xl font-bold text-gray-900">Giriş Yapılması Gerekiyor</h2>
          <p className="text-sm text-gray-600">
            Profilinizi ve API anahtarlarınızı yönetmek için lütfen oturum açın.
          </p>
          <div className="pt-2 flex flex-col gap-2">
            <Link
              href="/login"
              className="w-full py-2.5 px-4 bg-blue-600 hover:bg-blue-700 text-white font-medium rounded-xl text-sm transition-all text-center"
            >
              Giriş Yap
            </Link>
            <Link
              href="/register"
              className="w-full py-2.5 px-4 bg-gray-100 hover:bg-gray-200 text-gray-700 font-medium rounded-xl text-sm transition-all text-center"
            >
              Yeni Hesap Oluştur
            </Link>
          </div>
        </div>
      </div>
    );
  }

  const roleColors = {
    admin: 'bg-purple-100 text-purple-700 border-purple-200',
    researcher: 'bg-blue-100 text-blue-700 border-blue-200',
    viewer: 'bg-gray-100 text-gray-700 border-gray-200',
  };

  return (
    <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
      {/* Profile Header */}
      <div className="bg-white/90 backdrop-blur-md rounded-2xl border border-gray-200/80 p-6 sm:p-8 shadow-sm flex flex-col sm:flex-row items-start sm:items-center justify-between gap-6">
        <div className="flex items-center gap-5">
          <div className="w-20 h-20 rounded-2xl bg-gradient-to-br from-blue-600 to-indigo-700 text-white flex items-center justify-center text-3xl font-black shadow-lg shadow-blue-500/20">
            {user.username.charAt(0).toUpperCase()}
          </div>
          <div className="space-y-1">
            <div className="flex items-center gap-3">
              <h1 className="text-2xl font-black text-gray-900">
                {user.full_name || user.username}
              </h1>
              <span className={`px-2.5 py-0.5 rounded-full text-xs font-bold uppercase tracking-wider border ${roleColors[user.role] || roleColors.viewer}`}>
                {user.role}
              </span>
            </div>
            <p className="text-sm text-gray-500 font-mono">@{user.username} • {user.email}</p>
            <p className="text-xs text-gray-600">
              Kayıt: {new Date(user.created_at).toLocaleDateString('tr-TR')}
              {user.last_login && ` • Son Giriş: ${new Date(user.last_login).toLocaleTimeString('tr-TR')}`}
            </p>
          </div>
        </div>

        <button
          onClick={logout}
          className="px-4 py-2 bg-gray-100 hover:bg-red-50 hover:text-red-700 text-gray-700 font-medium rounded-xl text-sm transition-all border border-gray-200 flex items-center gap-2"
        >
          <span>🚪</span>
          <span>Çıkış Yap</span>
        </button>
      </div>

      {/* Tabs */}
      <div className="flex items-center gap-2 border-b border-gray-200 pb-3">
        <button
          onClick={() => setActiveTab('profile')}
          className={`px-4 py-2 rounded-xl text-sm font-semibold transition-all flex items-center gap-2 ${
            activeTab === 'profile'
              ? 'bg-blue-600 text-white shadow-sm'
              : 'text-gray-600 hover:bg-gray-100'
          }`}
        >
          <span>👤</span>
          <span>Yetki & Rol Detayları</span>
        </button>

        <button
          onClick={() => setActiveTab('apikeys')}
          className={`px-4 py-2 rounded-xl text-sm font-semibold transition-all flex items-center gap-2 ${
            activeTab === 'apikeys'
              ? 'bg-blue-600 text-white shadow-sm'
              : 'text-gray-600 hover:bg-gray-100'
          }`}
        >
          <span>🔑</span>
          <span>API Anahtarları (SDK)</span>
          {apiKeys.length > 0 && (
            <span className="px-1.5 py-0.2 rounded-full text-xs bg-blue-100 text-blue-700 font-mono font-bold">
              {apiKeys.length}
            </span>
          )}
        </button>

        {user.role === 'admin' && (
          <button
            onClick={() => setActiveTab('users')}
            className={`px-4 py-2 rounded-xl text-sm font-semibold transition-all flex items-center gap-2 ${
              activeTab === 'users'
                ? 'bg-purple-600 text-white shadow-sm'
                : 'text-gray-600 hover:bg-gray-100'
            }`}
          >
            <span>👑</span>
            <span>Kullanıcı Yönetimi</span>
          </button>
        )}
      </div>

      {/* TAB 1: Profile & Permissions */}
      {activeTab === 'profile' && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="bg-white rounded-2xl border border-gray-200/80 p-6 space-y-4 shadow-sm">
            <h3 className="text-base font-bold text-gray-900 flex items-center gap-2">
              <span>🛡️</span>
              <span>Rol Yetkileri (RBAC)</span>
            </h3>
            <div className="space-y-2.5">
              {[
                { label: 'Model Eğitimi & Fine-Tuning Başlatma', ok: user.role === 'admin' || user.role === 'researcher' },
                { label: 'Dataset Derleme & Tokenizer Eğitimi', ok: user.role === 'admin' || user.role === 'researcher' },
                { label: 'Benchmark & Değerlendirme Çalıştırma', ok: user.role === 'admin' || user.role === 'researcher' },
                { label: 'Model Silme & Registry Temizliği', ok: user.role === 'admin' },
                { label: 'Kullanıcı Rollerini Değiştirme', ok: user.role === 'admin' },
                { label: 'Headless API Key Üretme', ok: true },
                { label: 'Attention, Embedding & RAG Lab İncelemesi', ok: true },
              ].map((item, idx) => (
                <div key={idx} className="flex items-center justify-between p-2.5 rounded-xl bg-gray-50 border border-gray-100 text-xs">
                  <span className="font-medium text-gray-700">{item.label}</span>
                  <span className={`px-2 py-0.5 rounded font-bold ${item.ok ? 'bg-green-100 text-green-700' : 'bg-red-50 text-red-500'}`}>
                    {item.ok ? 'İzin Verildi' : 'Kısıtlı'}
                  </span>
                </div>
              ))}
            </div>
          </div>

          <div className="bg-white rounded-2xl border border-gray-200/80 p-6 space-y-4 shadow-sm">
            <h3 className="text-base font-bold text-gray-900 flex items-center gap-2">
              <span>⚡</span>
              <span>Oturum & Güvenlik Özeti</span>
            </h3>
            <div className="space-y-3 text-xs text-gray-600">
              <div className="p-3 bg-blue-50 rounded-xl border border-blue-100">
                <span className="font-semibold text-blue-900 block mb-1">JWT HS256 Koruması</span>
                Oturumunuz, HMAC-SHA256 imzalı kriptografik JSON Web Token ile korunmaktadır.
                İstekler otomatik olarak yenilenir.
              </div>
              <div className="p-3 bg-amber-50 rounded-xl border border-amber-100">
                <span className="font-semibold text-amber-900 block mb-1">Kayan Pencereli Rate Limiter</span>
                Tüm API endpoint&apos;leri istemci IP ve kullanıcı bazında DoS korumalıdır.
                Giriş denemeleri dakikada maksimum 15 istek ile sınırlandırılmıştır.
              </div>
            </div>
          </div>
        </div>
      )}

      {/* TAB 2: API Keys Management */}
      {activeTab === 'apikeys' && (
        <div className="space-y-6">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
            <div>
              <h2 className="text-lg font-bold text-gray-900">API Anahtarları (Headless & SDK Erişimi)</h2>
              <p className="text-xs text-gray-500">
                Python betikleri, CLI ve harici servisler için <code className="bg-gray-100 px-1.5 py-0.5 rounded font-mono">sk_live_...</code> anahtarları.
              </p>
            </div>
            <button
              id="create-api-key-button"
              onClick={() => {
                setShowKeyModal(true);
                setNewlyCreatedKey(null);
              }}
              className="px-4 py-2.5 bg-blue-600 hover:bg-blue-700 text-white font-medium rounded-xl text-sm transition-all shadow-sm flex items-center gap-2 self-start"
            >
              <span>➕</span>
              <span>Yeni API Anahtarı Üret</span>
            </button>
          </div>

          {/* Newly created key banner */}
          {newlyCreatedKey && (
            <div className="p-5 bg-amber-50 border-2 border-amber-300 rounded-2xl space-y-3 animate-in fade-in">
              <div className="flex items-center gap-2 text-amber-800 font-bold text-sm">
                <span>⚠️</span>
                <span>DİKKAT: API Anahtarınızı Güvenli Bir Yere Kaydedin!</span>
              </div>
              <p className="text-xs text-amber-700">
                Bu gizli anahtar güvenlik amacıyla veritabanında SHA-256 olarak özetlenir ve <strong>yalnızca bir defa gösterilir</strong>.
              </p>
              <div className="flex items-center gap-2">
                <input
                  type="text"
                  readOnly
                  value={newlyCreatedKey}
                  className="flex-1 bg-white font-mono text-xs px-3.5 py-2.5 rounded-xl border border-amber-300 text-gray-900"
                />
                <button
                  onClick={() => copyToClipboard(newlyCreatedKey)}
                  className="px-4 py-2.5 bg-amber-600 hover:bg-amber-700 text-white font-semibold rounded-xl text-xs transition-all"
                >
                  {copiedKey ? 'Kopyalandı! ✓' : 'Kopyala'}
                </button>
              </div>
            </div>
          )}

          {/* Keys Table */}
          <div className="bg-white rounded-2xl border border-gray-200/80 overflow-hidden shadow-sm">
            {loadingKeys ? (
              <div className="p-8 text-center text-sm text-gray-500">API anahtarları yükleniyor...</div>
            ) : apiKeys.length === 0 ? (
              <div className="p-12 text-center space-y-3">
                <span className="text-3xl">🔑</span>
                <p className="text-sm font-semibold text-gray-700">Henüz bir API anahtarı üretilmemiş</p>
                <p className="text-xs text-gray-500">
                  Python SDK veya komut satırı ile modellerinize erişmek için yukarıdaki butondan bir anahtar oluşturun.
                </p>
              </div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead className="bg-gray-50/80 border-b border-gray-200 text-gray-500 font-semibold uppercase tracking-wider">
                    <tr>
                      <th className="py-3.5 px-4">Etiket / İsim</th>
                      <th className="py-3.5 px-4">Ön Ek</th>
                      <th className="py-3.5 px-4">Rol</th>
                      <th className="py-3.5 px-4">Oluşturulma</th>
                      <th className="py-3.5 px-4">Son Kullanım</th>
                      <th className="py-3.5 px-4 text-right">İşlem</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-100">
                    {apiKeys.map((key) => (
                      <tr key={key.key_id} className="hover:bg-gray-50/50">
                        <td className="py-3 px-4 font-semibold text-gray-900">{key.name}</td>
                        <td className="py-3 px-4 font-mono text-gray-600">{key.key_prefix}</td>
                        <td className="py-3 px-4">
                          <span className="px-2 py-0.5 rounded text-[11px] font-mono uppercase bg-blue-50 text-blue-700 border border-blue-200">
                            {key.role}
                          </span>
                        </td>
                        <td className="py-3 px-4 text-gray-500">
                          {new Date(key.created_at).toLocaleDateString('tr-TR')}
                        </td>
                        <td className="py-3 px-4 text-gray-500">
                          {key.last_used_at ? new Date(key.last_used_at).toLocaleDateString('tr-TR') : 'Hiç kullanılmadı'}
                        </td>
                        <td className="py-3 px-4 text-right">
                          <button
                            onClick={() => handleRevokeKey(key.key_id)}
                            className="px-2.5 py-1 text-red-600 hover:bg-red-50 rounded-lg font-medium transition-all"
                          >
                            İptal Et
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>

          {/* Integration Guide */}
          <div className="bg-gray-900 text-gray-100 rounded-2xl p-6 space-y-4 font-mono text-xs shadow-md">
            <div className="flex items-center justify-between text-gray-400 border-b border-gray-800 pb-3">
              <span className="font-bold text-gray-200">💡 API Anahtarı ile İstek Gönderme</span>
              <span>Python & cURL Örneği</span>
            </div>
            <div className="space-y-3">
              <div>
                <span className="text-gray-400 block mb-1"># 1. cURL ile Model Listeleme:</span>
                <pre className="bg-black/60 p-3 rounded-xl overflow-x-auto text-green-400">
                  {`curl -H "X-API-Key: sk_live_your_key_here" http://localhost:8000/api/v1/models/available`}
                </pre>
              </div>
              <div>
                <span className="text-gray-400 block mb-1"># 2. Python Requests ile Doğrulama:</span>
                <pre className="bg-black/60 p-3 rounded-xl overflow-x-auto text-blue-300">
                  {`import requests

url = "http://localhost:8000/api/v1/auth/me"
headers = {"X-API-Key": "sk_live_your_key_here"}
res = requests.get(url, headers=headers)
print("Kullanıcı Bilgileri:", res.json())`}
                </pre>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* TAB 3: Admin User Management */}
      {activeTab === 'users' && user.role === 'admin' && (
        <div className="space-y-6">
          <div>
            <h2 className="text-lg font-bold text-gray-900">Kayıtlı Kullanıcı Yönetimi (Admin Only)</h2>
            <p className="text-xs text-gray-500">Kullanıcı yetkilerini ve rollerini dinamik olarak yönetin.</p>
          </div>

          <div className="bg-white rounded-2xl border border-gray-200/80 overflow-hidden shadow-sm">
            {loadingUsers ? (
              <div className="p-8 text-center text-sm text-gray-500">Kullanıcı listesi alınıyor...</div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead className="bg-gray-50/80 border-b border-gray-200 text-gray-500 font-semibold uppercase tracking-wider">
                    <tr>
                      <th className="py-3.5 px-4">Kullanıcı</th>
                      <th className="py-3.5 px-4">E-posta</th>
                      <th className="py-3.5 px-4">Kayıt Tarihi</th>
                      <th className="py-3.5 px-4">Mevcut Rol</th>
                      <th className="py-3.5 px-4 text-right">Rolü Güncelle</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-100">
                    {allUsers.map((u) => (
                      <tr key={u.user_id} className="hover:bg-gray-50/50">
                        <td className="py-3 px-4 font-semibold text-gray-900">
                          {u.full_name ? `${u.full_name} (@${u.username})` : `@${u.username}`}
                        </td>
                        <td className="py-3 px-4 text-gray-600 font-mono">{u.email}</td>
                        <td className="py-3 px-4 text-gray-500">
                          {new Date(u.created_at).toLocaleDateString('tr-TR')}
                        </td>
                        <td className="py-3 px-4">
                          <span className={`px-2.5 py-0.5 rounded-full text-xs font-bold uppercase border ${roleColors[u.role] || roleColors.viewer}`}>
                            {u.role}
                          </span>
                        </td>
                        <td className="py-3 px-4 text-right">
                          <select
                            value={u.role}
                            onChange={(e) => handleChangeRole(u.user_id, e.target.value)}
                            disabled={u.username === user.username}
                            className="text-xs px-2.5 py-1 rounded-lg border border-gray-300 bg-white font-medium"
                          >
                            <option value="admin">Admin</option>
                            <option value="researcher">Researcher</option>
                            <option value="viewer">Viewer</option>
                          </select>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Create Key Modal */}
      {showKeyModal && (
        <div className="fixed inset-0 z-50 bg-black/50 backdrop-blur-sm flex items-center justify-center p-4 animate-in fade-in">
          <div className="bg-white rounded-2xl max-w-md w-full p-6 space-y-4 shadow-2xl border border-gray-200">
            <h3 className="text-base font-bold text-gray-900">Yeni API Anahtarı Üret</h3>
            <form onSubmit={handleCreateKey} className="space-y-4">
              <div>
                <label className="block text-xs font-semibold text-gray-700 uppercase tracking-wider mb-1">
                  Anahtar Etiketi *
                </label>
                <input
                  id="modal-key-name"
                  type="text"
                  required
                  placeholder="örn: Kaggle GPU Pipeline veya Local Script"
                  value={keyName}
                  onChange={(e) => setKeyName(e.target.value)}
                  className="w-full px-3.5 py-2.5 rounded-xl border border-gray-300 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500/20"
                />
              </div>

              <div>
                <label className="block text-xs font-semibold text-gray-700 uppercase tracking-wider mb-1">
                  Geçerlilik Süresi
                </label>
                <select
                  value={keyDays}
                  onChange={(e) => setKeyDays(Number(e.target.value))}
                  className="w-full px-3.5 py-2.5 rounded-xl border border-gray-300 bg-white text-sm"
                >
                  <option value={30}>30 Gün</option>
                  <option value={90}>90 Gün</option>
                  <option value={365}>1 Yıl (365 Gün)</option>
                  <option value={0}>Süresiz</option>
                </select>
              </div>

              <div className="flex items-center justify-end gap-2 pt-2">
                <button
                  type="button"
                  onClick={() => setShowKeyModal(false)}
                  className="px-4 py-2 rounded-xl text-sm font-medium text-gray-700 hover:bg-gray-100"
                >
                  Vazgeç
                </button>
                <button
                  id="modal-submit-key"
                  type="submit"
                  disabled={creatingKey}
                  className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white text-sm font-semibold rounded-xl"
                >
                  {creatingKey ? 'Üretiliyor...' : 'Oluştur'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
