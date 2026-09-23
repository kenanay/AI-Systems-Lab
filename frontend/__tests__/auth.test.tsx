import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import axios from 'axios';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { AuthProvider, useAuth } from '@/lib/auth-context';
import LoginPage from '@/app/login/page';
import RegisterPage from '@/app/register/page';
import ProfilePage from '@/app/profile/page';

// Mock next/navigation
jest.mock('next/navigation', () => ({
  useRouter: () => ({
    push: jest.fn(),
    replace: jest.fn(),
  }),
  usePathname: () => '/login',
}));

// Mock axios instance methods
jest.mock('axios', () => {
  const getFn = jest.fn();
  const postFn = jest.fn();
  const putFn = jest.fn();
  const deleteFn = jest.fn();
  const patchFn = jest.fn();
  const defaultsObj = { withCredentials: true, headers: { common: {} } };
  const interceptorsObj = {
    request: { use: jest.fn(), eject: jest.fn() },
    response: { use: jest.fn(), eject: jest.fn() },
  };

  const mAxios = {
    create: jest.fn(() => mAxios),
    defaults: defaultsObj,
    get: getFn,
    post: postFn,
    put: putFn,
    delete: deleteFn,
    patch: patchFn,
    interceptors: interceptorsObj,
  };

  return {
    __esModule: true,
    default: mAxios,
    ...mAxios,
  };
});

function renderWithProviders(ui: React.ReactElement) {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
    },
  });
  return render(
    <QueryClientProvider client={queryClient}>
      <AuthProvider>
        {ui}
      </AuthProvider>
    </QueryClientProvider>
  );
}

// Test Component to test useAuth directly
function TestConsumer() {
  const { user, isAuthenticated, login, logout } = useAuth();
  return (
    <div>
      <div data-testid="auth-status">{isAuthenticated ? 'LOGGED_IN' : 'LOGGED_OUT'}</div>
      <div data-testid="username">{user?.username || 'GUEST'}</div>
      <div data-testid="role">{user?.role || 'NONE'}</div>
      <button onClick={() => login('admin', 'admin')} data-testid="test-login">
        Login
      </button>
      <button onClick={logout} data-testid="test-logout">
        Logout
      </button>
    </div>
  );
}

describe('Auth & Security Frontend Suite', () => {
  beforeEach(() => {
    localStorage.clear();
    jest.clearAllMocks();
    (axios.get as jest.Mock).mockImplementation(() => Promise.reject(new Error('No auth')));
    (axios.post as jest.Mock).mockImplementation(() => Promise.reject(new Error('No auth')));
  });

  it('provides default guest auth state and logs in successfully', async () => {
    (axios.post as jest.Mock).mockImplementation((url: string) => {
      if (url.includes('/auth/login')) {
        return Promise.resolve({
          data: {
            access_token: 'fake-jwt-token-123',
            refresh_token: 'fake-refresh-token-456',
            token_type: 'bearer',
            expires_in: 3600,
            user: {
              user_id: 'usr_admin',
              username: 'admin',
              email: 'admin@ailab.local',
              role: 'admin',
              full_name: 'Administrator',
              created_at: new Date().toISOString(),
            },
          },
        });
      }
      if (url.includes('/auth/logout')) {
        return Promise.resolve({ data: { message: 'Logged out' } });
      }
      return Promise.reject(new Error('No auth'));
    });

    renderWithProviders(<TestConsumer />);

    expect(screen.getByTestId('auth-status')).toHaveTextContent('LOGGED_OUT');
    expect(screen.getByTestId('username')).toHaveTextContent('GUEST');

    fireEvent.click(screen.getByTestId('test-login'));

    await waitFor(() => {
      expect(screen.getByTestId('auth-status')).toHaveTextContent('LOGGED_IN');
      expect(screen.getByTestId('username')).toHaveTextContent('admin');
      expect(screen.getByTestId('role')).toHaveTextContent('admin');
    });

    // Test logout
    fireEvent.click(screen.getByTestId('test-logout'));
    expect(screen.getByTestId('auth-status')).toHaveTextContent('LOGGED_OUT');
  });

  it('renders LoginPage with form inputs and 1-click Demo buttons', () => {
    renderWithProviders(<LoginPage />);

    expect(screen.getByText('Local AI Research Lab')).toBeInTheDocument();
    expect(screen.getByText('Hesaba Giriş Yap')).toBeInTheDocument();
    expect(screen.getByLabelText(/Kullanıcı Adı veya E-posta/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/Parola/i)).toBeInTheDocument();

    // Check demo buttons
    expect(screen.getByText('Demo Admin')).toBeInTheDocument();
    expect(screen.getByText('Demo Researcher')).toBeInTheDocument();
  });

  it('renders RegisterPage with role options', () => {
    renderWithProviders(<RegisterPage />);

    expect(screen.getByText('Yeni Araştırmacı Hesabı')).toBeInTheDocument();
    expect(screen.getByLabelText(/Kullanıcı Adı \*/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/E-posta Adresi \*/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/Parola/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/Rol Yetkisi/i)).toBeInTheDocument();
  });

  it('renders ProfilePage unauthenticated prompt when not logged in', async () => {
    renderWithProviders(<ProfilePage />);

    await waitFor(() => {
      expect(screen.getByText('Giriş Yapılması Gerekiyor')).toBeInTheDocument();
    });
  });
});

