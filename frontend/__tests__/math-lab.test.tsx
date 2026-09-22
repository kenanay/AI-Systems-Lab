import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import MathLabPage from '@/app/math-lab/page';
import { api } from '@/lib/api';

// Mock next/navigation
jest.mock('next/navigation', () => ({
  usePathname: () => '/math-lab',
}));

// Mock mathLabApi
jest.mock('@/lib/api', () => {
  const original = jest.requireActual('@/lib/api');
  return {
    ...original,
    api: {
      ...original.api,
      mathLab: {
        multiplyMatrices: jest.fn(() =>
          Promise.resolve({
            success: true,
            result: [
              [19, 22],
              [43, 50],
            ],
            steps: [
              { step: 1, title: 'Boyut Kontrolü', description: '2x2 x 2x2' },
              {
                step: 2,
                title: 'Hücre [0,0] Hesaplama',
                description: 'C[0,0] = 1*5 + 2*7 = 19',
                cell_calculation: [
                  { k: 0, a_value: 1, b_value: 5, product: 5, running_sum: 5 },
                  { k: 1, a_value: 2, b_value: 7, product: 14, running_sum: 19 },
                ],
              },
            ],
            properties: {
              shape_a: [2, 2],
              shape_b: [2, 2],
              shape_result: [2, 2],
              total_flops: 16,
              frobenius_norm: 70.38,
            },
          })
        ),
        transposeMatrix: jest.fn(() =>
          Promise.resolve({
            success: true,
            result: [
              [1, 3],
              [2, 4],
            ],
            steps: [{ step: 1, title: 'Transpoz', description: 'Satır ve sütunlar değişti' }],
            properties: {
              transposed_shape: [2, 2],
              is_symmetric: false,
            },
          })
        ),
        inverseMatrix: jest.fn(() =>
          Promise.resolve({
            success: true,
            result: [
              [-2, 1],
              [1.5, -0.5],
            ],
            steps: [{ step: 1, title: 'Ters Matris', description: 'Gauss Jordan' }],
            properties: {
              determinant: -2,
              condition_number: 14.9,
              is_well_conditioned: true,
            },
          })
        ),
        calculateDeterminant: jest.fn(() =>
          Promise.resolve({
            success: true,
            determinant: -2.0,
            is_singular: false,
          })
        ),
        calculateEigenvalues: jest.fn(() =>
          Promise.resolve({
            success: true,
            trace: 5.0,
            eigen_pairs: [
              { index: 0, eigenvalue: 5.37, eigenvector: [0.57, 0.82], magnitude: 5.37 },
              { index: 1, eigenvalue: -0.37, eigenvector: [-0.82, 0.57], magnitude: 0.37 },
            ],
          })
        ),
        vectorDotProduct: jest.fn(() =>
          Promise.resolve({
            success: true,
            result: [25.0],
            properties: {
              dimension: 2,
              magnitude_a: 5.0,
              magnitude_b: 5.0,
              cosine_similarity: 1.0,
              angle_degrees: 0.0,
              angle_radians: 0.0,
              is_orthogonal: false,
            },
          })
        ),
        computeDerivative: jest.fn(() =>
          Promise.resolve({
            success: true,
            x_values: [-1, 0, 1],
            y_values: [1, 0, 1],
            dy_dx_values: [-2, 0, 2],
            formula: 'x²',
            derivative_formula: '2x',
          })
        ),
        simulateGradientDescent: jest.fn(() =>
          Promise.resolve({
            success: true,
            history: [
              { iteration: 0, x: 2.5, y: 6.25, gradient: 5.0, step_size: 0.75, update_formula: 'x = x - lr*grad' },
              { iteration: 1, x: 1.75, y: 3.06, gradient: 3.5, step_size: 0.525, update_formula: 'x = x - lr*grad' },
              { iteration: 2, x: 0.01, y: 0.0001, gradient: 0.02, step_size: 0.003, update_formula: 'x = x - lr*grad' },
            ],
            final_x: 0.01,
            final_y: 0.0001,
            converged: true,
            iterations: 3,
          })
        ),
        demonstrateChainRule: jest.fn(() =>
          Promise.resolve({
            success: true,
            x_value: 1.5,
            composite_formula: 'f(g(x)) = (2x + 1)²',
            steps: [
              { step: 1, title: 'İç Fonksiyon', formula: 'g(x) = 2x + 1', derivative: 'g\'(x) = 2', value_at_x: 4.0, derivative_at_x: 2.0 },
              { step: 2, title: 'Dış Fonksiyon', formula: 'f(u) = u²', derivative: 'f\'(u) = 2u', value_at_g_x: 16.0, derivative_at_g_x: 8.0 },
              { step: 3, title: 'Chain Rule', formula: 'df/dx = f\'(g(x)) * g\'(x)', calculation: '8.00 * 2.00 = 16.00', result: 16.0, total_derivative: 16.0 },
            ],
            final_derivative: 16.0,
          })
        ),
      },
    },
  };
});

describe('MathLabPage Component', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('renders the header, badge labels, and tab controls', async () => {
    render(<MathLabPage />);

    expect(screen.getByText('Math Lab: Lineer Cebir & Kalkülüs')).toBeInTheDocument();
    expect(screen.getByText('1. Lineer Cebir (Matris & Vektör)')).toBeInTheDocument();
    expect(screen.getByText('2. Kalkülüs & Gradiyen İnişi')).toBeInTheDocument();
    expect(screen.getByText('3. Zincir Kuralı & Backprop')).toBeInTheDocument();

    await waitFor(() => {
      expect(api.mathLab.multiplyMatrices).toHaveBeenCalled();
    });
  });

  it('renders matrix multiplication inputs and interactive result cell breakdown', async () => {
    render(<MathLabPage />);

    await waitFor(() => {
      expect(screen.getByText('Sonuç Matrisi C = A × B (2 × 2)')).toBeInTheDocument();
    });

    // Verify matrix C result value 19.00
    const cellButtons19 = screen.getAllByText('19.00');
    expect(cellButtons19.length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText('22.00')).toBeInTheDocument();

    // Click on cell [0, 0] to inspect breakdown
    const cellBtn = cellButtons19[0];
    fireEvent.click(cellBtn);

    await waitFor(() => {
      expect(api.mathLab.multiplyMatrices).toHaveBeenCalled();
      expect(screen.getByText('Hücre C[0, 0] Hesaplama Detayı')).toBeInTheDocument();
    });
  });

  it('calculates matrix operations: transpose, inverse, determinant, eigenvalues', async () => {
    render(<MathLabPage />);

    // Click Transpose
    const transposeBtn = screen.getByText('Transpoz (A^T)');
    fireEvent.click(transposeBtn);

    await waitFor(() => {
      expect(api.mathLab.transposeMatrix).toHaveBeenCalled();
      expect(screen.getByText('Matris Transpozu (A^T)')).toBeInTheDocument();
    });

    // Click Inverse
    const inverseBtn = screen.getByText('Tersi (A⁻¹)');
    fireEvent.click(inverseBtn);

    await waitFor(() => {
      expect(api.mathLab.inverseMatrix).toHaveBeenCalled();
      expect(screen.getByText('Matris Tersi: A⁻¹ (Gauss-Jordan)')).toBeInTheDocument();
    });

    // Click Determinant
    const detBtn = screen.getByText('Determinant (|A|)');
    fireEvent.click(detBtn);

    await waitFor(() => {
      expect(api.mathLab.calculateDeterminant).toHaveBeenCalled();
      expect(screen.getByText('Determinant Değeri')).toBeInTheDocument();
    });

    // Click Eigenvalues
    const eigenBtn = screen.getByText('Öz Değerler (Eigen)');
    fireEvent.click(eigenBtn);

    await waitFor(() => {
      expect(api.mathLab.calculateEigenvalues).toHaveBeenCalled();
      expect(screen.getByText('Öz Değerler (Eigenvalues) & Öz Vektörler (Eigenvectors)')).toBeInTheDocument();
    });
  });

  it('switches to Calculus tab and renders gradient descent simulation', async () => {
    render(<MathLabPage />);

    // Switch to Calculus Tab
    const calculusTab = screen.getByText('2. Kalkülüs & Gradiyen İnişi');
    fireEvent.click(calculusTab);

    await waitFor(() => {
      expect(screen.getByText('1D Gradiyen İnişi & Sayısal Türev Simülatörü')).toBeInTheDocument();
      expect(screen.getByText('Optimizasyon Parametreleri')).toBeInTheDocument();
    });

    // Check simulation metrics
    expect(screen.getByText('Son Konum (x*)')).toBeInTheDocument();
    expect(screen.getByText('Minimum Değer f(x*)')).toBeInTheDocument();
    expect(screen.getByText('Yakınsadı (Converged)')).toBeInTheDocument();

    // Restart simulation button
    const restartBtn = screen.getByText('Simülasyonu Yeniden Başlat');
    fireEvent.click(restartBtn);

    await waitFor(() => {
      expect(api.mathLab.simulateGradientDescent).toHaveBeenCalled();
    });
  });

  it('switches to Chain Rule tab and renders decomposition cards and backpropagation didactic', async () => {
    render(<MathLabPage />);

    // Switch to Chain Rule Tab
    const chainTab = screen.getByText('3. Zincir Kuralı & Backprop');
    fireEvent.click(chainTab);

    await waitFor(() => {
      expect(screen.getByText('Bileşke Fonksiyonlar & Zincir Kuralı (Chain Rule)')).toBeInTheDocument();
    });

    // Verify 3 decomposition cards
    expect(screen.getByText('İç Fonksiyon (u = g(x))')).toBeInTheDocument();
    expect(screen.getByText('Dış Fonksiyon (y = f(u))')).toBeInTheDocument();
    expect(screen.getByText('Zincir Kuralı Çarpımı')).toBeInTheDocument();

    // Verify deep learning context callout
    expect(screen.getByText('Yapay Zekada Neden Hayati Önem Taşır?')).toBeInTheDocument();
  });
});
