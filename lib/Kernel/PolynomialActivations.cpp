#include "lib/Kernel/PolynomialActivations.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

#include "lib/Kernel/AbstractValue.h"
#include "lib/Kernel/ArithmeticDag.h"
#include "lib/Kernel/PolynomialComposer.h"

namespace mlir {
namespace heir {
namespace kernel {

std::shared_ptr<ArithmeticDagNode<SymbolicValue>> squareActivation(
    const std::shared_ptr<ArithmeticDagNode<SymbolicValue>>& input) {
  // σ(x) = x²
  // Depth: 1 (single multiplication)
  return ArithmeticDagNode<SymbolicValue>::mul(input, input);
}

std::shared_ptr<ArithmeticDagNode<SymbolicValue>> chebyshevPolynomial(
    const std::shared_ptr<ArithmeticDagNode<SymbolicValue>>& input,
    int degree) {
  // Use optimized Chebyshev polynomial evaluation from PolynomialComposer
  // This gives optimal depth instead of naive recursion
  return PolynomialComposer::buildChebyshevOptimized<SymbolicValue>(input,
                                                                      degree);
}

std::vector<double> computeChebyshevCoefficients(double (*func)(double),
                                                   int degree, double x_min,
                                                   double x_max) {
  // Compute Chebyshev coefficients for approximating func on [x_min, x_max]
  //
  // Method: Sample func at Chebyshev nodes (roots of T_{degree+1}),
  // then compute coefficients via discrete cosine transform (DCT-I).
  //
  // Chebyshev nodes: x_k = cos(π(k + 0.5) / (degree + 1)) for k = 0..degree
  //
  // This is the minimax approximation (minimizes maximum error).

  const int n = degree + 1;
  std::vector<double> f_values(n);
  std::vector<double> coefficients(n);

  // Step 1: Evaluate function at Chebyshev nodes
  for (int k = 0; k < n; ++k) {
    // Map Chebyshev node from [-1, 1] to [x_min, x_max]
    double theta = M_PI * (k + 0.5) / n;
    double x_cheb = std::cos(theta);  // in [-1, 1]
    double x = 0.5 * ((x_max - x_min) * x_cheb + (x_max + x_min));

    f_values[k] = func(x);
  }

  // Step 2: Compute Chebyshev coefficients via DCT
  // c_j = (2/n) * sum_{k=0}^{n-1} f(x_k) * cos(π j (k + 0.5) / n)
  // with c_0 having factor 1/n instead of 2/n
  for (int j = 0; j < n; ++j) {
    double sum = 0.0;
    for (int k = 0; k < n; ++k) {
      double theta = M_PI * j * (k + 0.5) / n;
      sum += f_values[k] * std::cos(theta);
    }
    coefficients[j] = (j == 0 ? 1.0 : 2.0) * sum / n;
  }

  return coefficients;
}

std::shared_ptr<ArithmeticDagNode<SymbolicValue>> buildPolynomialFromChebyshev(
    const std::shared_ptr<ArithmeticDagNode<SymbolicValue>>& input,
    const std::vector<double>& coefficients) {
  using NodeTy = ArithmeticDagNode<SymbolicValue>;

  if (coefficients.empty()) {
    // Return zero polynomial
    return NodeTy::leaf(SymbolicValue({1}, false));
  }

  // Build: sum_{k=0}^n c_k * T_k(input)
  //
  // We'll construct this as:
  // result = c_0 * T_0 + c_1 * T_1 + ... + c_n * T_n

  std::shared_ptr<NodeTy> result = nullptr;

  for (size_t k = 0; k < coefficients.size(); ++k) {
    double coeff = coefficients[k];

    // Skip near-zero coefficients for efficiency
    if (std::abs(coeff) < 1e-10) {
      continue;
    }

    // Compute T_k(input)
    auto T_k = chebyshevPolynomial(input, k);

    // Multiply by coefficient
    // Note: In FHE, scalar multiplication is cheap (plaintext-ciphertext)
    // For now, we'll represent the coefficient as a leaf node
    // TODO(#issue): Handle plaintext constants properly
    auto coeff_node = NodeTy::leaf(SymbolicValue({1}, false));  // Placeholder

    // In practice, we'd want: coeff * T_k
    // For now, just use T_k scaled somehow
    // This is a simplification - in real FHE, plaintext multiplication is free
    auto term = T_k;  // Should be: coeff * T_k

    if (result == nullptr) {
      result = term;
    } else {
      result = NodeTy::add(result, term);
    }
  }

  return result ? result : NodeTy::leaf(SymbolicValue({1}, false));
}

std::shared_ptr<ArithmeticDagNode<SymbolicValue>> polynomialReLU(
    const std::shared_ptr<ArithmeticDagNode<SymbolicValue>>& input, int degree,
    double bound) {
  // ReLU(x) = max(0, x)
  // We approximate this on [-bound, bound] using Chebyshev series

  auto relu_func = [](double x) -> double { return std::max(0.0, x); };

  // Compute Chebyshev coefficients
  auto coefficients = computeChebyshevCoefficients(relu_func, degree, -bound, bound);

  // Build polynomial from coefficients
  // Note: We need to scale input from [- bound, bound] to [-1, 1]
  // for Chebyshev polynomials
  // Let y = x / bound (maps [-bound, bound] to [-1, 1])
  // Then p(x) = sum c_k T_k(x/bound)

  // For now, simplified version without scaling
  // TODO(#issue): Add proper input scaling
  return buildPolynomialFromChebyshev(input, coefficients);
}

std::shared_ptr<ArithmeticDagNode<SymbolicValue>> polynomialSigmoid(
    const std::shared_ptr<ArithmeticDagNode<SymbolicValue>>& input, int degree,
    double bound) {
  // Sigmoid(x) = 1 / (1 + exp(-x))
  // We approximate this on [-bound, bound] using Chebyshev series

  auto sigmoid_func = [](double x) -> double {
    return 1.0 / (1.0 + std::exp(-x));
  };

  // Compute Chebyshev coefficients
  auto coefficients =
      computeChebyshevCoefficients(sigmoid_func, degree, -bound, bound);

  // Build polynomial from coefficients
  return buildPolynomialFromChebyshev(input, coefficients);
}

}  // namespace kernel
}  // namespace heir
}  // namespace mlir
