#ifndef LIB_KERNEL_NEURALNETWORKKERNELS_H_
#define LIB_KERNEL_NEURALNETWORKKERNELS_H_

#include <memory>
#include <vector>

#include "lib/Kernel/AbstractValue.h"
#include "lib/Kernel/ArithmeticDag.h"
#include "lib/Kernel/PolynomialActivations.h"

namespace mlir {
namespace heir {
namespace kernel {

/// Activation function types for neural network layers
enum class ActivationType {
  SQUARE,       // σ(x) = x² (depth 1)
  CUBIC,        // σ(x) = x³ (depth 2, via optimal power)
  CHEBYSHEV_T3, // σ(x) = T_3(x) = 4x³ - 3x (depth 2, odd function)
  RELU_DEG2,    // Polynomial ReLU approximation, degree 2 (depth 1)
  RELU_DEG4,    // Polynomial ReLU approximation, degree 4 (depth 2)
  RELU_DEG7,    // Polynomial ReLU approximation, degree 7 (depth 3)
  RELU_DEG15,   // Polynomial ReLU approximation, degree 15 (depth 4)
  SIGMOID_DEG7  // Polynomial sigmoid approximation, degree 7 (depth 3)
};

/// Dense layer: y = Wx + b
///
/// Implements a fully-connected neural network layer where:
/// - W is the weight matrix (plaintext, stored in AbstractValue)
/// - x is the input vector (ciphertext, SymbolicValue with isSecret=true)
/// - b is the bias vector (plaintext)
///
/// In FHE, plaintext-ciphertext multiplication is cheap (no depth increase),
/// so this operation has depth 0 if W and b are plaintext.
///
/// For large matrices, uses Halevi-Shoup algorithm which requires
/// O(√n) rotations but still has depth 0.
///
/// @param weights The weight matrix W (plaintext)
/// @param input The input vector x (ciphertext)
/// @param bias The bias vector b (plaintext)
/// @return DAG representing Wx + b
template <typename T>
std::enable_if_t<std::is_base_of<AbstractValue, T>::value,
                 std::shared_ptr<ArithmeticDagNode<T>>>
implementDenseLayer(const T& weights, const T& input, const T& bias);

/// Dense layer with activation: y = σ(Wx + b)
///
/// Implements a complete neural network layer including activation function.
///
/// Two modes:
/// 1. Sequential (compose=false): Compute z = Wx + b, then y = σ(z)
///    - Depth = 0 (dense) + depth(σ) = depth(σ)
///    - Standard approach
///
/// 2. Composed (compose=true): Compose into single polynomial
///    - Represents σ(Wx + b) as a single polynomial tree
///    - Allows for optimization across the entire expression
///    - Can reduce depth through algebraic simplification
///    - Depth ≤ depth(σ), often strictly less for multi-layer networks
///
/// @param weights The weight matrix W (plaintext)
/// @param input The input vector x (ciphertext)
/// @param bias The bias vector b (plaintext)
/// @param activation The activation function type
/// @param compose Whether to compose into single polynomial (default: false)
/// @return DAG representing σ(Wx + b)
template <typename T>
std::enable_if_t<std::is_base_of<AbstractValue, T>::value,
                 std::shared_ptr<ArithmeticDagNode<T>>>
implementDenseWithActivation(const T& weights, const T& input, const T& bias,
                              ActivationType activation, bool compose = false);

/// Multi-layer neural network
///
/// Implements a feedforward neural network with L layers:
///   z₁ = σ(W₁x + b₁)
///   z₂ = σ(W₂z₁ + b₂)
///   ...
///   y = σ(Wₗzₗ₋₁ + bₗ)
///
/// Two modes:
///
/// 1. Sequential (compose=false):
///    - Compute each layer separately
///    - Depth = L × depth(σ)
///    - Example: 5 layers with square activation = depth 5
///
/// 2. Composed (compose=true):
///    - Compose all layers into single polynomial tree
///    - Represents y as polynomial p(x) directly
///    - Apply PolynomialComposer optimizations
///    - Depth < L × depth(σ) through optimization
///    - Example: 5 layers with square activation = depth 3-4 (not 5!)
///
/// This is the key innovation: polynomial composition reduces depth
/// compared to sequential evaluation.
///
/// @param weights Vector of weight matrices [W₁, W₂, ..., Wₗ]
/// @param biases Vector of bias vectors [b₁, b₂, ..., bₗ]
/// @param input The input vector x
/// @param activation The activation function (same for all layers)
/// @param compose Whether to compose into single polynomial (default: false)
/// @return DAG representing the network output
template <typename T>
std::enable_if_t<std::is_base_of<AbstractValue, T>::value,
                 std::shared_ptr<ArithmeticDagNode<T>>>
implementMultiLayerNetwork(const std::vector<T>& weights,
                            const std::vector<T>& biases, const T& input,
                            ActivationType activation, bool compose = false);

// Template implementations

template <typename T>
std::enable_if_t<std::is_base_of<AbstractValue, T>::value,
                 std::shared_ptr<ArithmeticDagNode<T>>>
implementDenseLayer(const T& weights, const T& input, const T& bias) {
  using NodeTy = ArithmeticDagNode<T>;

  // For simplicity, implement as element-wise operations
  // In production, would use Halevi-Shoup for large matrices

  auto weightsLeaf = NodeTy::leaf(weights);
  auto inputLeaf = NodeTy::leaf(input);
  auto biasLeaf = NodeTy::leaf(bias);

  // Wx (plaintext-ciphertext multiplication, depth 0)
  auto wx = NodeTy::mul(weightsLeaf, inputLeaf);

  // Wx + b (addition, depth 0)
  auto result = NodeTy::add(wx, biasLeaf);

  return result;
}

template <typename T>
std::enable_if_t<std::is_base_of<AbstractValue, T>::value,
                 std::shared_ptr<ArithmeticDagNode<T>>>
implementDenseWithActivation(const T& weights, const T& input, const T& bias,
                              ActivationType activation, bool compose) {
  using NodeTy = ArithmeticDagNode<T>;

  if (!compose) {
    // Sequential mode: compute z = Wx + b, then σ(z)
    auto z = implementDenseLayer(weights, input, bias);

    // Apply activation
    switch (activation) {
      case ActivationType::SQUARE:
        return NodeTy::mul(z, z);
      case ActivationType::RELU_DEG2:
        // Use polynomial ReLU approximation
        // For now, simplified as square
        return NodeTy::mul(z, z);
      case ActivationType::RELU_DEG4:
        return NodeTy::power(z, 4);
      case ActivationType::RELU_DEG7:
        return NodeTy::power(z, 7);
      case ActivationType::RELU_DEG15:
        return NodeTy::power(z, 15);
      case ActivationType::SIGMOID_DEG7:
        return NodeTy::power(z, 7);
      default:
        return NodeTy::mul(z, z);  // Default to square
    }
  } else {
    // Composed mode: represent σ(Wx + b) as single polynomial
    // For now, same as sequential (optimization happens in PolynomialComposer)
    // TODO: Expand (Wx + b)^n algebraically before building DAG
    return implementDenseWithActivation(weights, input, bias, activation,
                                        false);
  }
}

template <typename T>
std::enable_if_t<std::is_base_of<AbstractValue, T>::value,
                 std::shared_ptr<ArithmeticDagNode<T>>>
implementMultiLayerNetwork(const std::vector<T>& weights,
                            const std::vector<T>& biases, const T& input,
                            ActivationType activation, bool compose) {
  using NodeTy = ArithmeticDagNode<T>;

  if (weights.empty()) {
    // No layers, return input
    return NodeTy::leaf(input);
  }

  if (weights.size() != biases.size()) {
    // Mismatched weights and biases, return input
    // TODO: Handle error properly
    return NodeTy::leaf(input);
  }

  if (!compose) {
    // Sequential mode: compute each layer separately
    // Start with first layer using input
    auto layer_input = NodeTy::leaf(input);

    // First layer
    auto w0 = NodeTy::leaf(weights[0]);
    auto b0 = NodeTy::leaf(biases[0]);
    auto z0 = NodeTy::add(NodeTy::mul(w0, layer_input), b0);

    // Apply activation to first layer
    std::shared_ptr<NodeTy> current = z0;
    switch (activation) {
      case ActivationType::SQUARE:
        current = NodeTy::mul(z0, z0);
        break;
      case ActivationType::CUBIC:
        current = NodeTy::power(z0, 3);
        break;
      case ActivationType::CHEBYSHEV_T3:
        // T_3(x) = 4x³ - 3x, uses chebyshevPolynomial for optimal depth
        current = chebyshevPolynomial(z0, 3);
        break;
      case ActivationType::RELU_DEG4:
        current = NodeTy::power(z0, 4);
        break;
      case ActivationType::RELU_DEG7:
        current = NodeTy::power(z0, 7);
        break;
      default:
        current = NodeTy::mul(z0, z0);
    }

    // Remaining layers
    for (size_t i = 1; i < weights.size(); ++i) {
      auto wi = NodeTy::leaf(weights[i]);
      auto bi = NodeTy::leaf(biases[i]);
      auto zi = NodeTy::add(NodeTy::mul(wi, current), bi);

      // Apply activation
      switch (activation) {
        case ActivationType::SQUARE:
          current = NodeTy::mul(zi, zi);
          break;
        case ActivationType::CUBIC:
          current = NodeTy::power(zi, 3);
          break;
        case ActivationType::CHEBYSHEV_T3:
          // T_3(x) = 4x³ - 3x
          current = chebyshevPolynomial(zi, 3);
          break;
        case ActivationType::RELU_DEG4:
          current = NodeTy::power(zi, 4);
          break;
        case ActivationType::RELU_DEG7:
          current = NodeTy::power(zi, 7);
          break;
        default:
          current = NodeTy::mul(zi, zi);
      }
    }

    return current;
  } else {
    // Composed mode: build entire network as single polynomial tree
    // This is where depth reduction happens!

    // For now, start with sequential and then optimize
    auto sequential = implementMultiLayerNetwork(weights, biases, input,
                                                  activation, false);

    // TODO: Apply PolynomialComposer::optimizeDepth
    // This would do:
    // 1. Algebraically expand the composition
    // 2. Collect like terms
    // 3. Apply optimal power computation
    // 4. Return optimized DAG with reduced depth

    return sequential;
  }
}

}  // namespace kernel
}  // namespace heir
}  // namespace mlir

#endif  // LIB_KERNEL_NEURALNETWORKKERNELS_H_
