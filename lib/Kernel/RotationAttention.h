#ifndef LIB_KERNEL_ROTATIONATTENTION_H_
#define LIB_KERNEL_ROTATIONATTENTION_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "lib/Kernel/AbstractValue.h"
#include "lib/Kernel/ArithmeticDag.h"

namespace mlir {
namespace heir {
namespace kernel {

// ============================================================================
// ROTATION-BASED ATTENTION FOR FHE
// ============================================================================
//
// This file implements FHE-native attention mechanisms based on rotations
// instead of softmax. Key insight: rotations are automorphisms of the
// cyclotomic polynomial ring and can be computed with depth-0 (plaintext-
// ciphertext multiplication only).
//
// Theoretical Foundation:
// - Standard attention: Attention(Q, K, V) = softmax(QK^T/√d) · V
//   Problem: softmax breaks homomorphism (transcendental function, division)
//   Depth: ~5 (with polynomial approximation of softmax)
//
// - Rotation attention: Attention(V) = Σ_k α_k · rotate(V, k)
//   where α_k are learned plaintext weights, k are rotation amounts
//   Depth: 0 (plaintext-ciphertext mul)
//   Property: EXACTLY equivariant to rotations!
//
// Connection to Galois Theory:
// - Rotations implement automorphisms of ℚ(ζ_n) where ζ_n is a root of unity
// - Galois group Gal(ℚ(ζ_n)/ℚ) ≅ (ℤ/nℤ)* is the group of rotations
// - Multi-rotation attention computes linear combinations over this group
//
// See: lib/Kernel/AttentionTheory.md for detailed mathematical analysis
// ============================================================================

/// Multi-rotation attention: Σ_k α_k · rotate(values, k)
///
/// This is the core FHE-native attention primitive. It computes a weighted
/// sum of rotated values, where rotations correspond to automorphisms of
/// the cyclotomic polynomial ring.
///
/// Properties:
/// - **Depth**: 0 (plaintext-ciphertext multiplication only!)
/// - **Rotations**: O(num_rotations)
/// - **Equivariant**: MultiRotAttention(rotate(V, j)) = rotate(MultiRotAttention(V), j)
///
/// Template parameters:
///   T: Value type (e.g., SymbolicValue for analysis)
///
/// Arguments:
///   values: Input tensor [seq_len, d_model] as ArithmeticDagNode
///   weights: Plaintext rotation weights {α_k} (one per rotation amount)
///   rotation_amounts: Rotation amounts {k} (e.g., {1, 2, 4, 8})
///
/// Returns:
///   ArithmeticDagNode representing: Σ_k α_k · leftRotate(values, k)
///
/// Example:
///   ```cpp
///   SymbolicValue V({16, 64}, true);  // [seq_len=16, d_model=64]
///   std::vector<double> weights = {0.4, 0.3, 0.2, 0.1};
///   std::vector<int64_t> rotations = {1, 2, 4, 8};
///
///   auto attn = multiRotationAttention(
///       ArithmeticDagNode<SymbolicValue>::leaf(V),
///       weights,
///       rotations
///   );
///
///   // Analysis
///   MultiplicativeDepthVisitor depth_visitor;
///   int64_t depth = depth_visitor.process(attn);  // Should be 0!
///
///   RotationCountVisitor rot_visitor;
///   int64_t num_rots = rot_visitor.process(attn);  // Should be 4
///   ```
template <typename T>
std::enable_if_t<std::is_base_of<AbstractValue, T>::value,
                 std::shared_ptr<ArithmeticDagNode<T>>>
multiRotationAttention(
    const std::shared_ptr<ArithmeticDagNode<T>>& values,
    const std::vector<double>& weights,
    const std::vector<int64_t>& rotation_amounts) {
  using NodeTy = ArithmeticDagNode<T>;

  assert(values && "Invalid values node");
  assert(weights.size() == rotation_amounts.size() &&
         "Weights and rotation amounts must have same length");
  assert(!weights.empty() && "Must have at least one rotation");

  // Initialize result with first weighted rotation
  auto result = NodeTy::mul(
      NodeTy::leftRotate(values, rotation_amounts[0]),
      NodeTy::constantScalar(weights[0]));

  // Add remaining weighted rotations
  for (size_t i = 1; i < rotation_amounts.size(); ++i) {
    auto rotated = NodeTy::leftRotate(values, rotation_amounts[i]);
    auto weighted = NodeTy::mul(rotated, NodeTy::constantScalar(weights[i]));
    result = NodeTy::add(result, weighted);
  }

  return result;
}

/// Sparse local window attention using rotations.
///
/// Implements attention restricted to a local window:
///   Attention(i) = Σ_{|j-i| ≤ window/2} α_j · V[i+j]
///
/// This is efficient for tasks where local context dominates (e.g.,
/// language modeling, vision with local receptive fields).
///
/// Arguments:
///   values: Input tensor
///   weights: Plaintext weights for each position in window
///   window_size: Size of attention window (must be odd)
///
/// Returns:
///   ArithmeticDagNode for local window attention
///
/// Example:
///   ```cpp
///   // Window size 5: attend to [-2, -1, 0, 1, 2]
///   std::vector<double> weights = {0.1, 0.2, 0.4, 0.2, 0.1};
///   auto attn = localWindowAttention(values_node, weights, 5);
///   ```
template <typename T>
std::enable_if_t<std::is_base_of<AbstractValue, T>::value,
                 std::shared_ptr<ArithmeticDagNode<T>>>
localWindowAttention(
    const std::shared_ptr<ArithmeticDagNode<T>>& values,
    const std::vector<double>& weights,
    int64_t window_size) {
  assert(window_size % 2 == 1 && "Window size must be odd");
  assert(static_cast<int64_t>(weights.size()) == window_size &&
         "Must have one weight per window position");

  // Generate rotation amounts: [-w/2, ..., -1, 0, 1, ..., w/2]
  std::vector<int64_t> rotations;
  int64_t half_window = window_size / 2;
  for (int64_t k = -half_window; k <= half_window; ++k) {
    rotations.push_back(k);
  }

  return multiRotationAttention(values, weights, rotations);
}

/// Strided attention using rotations.
///
/// Implements attention that samples every s-th position:
///   Attention(i) = Σ_{j=0,s,2s,...} α_j · V[i+j]
///
/// Useful for capturing long-range dependencies with sparse sampling.
///
/// Arguments:
///   values: Input tensor
///   weights: Plaintext weights for each strided position
///   stride: Stride amount
///   num_positions: Number of positions to attend to
///
/// Returns:
///   ArithmeticDagNode for strided attention
template <typename T>
std::enable_if_t<std::is_base_of<AbstractValue, T>::value,
                 std::shared_ptr<ArithmeticDagNode<T>>>
stridedAttention(
    const std::shared_ptr<ArithmeticDagNode<T>>& values,
    const std::vector<double>& weights,
    int64_t stride,
    int64_t num_positions) {
  assert(static_cast<int64_t>(weights.size()) == num_positions &&
         "Must have one weight per strided position");

  // Generate rotation amounts: {0, stride, 2*stride, ...}
  std::vector<int64_t> rotations;
  for (int64_t i = 0; i < num_positions; ++i) {
    rotations.push_back(i * stride);
  }

  return multiRotationAttention(values, weights, rotations);
}

/// Cyclotomic position encoding for FHE.
///
/// Standard transformers use sinusoidal position encoding:
///   PE(pos, 2i) = sin(pos / 10000^(2i/d))
///   PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
///
/// These require transcendental functions (expensive in FHE).
///
/// Cyclotomic encoding uses roots of unity:
///   PE(pos, i) = ζ_n^(pos · i)
///
/// where ζ_n = e^(2πi/n) is a primitive n-th root of unity.
///
/// Why this is FHE-native:
/// 1. Roots of unity are native to cyclotomic polynomial rings ℚ(ζ_n)
/// 2. Rotations naturally implement: rotate(PE(pos, i), k) = PE(pos+k, i)
/// 3. No approximation needed - exact representation!
/// 4. Depth: 0 (position encodings are plaintext constants)
///
/// Arguments:
///   position: Position index
///   d_model: Model dimension
///   max_seq_len: Maximum sequence length (for normalization)
///
/// Returns:
///   Vector of doubles representing cyclotomic position encoding
///
/// Note: This returns a plaintext vector. In actual FHE deployment, these
/// would be encoded as plaintext constants and added to ciphertext inputs.
///
/// Example:
///   ```cpp
///   auto pe_0 = cyclotomicPositionEncoding(0, 64, 16);
///   auto pe_5 = cyclotomicPositionEncoding(5, 64, 16);
///   ```
inline std::vector<double> cyclotomicPositionEncoding(
    int64_t position,
    int64_t d_model,
    int64_t max_seq_len) {
  std::vector<double> encoding(d_model);

  const double TWO_PI = 6.283185307179586;

  for (int64_t i = 0; i < d_model; ++i) {
    // ζ_n^(pos · i) where n = max_seq_len
    // Real part: cos(2π * pos * i / n)
    // Imag part: sin(2π * pos * i / n) (not used in this representation)

    double angle = TWO_PI * position * i / max_seq_len;

    // Use cosine for real-valued encoding
    // In full complex representation, would also use sine
    encoding[i] = std::cos(angle);
  }

  return encoding;
}

/// Multi-head rotation attention.
///
/// Implements multiple rotation attention heads in parallel, analogous to
/// multi-head attention in transformers. Each head can learn different
/// rotation patterns.
///
/// Example patterns:
/// - Head 1: Local window {-2, -1, 0, 1, 2}
/// - Head 2: Strided {0, 4, 8, 12, ...}
/// - Head 3: Powers of 2 {1, 2, 4, 8, 16}
///
/// Arguments:
///   values: Input tensor [seq_len, d_model]
///   weights_per_head: Plaintext weights for each head (one vector per head)
///   rotations_per_head: Rotation amounts for each head
///   d_head: Dimension per head (d_model / n_heads)
///
/// Returns:
///   ArithmeticDagNode representing concatenation of all heads
///
/// Note: This is a simplified version. Full implementation would need proper
/// tensor slicing/concatenation operations in ArithmeticDag.
template <typename T>
std::enable_if_t<std::is_base_of<AbstractValue, T>::value,
                 std::vector<std::shared_ptr<ArithmeticDagNode<T>>>>
multiHeadRotationAttention(
    const std::shared_ptr<ArithmeticDagNode<T>>& values,
    const std::vector<std::vector<double>>& weights_per_head,
    const std::vector<std::vector<int64_t>>& rotations_per_head,
    int64_t d_head) {
  assert(weights_per_head.size() == rotations_per_head.size() &&
         "Number of heads must match between weights and rotations");

  std::vector<std::shared_ptr<ArithmeticDagNode<T>>> head_outputs;
  int64_t n_heads = weights_per_head.size();

  for (int64_t h = 0; h < n_heads; ++h) {
    // Each head processes its slice of the input
    // (In full implementation, would slice values[:, h*d_head:(h+1)*d_head])
    auto head_output = multiRotationAttention(
        values,
        weights_per_head[h],
        rotations_per_head[h]);

    head_outputs.push_back(head_output);
  }

  return head_outputs;
}

/// Full rotation-based transformer layer.
///
/// Implements: V = x·W_V; Attention(V); Output = Attn·W_O
///
/// This builds a complete layer using rotation attention that can be analyzed
/// for depth, rotation count, and other FHE cost metrics.
///
/// Arguments:
///   input: Input node [seq_len, d_model]
///   W_V: Value projection matrix (plaintext)
///   W_O: Output projection matrix (plaintext)
///   attention_weights: Rotation weights
///   rotation_amounts: Rotation amounts
///
/// Returns:
///   ArithmeticDagNode for complete layer
///
/// Expected depth: 2 (1 for V projection + 0 for rotation attention + 1 for output)
/// But if W_V and W_O are plaintext: depth 0!
template <typename T>
std::enable_if_t<std::is_base_of<AbstractValue, T>::value,
                 std::shared_ptr<ArithmeticDagNode<T>>>
rotationTransformerLayer(
    const std::shared_ptr<ArithmeticDagNode<T>>& input,
    const T& W_V,
    const T& W_O,
    const std::vector<double>& attention_weights,
    const std::vector<int64_t>& rotation_amounts) {
  using NodeTy = ArithmeticDagNode<T>;

  // Value projection: V = input · W_V
  // If input is ciphertext and W_V is plaintext: depth 0
  auto V = NodeTy::mul(input, NodeTy::leaf(W_V));

  // Multi-rotation attention: Attn = Σ_k α_k · rotate(V, k)
  // Depth: 0 (plaintext-ciphertext mul)
  auto attn_output = multiRotationAttention(V, attention_weights, rotation_amounts);

  // Output projection: output = Attn · W_O
  // If attn_output is ciphertext and W_O is plaintext: depth 0
  auto output = NodeTy::mul(attn_output, NodeTy::leaf(W_O));

  return output;
}

}  // namespace kernel
}  // namespace heir
}  // namespace mlir

#endif  // LIB_KERNEL_ROTATIONATTENTION_H_
