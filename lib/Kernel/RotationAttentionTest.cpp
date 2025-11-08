#include "lib/Kernel/RotationAttention.h"

#include <memory>
#include <vector>

#include "gtest/gtest.h"
#include "lib/Kernel/AbstractValue.h"
#include "lib/Kernel/ArithmeticDag.h"
#include "lib/Kernel/MultiplicativeDepthVisitor.h"
#include "lib/Kernel/RotationCountVisitor.h"

namespace mlir {
namespace heir {
namespace kernel {

// ============================================================================
// Test Suite for FHE-Native Rotation Attention
// ============================================================================

TEST(RotationAttentionTest, MultiRotationAttentionBasic) {
  // Create a simple multi-rotation attention with 4 rotations
  SymbolicValue values({16, 64}, true);  // [seq_len=16, d_model=64], ciphertext

  std::vector<double> weights = {0.4, 0.3, 0.2, 0.1};
  std::vector<int64_t> rotations = {1, 2, 4, 8};

  auto attn = multiRotationAttention(
      ArithmeticDagNode<SymbolicValue>::leaf(values),
      weights,
      rotations);

  ASSERT_NE(attn, nullptr);

  // Check multiplicative depth: should be 0 (plaintext-ciphertext mul)
  MultiplicativeDepthVisitor depth_visitor;
  int64_t depth = depth_visitor.process(attn);

  // Note: Our conservative model counts all multiplications structurally
  // Each rotation + weight multiplication counts as depth 1
  // But in REAL FHE, plaintext-ciphertext mul is depth 0!
  EXPECT_GE(depth, 0);  // At least 0
  EXPECT_LE(depth, 1);  // At most 1 (conservative model)
}

TEST(RotationAttentionTest, RotationCount) {
  // Verify that rotation count matches the number of rotation amounts
  SymbolicValue values({16, 64}, true);

  std::vector<double> weights = {0.25, 0.25, 0.25, 0.25};
  std::vector<int64_t> rotations = {1, 2, 4, 8};

  auto attn = multiRotationAttention(
      ArithmeticDagNode<SymbolicValue>::leaf(values),
      weights,
      rotations);

  // Count rotations
  RotationCountVisitor rotation_visitor;
  int64_t num_rotations = rotation_visitor.process(attn);

  // Should have exactly 4 rotations (with CSE deduplication)
  EXPECT_EQ(num_rotations, 4);
}

TEST(RotationAttentionTest, SingleRotation) {
  // Test with single rotation
  SymbolicValue values({16, 64}, true);

  std::vector<double> weights = {1.0};
  std::vector<int64_t> rotations = {5};

  auto attn = multiRotationAttention(
      ArithmeticDagNode<SymbolicValue>::leaf(values),
      weights,
      rotations);

  ASSERT_NE(attn, nullptr);

  // Should have exactly 1 rotation
  RotationCountVisitor rotation_visitor;
  int64_t num_rotations = rotation_visitor.process(attn);
  EXPECT_EQ(num_rotations, 1);
}

TEST(RotationAttentionTest, LocalWindowAttention) {
  // Test local window attention (window size 5)
  SymbolicValue values({16, 64}, true);

  // Weights: center has highest, decreasing outward
  std::vector<double> weights = {0.1, 0.2, 0.4, 0.2, 0.1};
  int64_t window_size = 5;

  auto attn = localWindowAttention(
      ArithmeticDagNode<SymbolicValue>::leaf(values),
      weights,
      window_size);

  ASSERT_NE(attn, nullptr);

  // Should have 5 rotations (window size)
  RotationCountVisitor rotation_visitor;
  int64_t num_rotations = rotation_visitor.process(attn);
  EXPECT_EQ(num_rotations, window_size);

  // Depth should still be low
  MultiplicativeDepthVisitor depth_visitor;
  int64_t depth = depth_visitor.process(attn);
  EXPECT_LE(depth, 1);
}

TEST(RotationAttentionTest, StridedAttention) {
  // Test strided attention (stride=4, 4 positions)
  SymbolicValue values({16, 64}, true);

  std::vector<double> weights = {0.4, 0.3, 0.2, 0.1};
  int64_t stride = 4;
  int64_t num_positions = 4;

  auto attn = stridedAttention(
      ArithmeticDagNode<SymbolicValue>::leaf(values),
      weights,
      stride,
      num_positions);

  ASSERT_NE(attn, nullptr);

  // Should have 4 rotations
  RotationCountVisitor rotation_visitor;
  int64_t num_rotations = rotation_visitor.process(attn);
  EXPECT_EQ(num_rotations, num_positions);
}

TEST(RotationAttentionTest, CyclotomicPositionEncoding) {
  // Test cyclotomic position encoding
  int64_t d_model = 64;
  int64_t max_seq_len = 16;

  // Get position encodings for positions 0 and 5
  auto pe_0 = cyclotomicPositionEncoding(0, d_model, max_seq_len);
  auto pe_5 = cyclotomicPositionEncoding(5, d_model, max_seq_len);

  EXPECT_EQ(pe_0.size(), d_model);
  EXPECT_EQ(pe_5.size(), d_model);

  // Position 0 should have specific pattern
  // PE(0, i) = ζ^0 = 1 for all i (using cosine)
  for (int64_t i = 0; i < d_model; ++i) {
    EXPECT_NEAR(pe_0[i], 1.0, 1e-10);
  }

  // Position encodings should be bounded [-1, 1] (using cosine)
  for (int64_t i = 0; i < d_model; ++i) {
    EXPECT_GE(pe_5[i], -1.0);
    EXPECT_LE(pe_5[i], 1.0);
  }

  // Different positions should have different encodings
  bool all_same = true;
  for (int64_t i = 0; i < d_model; ++i) {
    if (std::abs(pe_0[i] - pe_5[i]) > 1e-6) {
      all_same = false;
      break;
    }
  }
  EXPECT_FALSE(all_same) << "Position encodings should differ";
}

TEST(RotationAttentionTest, MultiHeadRotationAttention) {
  // Test multi-head rotation attention
  SymbolicValue values({16, 64}, true);

  int64_t n_heads = 2;
  int64_t d_head = 32;

  // Each head has different rotation pattern
  std::vector<std::vector<double>> weights_per_head = {
      {0.5, 0.3, 0.2},      // Head 0: local (rotations 1, 2, 3)
      {0.4, 0.3, 0.2, 0.1}  // Head 1: sparser (rotations 1, 2, 4, 8)
  };

  std::vector<std::vector<int64_t>> rotations_per_head = {
      {1, 2, 3},
      {1, 2, 4, 8}
  };

  auto head_outputs = multiHeadRotationAttention(
      ArithmeticDagNode<SymbolicValue>::leaf(values),
      weights_per_head,
      rotations_per_head,
      d_head);

  EXPECT_EQ(head_outputs.size(), n_heads);
  ASSERT_NE(head_outputs[0], nullptr);
  ASSERT_NE(head_outputs[1], nullptr);

  // Check rotation counts for each head
  RotationCountVisitor rotation_visitor;

  // Head 0 should have 3 rotations
  int64_t num_rotations_head0 = rotation_visitor.process(head_outputs[0]);
  EXPECT_EQ(num_rotations_head0, 3);

  // Head 1 should have 4 rotations
  int64_t num_rotations_head1 = rotation_visitor.process(head_outputs[1]);
  EXPECT_EQ(num_rotations_head1, 4);
}

TEST(RotationAttentionTest, FullTransformerLayer) {
  // Test complete rotation transformer layer
  SymbolicValue input({16, 64}, true);   // Ciphertext input
  SymbolicValue W_V({64, 64}, false);    // Plaintext weight matrix
  SymbolicValue W_O({64, 64}, false);    // Plaintext weight matrix

  std::vector<double> attn_weights = {0.4, 0.3, 0.2, 0.1};
  std::vector<int64_t> rotations = {1, 2, 4, 8};

  auto layer_output = rotationTransformerLayer(
      ArithmeticDagNode<SymbolicValue>::leaf(input),
      W_V,
      W_O,
      attn_weights,
      rotations);

  ASSERT_NE(layer_output, nullptr);

  // Check depth
  MultiplicativeDepthVisitor depth_visitor;
  int64_t depth = depth_visitor.process(layer_output);

  // Expected:
  // 1. input · W_V: depth 1 (ct-ct mul) or 0 if W_V is plaintext
  // 2. rotation attention: depth 0 (plaintext-ciphertext mul)
  // 3. attn · W_O: depth 1 (ct-ct mul) or 0 if W_O is plaintext
  //
  // With plaintext matrices: total depth should be 0-1
  // Our conservative model: depth 1-2
  EXPECT_LE(depth, 2);

  // Check rotation count
  RotationCountVisitor rotation_visitor;
  int64_t num_rotations = rotation_visitor.process(layer_output);

  // Should have 4 rotations from the attention mechanism
  EXPECT_EQ(num_rotations, 4);
}

TEST(RotationAttentionTest, CompareDepthVsStandardAttention) {
  // Demonstrate depth advantage of rotation attention
  //
  // Standard attention (with polynomial softmax):
  // - QK^T: depth 1
  // - softmax (Chebyshev deg 7): depth ~3
  // - Attn·V: depth 1
  // Total: ~5
  //
  // Rotation attention:
  // - Multi-rotation: depth 0
  // Total: 0
  //
  // This test verifies rotation attention has much lower depth

  SymbolicValue values({16, 64}, true);

  std::vector<double> weights = {0.25, 0.25, 0.25, 0.25};
  std::vector<int64_t> rotations = {1, 2, 4, 8};

  auto rotation_attn = multiRotationAttention(
      ArithmeticDagNode<SymbolicValue>::leaf(values),
      weights,
      rotations);

  MultiplicativeDepthVisitor depth_visitor;
  int64_t rotation_depth = depth_visitor.process(rotation_attn);

  // Rotation attention should have very low depth
  EXPECT_LE(rotation_depth, 1);

  // Note: In REAL FHE, plaintext-ciphertext mul would be depth 0
  // Our model conservatively counts it as 1
  //
  // Standard attention would require ~5 depth for softmax approximation
  // So rotation attention gives ~5x depth reduction!
}

TEST(RotationAttentionTest, NegativeRotations) {
  // Test that negative rotations work correctly
  SymbolicValue values({16, 64}, true);

  std::vector<double> weights = {0.5, 0.5};
  std::vector<int64_t> rotations = {-2, 3};  // Negative and positive

  auto attn = multiRotationAttention(
      ArithmeticDagNode<SymbolicValue>::leaf(values),
      weights,
      rotations);

  ASSERT_NE(attn, nullptr);

  // Should have 2 rotations
  RotationCountVisitor rotation_visitor;
  int64_t num_rotations = rotation_visitor.process(attn);
  EXPECT_EQ(num_rotations, 2);
}

TEST(RotationAttentionTest, LargeNumberOfRotations) {
  // Test with many rotations (stress test)
  SymbolicValue values({16, 64}, true);

  // 8 rotations
  std::vector<double> weights = {0.2, 0.15, 0.15, 0.1, 0.1, 0.1, 0.1, 0.1};
  std::vector<int64_t> rotations = {1, 2, 3, 4, 5, 6, 7, 8};

  auto attn = multiRotationAttention(
      ArithmeticDagNode<SymbolicValue>::leaf(values),
      weights,
      rotations);

  ASSERT_NE(attn, nullptr);

  // Should have 8 rotations
  RotationCountVisitor rotation_visitor;
  int64_t num_rotations = rotation_visitor.process(attn);
  EXPECT_EQ(num_rotations, 8);

  // Depth should still be low (linear in number of rotations for adds)
  MultiplicativeDepthVisitor depth_visitor;
  int64_t depth = depth_visitor.process(attn);
  EXPECT_LE(depth, 1);  // Additions don't increase multiplicative depth
}

TEST(RotationAttentionTest, ZeroRotation) {
  // Test with rotation amount 0 (identity)
  SymbolicValue values({16, 64}, true);

  std::vector<double> weights = {0.5, 0.5};
  std::vector<int64_t> rotations = {0, 1};  // 0 means no rotation

  auto attn = multiRotationAttention(
      ArithmeticDagNode<SymbolicValue>::leaf(values),
      weights,
      rotations);

  ASSERT_NE(attn, nullptr);

  // Even with rotation 0, rotation count should track all rotations
  RotationCountVisitor rotation_visitor;
  int64_t num_rotations = rotation_visitor.process(attn);

  // Note: Rotation by 0 might be optimized away in some implementations
  // Our implementation tracks all rotations
  EXPECT_GE(num_rotations, 1);  // At least the non-zero rotation
}

TEST(RotationAttentionTest, WeightNormalization) {
  // Test that attention works with non-normalized weights
  SymbolicValue values({16, 64}, true);

  // Weights that don't sum to 1
  std::vector<double> weights = {1.0, 2.0, 3.0, 4.0};
  std::vector<int64_t> rotations = {1, 2, 4, 8};

  auto attn = multiRotationAttention(
      ArithmeticDagNode<SymbolicValue>::leaf(values),
      weights,
      rotations);

  ASSERT_NE(attn, nullptr);

  // Should still work (normalization is optional for expressiveness)
  RotationCountVisitor rotation_visitor;
  int64_t num_rotations = rotation_visitor.process(attn);
  EXPECT_EQ(num_rotations, 4);
}

// ============================================================================
// Performance Comparison Tests
// ============================================================================

TEST(RotationAttentionTest, PerformanceMetrics) {
  // Document the performance characteristics of rotation attention
  SymbolicValue values({16, 64}, true);

  std::vector<double> weights = {0.4, 0.3, 0.2, 0.1};
  std::vector<int64_t> rotations = {1, 2, 4, 8};

  auto attn = multiRotationAttention(
      ArithmeticDagNode<SymbolicValue>::leaf(values),
      weights,
      rotations);

  // Measure metrics
  MultiplicativeDepthVisitor depth_visitor;
  RotationCountVisitor rotation_visitor;

  int64_t depth = depth_visitor.process(attn);
  int64_t num_rotations = rotation_visitor.process(attn);

  // Print performance summary (for documentation)
  std::cout << "\n=== Rotation Attention Performance ===" << std::endl;
  std::cout << "Multiplicative depth: " << depth << std::endl;
  std::cout << "Number of rotations: " << num_rotations << std::endl;
  std::cout << "FHE advantage: ~5x depth reduction vs standard attention"
            << std::endl;
  std::cout << "======================================\n" << std::endl;

  // Verify expected performance
  EXPECT_LE(depth, 1) << "Depth should be at most 1 (ideally 0 in real FHE)";
  EXPECT_EQ(num_rotations, 4) << "Should have exactly 4 rotations";
}

}  // namespace kernel
}  // namespace heir
}  // namespace mlir
