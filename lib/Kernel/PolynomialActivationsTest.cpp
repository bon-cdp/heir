#include <cmath>
#include <memory>

#include "gtest/gtest.h"
#include "lib/Kernel/AbstractValue.h"
#include "lib/Kernel/ArithmeticDag.h"
#include "lib/Kernel/MultiplicativeDepthVisitor.h"
#include "lib/Kernel/PolynomialActivations.h"

namespace mlir {
namespace heir {
namespace kernel {
namespace {

TEST(PolynomialActivationsTest, SquareActivation) {
  // Test σ(x) = x²
  SymbolicValue x({4}, true);  // Ciphertext input
  auto xLeaf = ArithmeticDagNode<SymbolicValue>::leaf(x);

  auto squared = squareActivation(xLeaf);

  // Verify depth is 1 (one multiplication)
  MultiplicativeDepthVisitor visitor;
  int64_t depth = visitor.process(squared);

  EXPECT_EQ(depth, 1);
}

TEST(PolynomialActivationsTest, ChebyshevPolynomialDegree0) {
  // T_0(x) = 1 (constant)
  SymbolicValue x({4}, true);
  auto xLeaf = ArithmeticDagNode<SymbolicValue>::leaf(x);

  auto T_0 = chebyshevPolynomial(xLeaf, 0);

  // T_0 should have depth 0 (it's a constant)
  MultiplicativeDepthVisitor visitor;
  int64_t depth = visitor.process(T_0);

  EXPECT_EQ(depth, 0);
}

TEST(PolynomialActivationsTest, ChebyshevPolynomialDegree1) {
  // T_1(x) = x (identity)
  SymbolicValue x({4}, true);
  auto xLeaf = ArithmeticDagNode<SymbolicValue>::leaf(x);

  auto T_1 = chebyshevPolynomial(xLeaf, 1);

  // T_1 should be the same as input
  // Depth should be 0 (no multiplications)
  MultiplicativeDepthVisitor visitor;
  int64_t depth = visitor.process(T_1);

  EXPECT_EQ(depth, 0);
}

TEST(PolynomialActivationsTest, ChebyshevPolynomialDegree2) {
  // T_2(x) = 2x² - 1
  // Using recurrence: T_2(x) = 2x·T_1(x) - T_0(x) = 2x·x - 1 = 2x² - 1
  SymbolicValue x({4}, true);
  auto xLeaf = ArithmeticDagNode<SymbolicValue>::leaf(x);

  auto T_2 = chebyshevPolynomial(xLeaf, 2);

  // T_2 involves one multiplication (x²)
  // Depth should be 1
  MultiplicativeDepthVisitor visitor;
  int64_t depth = visitor.process(T_2);

  EXPECT_EQ(depth, 1);
}

TEST(PolynomialActivationsTest, ChebyshevPolynomialDegree3) {
  // T_3(x) = 4x³ - 3x
  // Using recurrence: T_3(x) = 2x·T_2(x) - T_1(x)
  SymbolicValue x({4}, true);
  auto xLeaf = ArithmeticDagNode<SymbolicValue>::leaf(x);

  auto T_3 = chebyshevPolynomial(xLeaf, 3);

  // T_3 involves x³, which requires depth 2
  // (x² at depth 1, then x²·x at depth 2)
  MultiplicativeDepthVisitor visitor;
  int64_t depth = visitor.process(T_3);

  EXPECT_EQ(depth, 2);
}

TEST(PolynomialActivationsTest, ChebyshevPolynomialDegree4) {
  // T_4(x) = 8x⁴ - 8x² + 1
  SymbolicValue x({4}, true);
  auto xLeaf = ArithmeticDagNode<SymbolicValue>::leaf(x);

  auto T_4 = chebyshevPolynomial(xLeaf, 4);

  // T_4 involves x⁴, which requires depth 2 with optimal evaluation
  // (compute x², then (x²)²)
  MultiplicativeDepthVisitor visitor;
  int64_t depth = visitor.process(T_4);

  // Depth should be 2
  EXPECT_EQ(depth, 2);
}

TEST(PolynomialActivationsTest, PolynomialReLUDepthDegree2) {
  // Degree-2 polynomial approximation of ReLU
  SymbolicValue x({4}, true);
  auto xLeaf = ArithmeticDagNode<SymbolicValue>::leaf(x);

  auto approx_relu = polynomialReLU(xLeaf, 2);

  // Degree-2 polynomial should have depth 1
  MultiplicativeDepthVisitor visitor;
  int64_t depth = visitor.process(approx_relu);

  EXPECT_LE(depth, 1);  // Should be at most depth 1
}

TEST(PolynomialActivationsTest, PolynomialReLUDepthDegree7) {
  // Degree-7 polynomial approximation of ReLU
  SymbolicValue x({4}, true);
  auto xLeaf = ArithmeticDagNode<SymbolicValue>::leaf(x);

  auto approx_relu = polynomialReLU(xLeaf, 7);

  // Degree-7 polynomial should have depth ceil(log2(7)) = 3
  MultiplicativeDepthVisitor visitor;
  int64_t depth = visitor.process(approx_relu);

  EXPECT_LE(depth, 3);  // Should be at most depth 3
}

TEST(PolynomialActivationsTest, PolynomialSigmoidDepth) {
  // Polynomial approximation of sigmoid
  SymbolicValue x({4}, true);
  auto xLeaf = ArithmeticDagNode<SymbolicValue>::leaf(x);

  auto approx_sigmoid = polynomialSigmoid(xLeaf, 7);

  // Degree-7 polynomial should have depth at most 3
  MultiplicativeDepthVisitor visitor;
  int64_t depth = visitor.process(approx_sigmoid);

  EXPECT_LE(depth, 3);
}

TEST(PolynomialActivationsTest, ComputeChebyshevCoefficientsReLU) {
  // Test that we can compute Chebyshev coefficients for ReLU
  auto relu_func = [](double x) -> double { return std::max(0.0, x); };

  auto coefficients = computeChebyshevCoefficients(relu_func, 7, -5.0, 5.0);

  // Should return 8 coefficients (degree 7 = 8 coefficients)
  EXPECT_EQ(coefficients.size(), 8);

  // Coefficients should be non-trivial (not all zero)
  double sum_abs = 0.0;
  for (double c : coefficients) {
    sum_abs += std::abs(c);
  }
  EXPECT_GT(sum_abs, 0.1);  // Should have some non-zero coefficients
}

TEST(PolynomialActivationsTest, ComputeChebyshevCoefficientsSigmoid) {
  // Test that we can compute Chebyshev coefficients for sigmoid
  auto sigmoid_func = [](double x) -> double {
    return 1.0 / (1.0 + std::exp(-x));
  };

  auto coefficients = computeChebyshevCoefficients(sigmoid_func, 7, -5.0, 5.0);

  // Should return 8 coefficients
  EXPECT_EQ(coefficients.size(), 8);

  // Coefficients should be non-trivial
  double sum_abs = 0.0;
  for (double c : coefficients) {
    sum_abs += std::abs(c);
  }
  EXPECT_GT(sum_abs, 0.1);
}

TEST(PolynomialActivationsTest, SquareActivationComposition) {
  // Test composing square activations: (x²)² = x⁴
  SymbolicValue x({4}, true);
  auto xLeaf = ArithmeticDagNode<SymbolicValue>::leaf(x);

  auto squared_once = squareActivation(xLeaf);
  auto squared_twice = squareActivation(squared_once);

  // x⁴ should have depth 2
  MultiplicativeDepthVisitor visitor;
  int64_t depth = visitor.process(squared_twice);

  EXPECT_EQ(depth, 2);
}

TEST(PolynomialActivationsTest, ComparePolynomialDegrees) {
  // Compare depth of different polynomial degrees
  SymbolicValue x({4}, true);
  auto xLeaf = ArithmeticDagNode<SymbolicValue>::leaf(x);

  MultiplicativeDepthVisitor visitor;

  // Degree 2: depth 1
  auto relu_deg2 = polynomialReLU(xLeaf, 2);
  int64_t depth_2 = visitor.process(relu_deg2);

  // Degree 4: depth 2
  auto relu_deg4 = polynomialReLU(xLeaf, 4);
  int64_t depth_4 = visitor.process(relu_deg4);

  // Degree 7: depth 3
  auto relu_deg7 = polynomialReLU(xLeaf, 7);
  int64_t depth_7 = visitor.process(relu_deg7);

  // Higher degree should have equal or greater depth
  EXPECT_LE(depth_2, depth_4);
  EXPECT_LE(depth_4, depth_7);

  // Specific bounds
  EXPECT_LE(depth_2, 1);
  EXPECT_LE(depth_4, 2);
  EXPECT_LE(depth_7, 3);
}

}  // namespace
}  // namespace kernel
}  // namespace heir
}  // namespace mlir
