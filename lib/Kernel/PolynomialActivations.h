#ifndef LIB_KERNEL_POLYNOMIALACTIVATIONS_H_
#define LIB_KERNEL_POLYNOMIALACTIVATIONS_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "lib/Kernel/AbstractValue.h"
#include "lib/Kernel/ArithmeticDag.h"

namespace mlir {
namespace heir {
namespace kernel {

/// Polynomial activation functions for neural networks in FHE.
///
/// These functions provide polynomial approximations of common neural network
/// activation functions. Since FHE can only compute polynomials (addition and
/// multiplication), non-polynomial functions like ReLU, sigmoid, and tanh must
/// be approximated using low-degree polynomials to minimize multiplicative
/// depth.

/// Square activation function: σ(x) = x²
///
/// This is the simplest non-linear activation with depth 1. While it doesn't
/// approximate any standard activation function, it provides non-linearity for
/// neural networks with minimal depth cost.
///
/// @param input The input DAG node
/// @return DAG representing x²
std::shared_ptr<ArithmeticDagNode<SymbolicValue>> squareActivation(
    const std::shared_ptr<ArithmeticDagNode<SymbolicValue>>& input);

/// Chebyshev polynomial of degree n evaluated at input.
///
/// Chebyshev polynomials T_n(x) are optimal for polynomial approximation
/// (minimize maximum error). They satisfy the recurrence:
///   T_0(x) = 1
///   T_1(x) = x
///   T_{n+1}(x) = 2x·T_n(x) - T_{n-1}(x)
///
/// These are used as basis functions for approximating other functions.
///
/// @param input The input DAG node
/// @param degree The degree of the Chebyshev polynomial (must be >= 0)
/// @return DAG representing T_degree(input)
std::shared_ptr<ArithmeticDagNode<SymbolicValue>> chebyshevPolynomial(
    const std::shared_ptr<ArithmeticDagNode<SymbolicValue>>& input,
    int degree);

/// Polynomial approximation of ReLU activation function.
///
/// ReLU(x) = max(0, x) cannot be computed exactly in FHE. This function
/// returns a polynomial approximation of ReLU using Chebyshev series
/// expansion. The approximation is optimized for the interval [-bound, bound].
///
/// Typical approximation errors (L2 over [-5, 5]):
///   degree 2:  ~2.5 (poor)
///   degree 4:  ~0.8 (acceptable)
///   degree 7:  ~0.08 (good)
///   degree 15: ~0.005 (excellent)
///
/// Depth cost: ceil(log2(degree))
///   degree 2:  depth 1
///   degree 4:  depth 2
///   degree 7:  depth 3
///   degree 15: depth 4
///
/// @param input The input DAG node
/// @param degree The degree of the approximating polynomial
/// @param bound The input range to optimize for (approximation is good on
///              [-bound, bound])
/// @return DAG representing polynomial approximation of ReLU(input)
std::shared_ptr<ArithmeticDagNode<SymbolicValue>> polynomialReLU(
    const std::shared_ptr<ArithmeticDagNode<SymbolicValue>>& input, int degree,
    double bound = 5.0);

/// Polynomial approximation of sigmoid activation function.
///
/// Sigmoid(x) = 1 / (1 + exp(-x)) is approximated using a polynomial.
/// The approximation is optimized for the interval [-bound, bound].
///
/// @param input The input DAG node
/// @param degree The degree of the approximating polynomial
/// @param bound The input range to optimize for
/// @return DAG representing polynomial approximation of sigmoid(input)
std::shared_ptr<ArithmeticDagNode<SymbolicValue>> polynomialSigmoid(
    const std::shared_ptr<ArithmeticDagNode<SymbolicValue>>& input, int degree,
    double bound = 5.0);

/// Compute Chebyshev coefficients for approximating a target function.
///
/// Given a function f, computes coefficients c_0, ..., c_n such that:
///   f(x) ≈ sum_{k=0}^n c_k T_k(x)
///
/// where T_k are Chebyshev polynomials.
///
/// This is used internally by polynomialReLU, polynomialSigmoid, etc.
///
/// @param func Target function to approximate (takes double, returns double)
/// @param degree Degree of approximating polynomial
/// @param x_min Minimum of approximation interval
/// @param x_max Maximum of approximation interval
/// @return Vector of Chebyshev coefficients [c_0, c_1, ..., c_n]
std::vector<double> computeChebyshevCoefficients(
    double (*func)(double), int degree, double x_min = -1.0,
    double x_max = 1.0);

/// Build polynomial from Chebyshev coefficients.
///
/// Given coefficients c_0, ..., c_n, constructs DAG representing:
///   sum_{k=0}^n c_k T_k(input)
///
/// where T_k are Chebyshev polynomials.
///
/// @param input The input DAG node
/// @param coefficients Chebyshev coefficients
/// @return DAG representing the polynomial
std::shared_ptr<ArithmeticDagNode<SymbolicValue>> buildPolynomialFromChebyshev(
    const std::shared_ptr<ArithmeticDagNode<SymbolicValue>>& input,
    const std::vector<double>& coefficients);

}  // namespace kernel
}  // namespace heir
}  // namespace mlir

#endif  // LIB_KERNEL_POLYNOMIALACTIVATIONS_H_
