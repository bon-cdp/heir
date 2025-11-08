# Theoretical Analysis: FHE-Native Attention via Rotations

**Authors**: Research exploration for HEIR
**Date**: 2025-11-06
**Status**: Experimental - Theory Development

## Abstract

This document provides rigorous mathematical analysis of attention mechanisms in Fully Homomorphic Encryption (FHE). We prove that standard attention does not preserve under homomorphic encoding due to the softmax non-linearity, and propose rotation-based attention as an FHE-native alternative grounded in Galois theory.

---

## 1. Homomorphism Breaking: A Rigorous Proof

### 1.1 Setup and Definitions

**Definition 1.1 (FHE Encoding Map)**:
Let φ: ℝ → R_q = ℤ_q[x]/(Φ_n(x)) be the FHE encoding map, where:
- ℝ is the plaintext space (real numbers)
- R_q is the FHE ciphertext space (cyclotomic polynomial ring)
- Φ_n(x) is the n-th cyclotomic polynomial (typically Φ_n(x) = x^n + 1 for n = 2^k)
- q is the ciphertext modulus

**Definition 1.2 (Ring Homomorphism)**:
φ is a ring homomorphism if and only if:
1. φ(a + b) = φ(a) + φ(b) for all a, b ∈ ℝ
2. φ(a · b) = φ(a) · φ(b) for all a, b ∈ ℝ
3. φ(1) = 1

**Note**: In practice, φ is only approximately homomorphic due to noise growth, but we consider the idealized case.

### 1.2 Attention Decomposition

**Definition 1.3 (Standard Attention)**:
For query Q ∈ ℝ^{n×d}, key K ∈ ℝ^{n×d}, value V ∈ ℝ^{n×d}:

```
Attention(Q, K, V) = softmax(QK^T / √d) · V
```

where softmax is applied row-wise:
```
softmax(x)_i = exp(x_i) / Σ_j exp(x_j)
```

**Decomposition into operations**:
1. **Score computation**: S = QK^T  [matrix multiplication]
2. **Scaling**: S' = S / √d  [scalar division]
3. **Normalization**: A = softmax(S')  [non-linear activation]
4. **Output**: O = A · V  [matrix multiplication]

### 1.3 Homomorphism Check

**Lemma 1.1 (Linear Operations Preserve)**:
Matrix multiplication and scalar operations preserve under φ:
- φ(QK^T) = φ(Q) · φ(K)^T  ✓
- φ(S / √d) = φ(S) / φ(√d)  ✓ (plaintext scalar)

**Proof**: Follows directly from ring homomorphism properties. □

**Theorem 1.1 (Softmax Breaks Homomorphism)**:
The softmax function does NOT preserve under homomorphic encoding:

```
φ(softmax(x)) ≠ softmax(φ(x))
```

**Proof**:
Consider softmax(x)_i = exp(x_i) / Σ_j exp(x_j).

*Part 1: exp is not polynomial*
- exp(x) = Σ_{k=0}^∞ x^k / k! is transcendental (infinite series)
- FHE can only compute polynomials of bounded degree
- Therefore, φ(exp(x)) cannot be computed exactly in FHE

*Part 2: Division by ciphertext not supported*
- softmax requires division by Σ_j exp(x_j)
- This sum is a ciphertext (encrypted value)
- FHE schemes (CKKS, BGV, BFV) do NOT support ciphertext-ciphertext division
- Only multiplication is homomorphic

*Part 3: Constructive counterexample*
Let x = [1, 2] ∈ ℝ²

Plaintext:
```
softmax([1, 2]) = [exp(1)/(exp(1)+exp(2)), exp(2)/(exp(1)+exp(2))]
                 ≈ [0.269, 0.731]
φ([0.269, 0.731]) = [φ(0.269), φ(0.731)]
```

Ciphertext:
```
φ([1, 2]) = [φ(1), φ(2)]
softmax([φ(1), φ(2)]) = [exp(φ(1))/(exp(φ(1))+exp(φ(2))), ...]
```
But exp(φ(1)) is not computable in FHE! □

**Corollary 1.1**:
Standard attention cannot be computed homomorphically without approximation.

### 1.4 Approximation Error Bound

**Polynomial Approximation of Softmax**:
We can approximate softmax with degree-d Chebyshev polynomial p_d(x):
```
softmax(x) ≈ p_d(x)
```

**Theorem 1.2 (Approximation Error)**:
Let ε_approx = max_x ||softmax(x) - p_d(x)||.
Then the total attention error satisfies:
```
||Attention(Q,K,V) - Attention_FHE(φ(Q),φ(K),φ(V))|| ≤ ε_approx · ||V||
```

**Proof**:
```
Attention_FHE = p_d(φ(QK^T/√d)) · φ(V)
Attention = softmax(QK^T/√d) · V

||Attention - Attention_FHE||
  = ||(softmax - p_d)(QK^T/√d) · V||
  ≤ ||softmax - p_d|| · ||V||
  = ε_approx · ||V||
```
□

**Consequence**: Approximation error compounds through layers in deep networks!

---

## 2. Rotation-Based Attention: FHE-Native Design

### 2.1 Mathematical Foundation

**Definition 2.1 (FHE Rotation)**:
A rotation τ_k: R_q → R_q is defined as:
```
τ_k(a(x)) = a(x · ζ_n^k)
```
where ζ_n = e^{2πi/n} is a primitive n-th root of unity.

In the coefficient representation a(x) = Σ_i a_i x^i:
```
τ_k([a_0, a_1, ..., a_{n-1}]) = [a_{n-k}, a_{n-k+1}, ..., a_{n-k-1}]
```
(cyclic left shift by k positions in SIMD slots)

**Lemma 2.1 (Rotations are Automorphisms)**:
τ_k is a ring automorphism:
1. τ_k(a + b) = τ_k(a) + τ_k(b)
2. τ_k(a · b) = τ_k(a) · τ_k(b)
3. τ_k(1) = 1

**Proof**: Substitution x → x · ζ_n^k is a ring automorphism. □

### 2.2 Connection to Galois Theory

**Theorem 2.1 (Galois Group of Cyclotomic Field)**:
The Galois group of the cyclotomic field extension ℚ(ζ_n)/ℚ is:
```
Gal(ℚ(ζ_n)/ℚ) ≅ (ℤ/nℤ)* ≅ {τ_k : gcd(k, n) = 1}
```

**Proof**: See Pinter Chapter 31 (Galois Theory). □

**Interpretation**: Every automorphism of the cyclotomic field IS a rotation!
This is why FHE schemes based on Ring-LWE naturally support rotations.

### 2.3 Multi-Rotation Attention

**Definition 2.2 (Multi-Rotation Attention)**:
For value tensor V ∈ R_q^n (ciphertext), rotation set K ⊂ {0,1,...,n-1},
and plaintext weights {α_k : k ∈ K}:

```
MultiRotAttention(V; {α_k}, K) = Σ_{k∈K} α_k · τ_k(V)
```

**Key properties**:
1. **Linearity**: This is a linear combination (preserves homomorphism!)
2. **Depth**: If α_k are plaintext, depth = 0 (plaintext-ciphertext mul)
3. **Rotations**: |K| rotation operations required
4. **Expressiveness**: Can represent any linear combination of rotated values

### 2.4 Equivariance Property

**Theorem 2.2 (Rotation Equivariance)**:
Multi-rotation attention is equivariant to rotations:
```
MultiRotAttention(τ_j(V)) = τ_j(MultiRotAttention(V))
```

**Proof**:
```
MultiRotAttention(τ_j(V))
  = Σ_k α_k · τ_k(τ_j(V))
  = Σ_k α_k · τ_{k+j}(V)         [automorphisms compose]
  = τ_j(Σ_k α_k · τ_k(V))        [pull out common rotation]
  = τ_j(MultiRotAttention(V))
```
□

**Interpretation**: Rotating the input is equivalent to rotating the output!
This is the fundamental property of permutation-equivariant functions.

### 2.5 Expressiveness Analysis

**Question**: Can multi-rotation attention approximate standard attention?

**Theorem 2.3 (Cyclic Convolution Representation)**:
Any cyclic convolution can be exactly represented:
```
y_i = Σ_j w_{i-j} · v_j  =  Σ_k α_k · τ_k(V)[i]
```
where α_k are the Fourier coefficients of the kernel w.

**Proof**: This is the convolution theorem. □

**Limitation**: Standard attention is NOT a cyclic convolution!
```
y_i = Σ_j softmax(q_i·k_j) · v_j
```
The weights depend on content (q_i·k_j), not just position (i-j).

**Corollary 2.1**:
Multi-rotation attention with fixed weights CANNOT represent standard attention.

**Open Question**: Can learned α_k(Q, K) (data-dependent weights) approximate attention?

---

## 3. Sparse Attention via Rotations

### 3.1 Local Window Attention

**Definition 3.1 (Local Window Attention)**:
Restrict attention to a window of size w:
```
Attention_local(Q, K, V)[i] = Σ_{|j-i| ≤ w/2} softmax_j(q_i·k_j) · v_j
```

**FHE Implementation**:
```
K = {-w/2, ..., -1, 0, 1, ..., w/2}
MultiRotAttention(V; {α_k}, K)
```
Only w+1 rotations needed!

**Depth**: 1 (one multiplication per rotation)
**Rotations**: O(w)
**Expressiveness**: Good for local dependencies (e.g., language modeling)

### 3.2 Strided Attention

**Definition 3.2 (Strided Attention)**:
Attend to every s-th token:
```
K = {0, s, 2s, 3s, ..., ⌊n/s⌋·s}
```

**Rotations**: O(n/s)
**Use case**: Long-range dependencies with sparse sampling

### 3.3 Multi-Head Rotation Attention

**Definition 3.3**:
Multiple rotation patterns in parallel (like multi-head attention):
```
Head_h(V) = MultiRotAttention(V; {α_k^h}, K_h)
MultiHeadRotAttention(V) = Concat(Head_1(V), ..., Head_H(V))
```

**Each head can learn different rotation patterns**:
- Head 1: Local window (K = {-2, -1, 0, 1, 2})
- Head 2: Strided (K = {0, 4, 8, 12, ...})
- Head 3: Power-of-2 (K = {1, 2, 4, 8, 16, ...})

---

## 4. Cyclotomic Position Encoding

### 4.1 Standard Position Encoding (Not FHE-Friendly)

Transformers use sinusoidal position encoding:
```
PE(pos, 2i)   = sin(pos / 10000^{2i/d})
PE(pos, 2i+1) = cos(pos / 10000^{2i/d})
```

**Problems for FHE**:
- Transcendental functions (sin, cos)
- Requires polynomial approximation
- Additional depth cost

### 4.2 Cyclotomic Position Encoding (FHE-Native!)

**Definition 4.1**:
```
PE_cyclotomic(pos, i) = ζ_n^{pos · i}
```
where ζ_n = e^{2πi/n} is the primitive n-th root of unity.

**Why this is elegant**:

1. **Already in the ring**: ζ_n is THE fundamental element of ℚ(ζ_n)!
2. **Rotation-compatible**:
   ```
   τ_k(PE(pos, i)) = PE(pos + k, i)
   ```
   Rotations naturally shift positions!

3. **Galois automorphisms**: The position encoding IS a Galois automorphism
   ```
   σ_{pos}(x) = x^{pos}  (in cyclotomic field)
   ```

4. **No approximation**: Exact representation, no depth cost!

**Lemma 4.1 (Rotation of Position Encoding)**:
```
τ_k(PE_cyclotomic(pos, i)) = PE_cyclotomic(pos + k, i)
```

**Proof**:
```
τ_k(ζ_n^{pos·i}) = (ζ_n^k)^{pos·i} = ζ_n^{(pos+k)·i}
```
□

**Consequence**: Position encoding and rotations are the SAME algebraic structure!

### 4.3 Learnable Cyclotomic Encoding

**Generalization**:
```
PE(pos) = Σ_i w_i · ζ_n^{pos · i}
```
where w_i are learned plaintext weights.

This allows the model to learn which "frequencies" (powers of ζ_n) are important.

---

## 5. QK^T Computation in FHE

### 5.1 The Challenge

Standard attention requires:
```
S[i,j] = q_i · k_j  for all i, j ∈ {1,...,n}
```

This is n² dot products with ciphertext-ciphertext multiplication.

**Cost**:
- **Depth**: 1 (all dot products in parallel)
- **Ciphertexts**: n² (one per attention score!)
- **Memory**: O(n²) ciphertexts - HUGE!

### 5.2 Halevi-Shoup Algorithm (Matrix-Vector)

For **plaintext** matrix W ∈ ℝ^{m×n}, **ciphertext** vector v ∈ R_q^n:
```
y = W · v
```

**Algorithm**:
1. Encode W into √n diagonal matrices: D_1, ..., D_{√n}
2. For each D_i:
   - Multiply: temp_i = D_i ⊙ v (element-wise, using SIMD)
   - Rotate and accumulate
3. Combine results

**Cost**: O(√n) rotations, depth 0 (plaintext matrix!)

### 5.3 Extending to Ciphertext-Ciphertext (Open Problem!)

For **ciphertext** Q, K ∈ R_q^{n×d}, computing QK^T is challenging:
- Both are encrypted!
- Halevi-Shoup doesn't directly apply

**Potential approaches**:

**Option A: Sparse Patterns**
Only compute scores for local window or stride pattern:
- S[i,j] only for |i-j| ≤ w
- Reduces to O(w·n) dot products
- Still depth 1, but fewer ciphertexts

**Option B: Rotation-Based Approximation**
Skip QK^T entirely! Use multi-rotation attention:
```
Output = Σ_k α_k · τ_k(V)
```
where α_k are learned (not data-dependent).

**Option C: Hybrid**
- Compute Q·K^T in plaintext (offline, on training data)
- Learn typical attention patterns
- Transfer to multi-rotation weights for FHE inference

---

## 6. Depth Analysis

### 6.1 Standard Attention with Polynomial Softmax

**Operations**:
1. Q·K^T: depth 1 (ct-ct mul)
2. Chebyshev_d softmax: depth ⌈log₂ d⌉
3. Attn·V: depth 1 (ct-ct mul)

**Total**: depth ≈ 2 + ⌈log₂ d⌉

For degree-7 Chebyshev: depth ≈ 2 + 3 = 5

### 6.2 Multi-Rotation Attention

**Operations**:
1. τ_k(V) for each k: depth 0 (rotation is free*)
2. α_k · τ_k(V): depth 0 (plaintext-ciphertext mul)
3. Σ_k results: depth 0 (addition)

**Total depth: 0** (!!!!)

*Note: Rotations have computational cost but don't increase noise/depth.

### 6.3 Depth Comparison

| Method | Depth | Rotations | Expressiveness |
|--------|-------|-----------|----------------|
| Standard + poly softmax | 5 | O(n²) | Full attention |
| Multi-rotation (fixed) | 0 | O(K) | Cyclic convolution only |
| Multi-rotation (learned) | 0 | O(K) | Approximates attention? |
| Sparse (window w) | 1 | O(w·n) | Local attention |

**Trade-off**: Depth vs. Expressiveness!

---

## 7. Open Research Questions

### 7.1 Expressiveness of Rotation-Based Attention

**Question**: Can learned multi-rotation attention approximate standard attention well enough for practical tasks?

**Empirical approach**:
- Train standard transformer on task T
- Extract attention patterns: analyze learned A = softmax(QK^T)
- Approximate with multi-rotation: find {α_k, K} such that ||A·V - Σ_k α_k·τ_k(V)|| is small
- Transfer to FHE inference

### 7.2 Data-Dependent Rotation Weights

**Question**: Can we make α_k depend on Q, K while staying FHE-friendly?

**Idea**:
```
α_k = polynomial(⟨Q, K⟩)
```
where ⟨Q, K⟩ is some aggregate feature (e.g., mean of Q·K).

**Challenge**: This adds depth! Need careful design.

### 7.3 Optimal Rotation Patterns

**Question**: What rotation sets K are optimal for different tasks?

**Hypothesis**:
- Language modeling: Local windows + strided
- Vision: Grid patterns (2D rotations)
- Time series: Power-of-2 dilations

### 7.4 Theoretical Expressiveness Bounds

**Question**: Can we prove impossibility results?

**Conjecture**: Multi-rotation attention with fixed weights cannot approximate content-based attention to arbitrary precision.

**Approach**: Information-theoretic argument - fixed rotations cannot capture O(n²) degrees of freedom.

---

## 8. Conclusion

**Main Results**:

1. **Proven**: Softmax breaks homomorphism (Theorem 1.1)
2. **Designed**: Multi-rotation attention as FHE-native alternative (Definition 2.2)
3. **Grounded**: Connection to Galois theory and cyclotomic fields (Theorem 2.1)
4. **Proposed**: Cyclotomic position encoding (Definition 4.1)
5. **Analyzed**: Depth trade-offs (Section 6)

**Next Steps**:

1. **Empirical**: Implement and test rotation-based attention on simple tasks
2. **Theoretical**: Prove expressiveness bounds
3. **Practical**: Optimize rotation patterns for specific applications
4. **Hybrid**: Combine rotations with polynomial approximations

**Impact**:

If rotation-based attention works well empirically, it could enable:
- **Transformers in FHE** with depth 0 attention!
- **Efficient inference** on encrypted data
- **Privacy-preserving LLMs** for sensitive applications

This is genuinely novel research territory.
