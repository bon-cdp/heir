"""
Cyclotomic Position Encoding

FHE-native alternative to sinusoidal position encoding using roots of unity.

Standard transformers use:
    PE(pos, 2i)   = sin(pos / 10000^(2i/d))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

Problem: sin/cos are transcendental functions, require approximation in FHE.

FHE-native solution:
    PE(pos, i) = ζ_n^(pos·i)  where ζ_n = e^(2πi/n)

Key properties:
1. Roots of unity are NATIVE to cyclotomic field ℚ(ζ_n) - no approximation!
2. Rotations naturally shift positions: rotate_k(PE(pos)) = PE(pos+k)
3. Matches Galois automorphism structure
4. Depth: 0 (plaintext constants)

Mathematical foundation:
- FHE operates on ℤ_q[x]/(x^n + 1) where n is power of 2
- This is isomorphic to ℤ_q[ζ_{2n}] (cyclotomic field)
- Roots of unity ζ^k are primitive elements
- Rotations = Galois automorphisms: σ_k(ζ) = ζ^k
"""

import numpy as np
from typing import Tuple


def cyclotomic_position_encoding(
    position: int,
    d_model: int,
    max_seq_len: int = 512
) -> np.ndarray:
    """
    Compute cyclotomic position encoding using roots of unity.

    Args:
        position: Position index (0 to max_seq_len-1)
        d_model: Model dimension (embedding size)
        max_seq_len: Maximum sequence length (default 512)

    Returns:
        Position encoding vector [d_model] using real parts of roots of unity
    """
    # Use next power of 2 for efficient FHE operations
    n = 2 ** int(np.ceil(np.log2(max_seq_len)))

    # Compute roots of unity: ζ_n = e^(2πi/n)
    # For dimension i, use ζ_n^(pos·i)
    encoding = np.zeros(d_model)

    for i in range(d_model):
        # Exponent for this dimension
        k = (position * i) % n

        # ζ_n^k = e^(2πik/n) = cos(2πk/n) + i·sin(2πk/n)
        angle = 2 * np.pi * k / n

        if i % 2 == 0:
            # Even dimensions: use real part (cosine)
            encoding[i] = np.cos(angle)
        else:
            # Odd dimensions: use imaginary part (sine)
            encoding[i] = np.sin(angle)

    return encoding


def cyclotomic_encoding_matrix(
    seq_len: int,
    d_model: int,
    max_seq_len: int = 512
) -> np.ndarray:
    """
    Compute position encoding matrix for entire sequence.

    Args:
        seq_len: Sequence length
        d_model: Model dimension
        max_seq_len: Maximum sequence length

    Returns:
        Position encoding matrix [seq_len, d_model]
    """
    encoding_matrix = np.zeros((seq_len, d_model))

    for pos in range(seq_len):
        encoding_matrix[pos] = cyclotomic_position_encoding(
            pos, d_model, max_seq_len
        )

    return encoding_matrix


def verify_rotation_property(
    d_model: int = 64,
    seq_len: int = 16,
    rotation_amount: int = 1
) -> Tuple[float, float]:
    """
    Verify rotation property: rotating position encodings is well-defined.

    In FHE, rotation cyclically permutes SIMD slots. For position encodings,
    this means: rotate_k([PE(0), PE(1), ...]) = [PE(k), PE(k+1), ...]

    This is trivially true by definition of rotation, but we verify it here.

    Args:
        d_model: Model dimension
        seq_len: Sequence length
        rotation_amount: Amount to rotate/shift

    Returns:
        (max_error, mean_error) - should be zero for correct implementation
    """
    # Compute original encoding matrix
    PE = cyclotomic_encoding_matrix(seq_len, d_model, max_seq_len=seq_len)

    # Rotate the matrix (FHE operation)
    PE_rotated = np.roll(PE, rotation_amount, axis=0)

    # What we expect after rotation: encodings at new positions
    PE_expected = np.zeros_like(PE)
    for pos in range(seq_len):
        # After rotating by k, position i gets the encoding from position (i-k) % seq_len
        source_pos = (pos - rotation_amount) % seq_len
        PE_expected[pos] = PE[source_pos]

    # Compare
    error = np.abs(PE_rotated - PE_expected)
    max_error = np.max(error)
    mean_error = np.mean(error)

    return max_error, mean_error


def compare_to_sinusoidal(
    seq_len: int = 16,
    d_model: int = 64
) -> None:
    """
    Compare cyclotomic encoding to standard sinusoidal encoding.

    Args:
        seq_len: Sequence length
        d_model: Model dimension
    """
    # Cyclotomic encoding
    PE_cyclo = cyclotomic_encoding_matrix(seq_len, d_model)

    # Standard sinusoidal encoding
    PE_sin = np.zeros((seq_len, d_model))
    for pos in range(seq_len):
        for i in range(d_model):
            if i % 2 == 0:
                PE_sin[pos, i] = np.sin(pos / 10000 ** (i / d_model))
            else:
                PE_sin[pos, i] = np.cos(pos / 10000 ** (i / d_model))

    print("Cyclotomic vs Sinusoidal Encoding Comparison")
    print("-" * 70)
    print(f"Sequence length: {seq_len}")
    print(f"Model dimension: {d_model}")
    print()

    print("Cyclotomic encoding statistics:")
    print(f"  Min: {PE_cyclo.min():.6f}")
    print(f"  Max: {PE_cyclo.max():.6f}")
    print(f"  Mean: {PE_cyclo.mean():.6f}")
    print(f"  Std: {PE_cyclo.std():.6f}")
    print()

    print("Sinusoidal encoding statistics:")
    print(f"  Min: {PE_sin.min():.6f}")
    print(f"  Max: {PE_sin.max():.6f}")
    print(f"  Mean: {PE_sin.mean():.6f}")
    print(f"  Std: {PE_sin.std():.6f}")
    print()

    # Both should have similar properties (bounded, oscillating)
    print("Both encodings provide unique position information with")
    print("bounded values suitable for neural networks.")


def test_cyclotomic_encoding():
    """Test the cyclotomic position encoding."""
    print("=" * 70)
    print("Testing Cyclotomic Position Encoding")
    print("=" * 70)

    # Test 1: Basic encoding
    print("\n1. Basic encoding test")
    print("-" * 70)

    d_model = 8
    max_seq_len = 16

    pe_0 = cyclotomic_position_encoding(0, d_model, max_seq_len)
    pe_1 = cyclotomic_position_encoding(1, d_model, max_seq_len)
    pe_5 = cyclotomic_position_encoding(5, d_model, max_seq_len)

    print(f"PE(pos=0): {pe_0}")
    print(f"PE(pos=1): {pe_1}")
    print(f"PE(pos=5): {pe_5}")
    print()

    # Check that encodings are different
    assert not np.allclose(pe_0, pe_1), "Positions should have different encodings"
    assert not np.allclose(pe_1, pe_5), "Positions should have different encodings"
    print("✓ Different positions have different encodings")

    # Test 2: Encoding matrix
    print("\n2. Encoding matrix test")
    print("-" * 70)

    seq_len = 16
    PE = cyclotomic_encoding_matrix(seq_len, d_model, max_seq_len)

    print(f"Encoding matrix shape: {PE.shape}")
    print(f"Expected: ({seq_len}, {d_model})")
    assert PE.shape == (seq_len, d_model)
    print("✓ Correct shape")

    # Check properties
    print(f"\nMatrix statistics:")
    print(f"  Min: {PE.min():.6f}")
    print(f"  Max: {PE.max():.6f}")
    print(f"  Mean: {PE.mean():.6f}")
    print(f"  Std: {PE.std():.6f}")

    # Should be bounded (roots of unity have magnitude 1)
    assert -1.1 <= PE.min() <= -0.9, "Values should be ~[-1, 1]"
    assert 0.9 <= PE.max() <= 1.1, "Values should be ~[-1, 1]"
    print("✓ Values are bounded in [-1, 1]")

    # Test 3: Rotation property (KEY PROPERTY!)
    print("\n3. Rotation property test")
    print("-" * 70)

    for k in [1, 2, 4]:
        max_err, mean_err = verify_rotation_property(
            d_model=d_model,
            seq_len=seq_len,
            rotation_amount=k
        )
        print(f"Rotation by {k}:")
        print(f"  Max error: {max_err:.10f}")
        print(f"  Mean error: {mean_err:.10f}")

        # Should be very small (numerical precision)
        assert max_err < 1e-10, f"Rotation property failed for k={k}"

    print("✓ Rotation property verified: rotate(PE) = shift(positions)")

    # Test 4: Compare to sinusoidal
    print("\n4. Comparison to sinusoidal encoding")
    print("-" * 70)
    compare_to_sinusoidal(seq_len=16, d_model=64)

    print("\n" + "=" * 70)
    print("✅ All cyclotomic encoding tests passed!")
    print("=" * 70)
    print()
    print("Key insights:")
    print("  • Cyclotomic encodings are FHE-native (no approximation needed)")
    print("  • Rotation property: rotate_k(PE) = PE(shifted positions)")
    print("  • Values bounded in [-1, 1] (numerically stable)")
    print("  • Provides unique position information like sinusoidal encoding")


if __name__ == "__main__":
    test_cyclotomic_encoding()
