"""
Simulated FHE (Fully Homomorphic Encryption)

This module simulates FHE operations while tracking multiplicative depth.
It allows us to:
1. Verify correctness of FHE computations (compare to plaintext)
2. Measure depth cost without heavyweight FHE library
3. Fast iteration during development

Key concepts:
- Plaintext-ciphertext multiplication: depth unchanged (depth 0)
- Ciphertext-ciphertext multiplication: depth = depth1 + depth2 + 1
- Addition: depth = max(depth1, depth2)
- Rotation: depth unchanged (native FHE operation)

Usage:
    x_enc = SimulatedFHE(x, depth=0)  # Encrypt
    y_enc = x_enc + x_enc              # Depth = 0
    z_enc = x_enc * x_enc              # Depth = 1 (ct-ct mul)
    result = z_enc.decrypt()           # Decrypt
"""

import numpy as np
from typing import Union


class SimulatedFHE:
    """
    Simulates FHE operations with depth tracking.

    Attributes:
        value: The actual plaintext value (for verification)
        depth: Current multiplicative depth
        is_encrypted: Whether this is encrypted (vs plaintext constant)
    """

    def __init__(
        self,
        value: Union[np.ndarray, float, int],
        depth: int = 0,
        is_encrypted: bool = True
    ):
        """
        Initialize simulated ciphertext.

        Args:
            value: The plaintext value (for simulation only)
            depth: Initial depth (default 0 for fresh encryption)
            is_encrypted: True for ciphertext, False for plaintext constant
        """
        self.value = np.asarray(value)
        self.depth = depth
        self.is_encrypted = is_encrypted

    def __add__(self, other: 'SimulatedFHE') -> 'SimulatedFHE':
        """
        Homomorphic addition.

        Depth = max(depth1, depth2) because addition doesn't increase noise.
        """
        if not isinstance(other, SimulatedFHE):
            # Adding plaintext constant
            other = SimulatedFHE(other, depth=0, is_encrypted=False)

        new_value = self.value + other.value
        new_depth = max(self.depth, other.depth)

        return SimulatedFHE(new_value, new_depth, is_encrypted=True)

    def __radd__(self, other):
        """Right addition (for sum() compatibility)."""
        if other == 0:  # Handle sum([...]) case
            return self
        return self.__add__(other)

    def __sub__(self, other: 'SimulatedFHE') -> 'SimulatedFHE':
        """Homomorphic subtraction (same as addition)."""
        if not isinstance(other, SimulatedFHE):
            other = SimulatedFHE(other, depth=0, is_encrypted=False)

        new_value = self.value - other.value
        new_depth = max(self.depth, other.depth)

        return SimulatedFHE(new_value, new_depth, is_encrypted=True)

    def __mul__(self, other: Union['SimulatedFHE', float, int]) -> 'SimulatedFHE':
        """
        Homomorphic multiplication.

        Two cases:
        1. Plaintext-ciphertext: depth unchanged (efficient!)
        2. Ciphertext-ciphertext: depth = depth1 + depth2 + 1 (expensive!)
        """
        if not isinstance(other, SimulatedFHE):
            # Multiplying by plaintext constant (depth 0 operation!)
            other = SimulatedFHE(other, depth=0, is_encrypted=False)

        new_value = self.value * other.value

        if self.is_encrypted and other.is_encrypted:
            # Both ciphertext: expensive!
            new_depth = self.depth + other.depth + 1
        else:
            # At least one plaintext: free!
            new_depth = max(self.depth, other.depth)

        return SimulatedFHE(new_value, new_depth, is_encrypted=True)

    def __rmul__(self, other):
        """Right multiplication."""
        return self.__mul__(other)

    def __matmul__(self, other: Union['SimulatedFHE', np.ndarray]) -> 'SimulatedFHE':
        """
        Matrix multiplication.

        This is element-wise multiply + sum, so depth behavior depends on
        whether we're doing ct-ct or pt-ct multiplication.
        """
        if isinstance(other, np.ndarray):
            # Plaintext matrix
            other = SimulatedFHE(other, depth=0, is_encrypted=False)

        new_value = self.value @ other.value

        if self.is_encrypted and other.is_encrypted:
            # Both ciphertext: depth increases
            new_depth = self.depth + other.depth + 1
        else:
            # At least one plaintext: depth unchanged
            new_depth = max(self.depth, other.depth)

        return SimulatedFHE(new_value, new_depth, is_encrypted=True)

    def __rmatmul__(self, other):
        """Right matrix multiplication."""
        if isinstance(other, np.ndarray):
            other = SimulatedFHE(other, depth=0, is_encrypted=False)
        return other.__matmul__(self)

    def rotate(self, k: int) -> 'SimulatedFHE':
        """
        FHE rotation (cyclically shift SIMD slots).

        In real FHE: This is a native operation (automorphism x → x^k)
        Depth: Unchanged! Rotations are "free" in terms of depth.

        Args:
            k: Rotation amount (can be negative)

        Returns:
            Rotated ciphertext with same depth
        """
        # Rotate along first axis (sequence dimension)
        new_value = np.roll(self.value, k, axis=0)

        # Rotation doesn't increase depth!
        return SimulatedFHE(new_value, self.depth, is_encrypted=True)

    def decrypt(self) -> np.ndarray:
        """
        Decrypt ciphertext to plaintext.

        In simulation, this just returns the underlying value.
        """
        return self.value

    def get_depth(self) -> int:
        """Get current multiplicative depth."""
        return self.depth

    def __repr__(self) -> str:
        """String representation for debugging."""
        shape_str = str(self.value.shape) if hasattr(self.value, 'shape') else 'scalar'
        enc_str = 'encrypted' if self.is_encrypted else 'plaintext'
        return f"SimulatedFHE(shape={shape_str}, depth={self.depth}, {enc_str})"


def encrypt(value: Union[np.ndarray, float, int], depth: int = 0) -> SimulatedFHE:
    """
    Convenience function to encrypt a value.

    Args:
        value: Plaintext to encrypt
        depth: Initial depth (default 0)

    Returns:
        Simulated ciphertext
    """
    return SimulatedFHE(value, depth=depth, is_encrypted=True)


def plaintext(value: Union[np.ndarray, float, int]) -> SimulatedFHE:
    """
    Create plaintext constant (for pt-ct multiplication).

    Args:
        value: Plaintext constant

    Returns:
        Simulated plaintext with depth 0, is_encrypted=False
    """
    return SimulatedFHE(value, depth=0, is_encrypted=False)


def test_simulated_fhe():
    """Test the simulated FHE implementation."""
    print("=" * 70)
    print("Testing Simulated FHE")
    print("=" * 70)

    # Test basic operations
    print("\n1. Basic operations")
    print("-" * 70)

    x = encrypt(np.array([1.0, 2.0, 3.0]))
    y = encrypt(np.array([4.0, 5.0, 6.0]))

    print(f"x = {x}")
    print(f"y = {y}")

    # Addition
    z_add = x + y
    print(f"\nx + y = {z_add}")
    print(f"  Value: {z_add.decrypt()}")
    print(f"  Depth: {z_add.get_depth()} (expected: 0)")
    assert z_add.get_depth() == 0, "Addition should not increase depth"

    # Plaintext-ciphertext multiplication
    z_pt_mul = x * 2.0
    print(f"\nx * 2.0 (plaintext) = {z_pt_mul}")
    print(f"  Value: {z_pt_mul.decrypt()}")
    print(f"  Depth: {z_pt_mul.get_depth()} (expected: 0)")
    assert z_pt_mul.get_depth() == 0, "Plaintext mul should not increase depth"

    # Ciphertext-ciphertext multiplication
    z_ct_mul = x * y
    print(f"\nx * y (ciphertext) = {z_ct_mul}")
    print(f"  Value: {z_ct_mul.decrypt()}")
    print(f"  Depth: {z_ct_mul.get_depth()} (expected: 1)")
    assert z_ct_mul.get_depth() == 1, "Ciphertext mul should increase depth to 1"

    # Test rotation
    print("\n2. Rotation operations")
    print("-" * 70)

    x = encrypt(np.array([1.0, 2.0, 3.0, 4.0]))
    print(f"x = {x.decrypt()}")

    x_rot1 = x.rotate(1)
    print(f"rotate(x, 1) = {x_rot1.decrypt()}")
    print(f"  Depth: {x_rot1.get_depth()} (expected: 0)")
    assert x_rot1.get_depth() == 0, "Rotation should not increase depth"
    assert np.allclose(x_rot1.decrypt(), [4.0, 1.0, 2.0, 3.0])

    x_rot_neg = x.rotate(-1)
    print(f"rotate(x, -1) = {x_rot_neg.decrypt()}")
    assert np.allclose(x_rot_neg.decrypt(), [2.0, 3.0, 4.0, 1.0])

    # Test matrix multiplication
    print("\n3. Matrix multiplication")
    print("-" * 70)

    x = encrypt(np.array([[1.0, 2.0], [3.0, 4.0]]))
    W = np.array([[1.0, 0.0], [0.0, 1.0]])  # Identity (plaintext)

    y = x @ W
    print(f"x @ W (plaintext matrix) = {y}")
    print(f"  Value:\n{y.decrypt()}")
    print(f"  Depth: {y.get_depth()} (expected: 0)")
    assert y.get_depth() == 0, "Plaintext matrix mul should not increase depth"

    # Test depth composition
    print("\n4. Depth composition")
    print("-" * 70)

    x = encrypt(5.0)

    # Square: depth 1
    x2 = x * x
    print(f"x^2: depth = {x2.get_depth()} (expected: 1)")
    assert x2.get_depth() == 1

    # Cube: depth 2 (via x^2 * x)
    x3 = x2 * x
    print(f"x^3 = x^2 * x: depth = {x3.get_depth()} (expected: 2)")
    assert x3.get_depth() == 2

    # Chebyshev T_3(x) = 4x^3 - 3x
    T3 = 4.0 * x3 - 3.0 * x
    print(f"T_3(x) = 4x^3 - 3x: depth = {T3.get_depth()} (expected: 2)")
    assert T3.get_depth() == 2
    print(f"  Value: {T3.decrypt()} (expected: {4*5**3 - 3*5})")
    assert np.isclose(T3.decrypt(), 4*5**3 - 3*5)

    print("\n" + "=" * 70)
    print("✅ All simulated FHE tests passed!")
    print("=" * 70)


if __name__ == "__main__":
    test_simulated_fhe()
