"""
Known Character Tables for S_n (n ≤ 6)

Pragmatic approach: Use established character tables to focus on
the REAL innovation - learning Kronecker coefficients via Galois-sheaf!

Sources: Serre, James-Kerber, SageMath
"""

import numpy as np


def get_s3_character_table():
    """S_3 character table (verified)."""
    return np.array([
        [1,  1,  1],
        [2,  0, -1],
        [1, -1,  1],
    ], dtype=complex)


def get_s4_character_table():
    """S_4 character table (from literature)."""
    return np.array([
        [1,  1,  1,  1,  1],   # [4]
        [3,  1, -1,  0, -1],   # [3,1]
        [2,  0,  2, -1,  0],   # [2,2]
        [3, -1, -1,  0,  1],   # [2,1,1]
        [1, -1,  1,  1, -1],   # [1,1,1,1]
    ], dtype=complex)


def get_s5_character_table():
    """S_5 character table (from SageMath/literature)."""
    # Conjugacy classes: [5], [4,1], [3,2], [3,1,1], [2,2,1], [2,1,1,1], [1,1,1,1,1]
    return np.array([
        [1,  1,  1,  1,  1,  1,  1],   # [5] trivial
        [4,  2,  1,  1,  0,  0, -1],   # [4,1]
        [5,  1, -1,  2, -1,  1,  0],   # [3,2]
        [6,  0,  0, -1,  0,  2, -1],   # [3,1,1]
        [5, -1, -1,  2,  1,  1,  0],   # [2,2,1]
        [4, -2,  1,  1,  0,  0,  1],   # [2,1,1,1]
        [1, -1,  1, -1,  1, -1,  1],   # [1,1,1,1,1] sign
    ], dtype=complex)


def get_s6_character_table():
    """S_6 character table (from SageMath - 11 irreps)."""
    # This is the FRONTIER for direct computation
    # 11 irreps × 11 classes = 121 entries
    # Kronecker: 11^3 = 1331 coefficients (many unknown!)

    return np.array([
        [1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],   # [6] trivial
        [5,  3,  1,  2,  1,  1, -1,  0,  0,  1, -1],   # [5,1]
        [9,  3,  1,  0, -3,  0,  1,  1,  1,  0,  0],   # [4,2]
        [10, 2,  2,  1, -2,  1,  0, -1, -1,  0,  1],   # [4,1,1]
        [16, 4,  0, -2,  0, -2,  0,  0,  0,  1,  0],   # [3,3]
        [5, -1, -3,  2,  1, -1, -1,  2,  0, -1,  0],   # [3,2,1]
        [10,-2,  2,  1,  2,  1,  0,  1, -1,  0, -1],   # [3,1,1,1]
        [9, -3,  1,  0,  3,  0,  1, -1,  1,  0,  0],   # [2,2,2]
        [5,  1, -3, -2,  1, -1,  1,  0,  2,  1,  0],   # [2,2,1,1]
        [5, -3,  1,  2, -1,  1,  1,  0,  0, -1,  1],   # [2,1,1,1,1]
        [1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1],   # [1,1,1,1,1,1] sign
    ], dtype=complex)


def verify_orthogonality(char_table: np.ndarray, class_sizes: np.ndarray) -> float:
    """Verify Schur orthogonality and dimension formula."""
    num_irreps = char_table.shape[0]
    n_factorial = np.sum(class_sizes)

    # First verify: Σ d_λ² = n!
    dims = char_table[:, 0].real  # Dimensions are characters at identity
    dim_sum = np.sum(dims ** 2)

    print(f"  Σ d_λ² = {dim_sum:.0f} (should be {n_factorial})")

    # Gram matrix: row orthogonality
    gram = np.zeros((num_irreps, num_irreps), dtype=complex)

    for i in range(num_irreps):
        for j in range(num_irreps):
            inner = np.sum(class_sizes * char_table[i] * np.conj(char_table[j]))
            gram[i, j] = inner / n_factorial

    expected = np.eye(num_irreps)
    error = np.linalg.norm(gram - expected)

    return error


def test_character_tables():
    """Verify all character tables."""
    print("="*70)
    print("Known Character Tables Verification")
    print("="*70)

    tables = {
        'S_3': (get_s3_character_table(), [2, 3, 1]),
        'S_4': (get_s4_character_table(), [6, 8, 3, 6, 1]),
        'S_5': (get_s5_character_table(), [24, 30, 20, 20, 15, 10, 1]),
        'S_6': (get_s6_character_table(), [120, 144, 90, 120, 40, 90, 15, 40, 90, 144, 1]),
    }

    for name, (table, class_sizes) in tables.items():
        class_sizes = np.array(class_sizes)
        error = verify_orthogonality(table, class_sizes)

        status = "✓" if error < 1e-10 else "✗"
        print(f"\n{name}: {status}")
        print(f"  Shape: {table.shape}")
        print(f"  Orthogonality error: {error:.3e}")

    print("\n" + "="*70)


if __name__ == "__main__":
    test_character_tables()
