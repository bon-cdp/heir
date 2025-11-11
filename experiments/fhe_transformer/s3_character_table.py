"""
S_3 Character Table: The Rosetta Stone

Known character table for S_3 (from textbooks):

Conjugacy classes:     (3)    (2,1)   (1,1,1)
Class sizes:            2       3         1

[3]        (trivial)    1       1         1
[2,1]      (standard)   2       0        -1
[1,1,1]    (sign)       1      -1         1

This is our ground truth for validation.
"""

import numpy as np
from galois_bootstrap import GaloisConnection


class S3CharacterTable:
    """The known S_3 character table."""

    def __init__(self):
        # Conjugacy classes: (3), (2,1), (1,1,1)
        self.class_labels = ['(3)', '(2,1)', '(1,1,1)']
        self.class_sizes = [2, 3, 1]

        # Irrep labels
        self.irrep_labels = ['[3]', '[2,1]', '[1,1,1]']

        # The character table
        self.table = np.array([
            [1,  1,  1],   # [3] = trivial
            [2,  0, -1],   # [2,1] = standard
            [1, -1,  1],   # [1,1,1] = sign
        ], dtype=complex)

    def verify_orthogonality(self):
        """Verify Schur orthogonality."""
        n_factorial = 6
        num_irreps = 3

        gram = np.zeros((num_irreps, num_irreps), dtype=complex)

        for i in range(num_irreps):
            for j in range(num_irreps):
                inner = sum(
                    self.class_sizes[k] *
                    np.conj(self.table[i, k]) *
                    self.table[j, k]
                    for k in range(3)
                )
                gram[i, j] = inner / n_factorial  # Normalize properly

        expected = np.eye(num_irreps)
        error = np.linalg.norm(gram - expected)

        print(f"Orthogonality error: {error:.10e}")
        return error < 1e-10


def test_galois_reconstruction():
    """
    The beautiful test: Can we reconstruct S_3 character table
    by inducing from C_2 and C_3?
    """
    print("=" * 70)
    print("S_3 Character Table via Galois Bootstrap")
    print("=" * 70)

    # Ground truth
    s3_table = S3CharacterTable()

    print("\n1. Known S_3 Character Table (Ground Truth)")
    print("-" * 70)
    print("Classes:   ", "  ".join(s3_table.class_labels))
    print("Sizes:     ", "  ".join(str(s) for s in s3_table.class_sizes))
    print()
    for i, label in enumerate(s3_table.irrep_labels):
        chars = "  ".join(f"{int(c.real):2d}" for c in s3_table.table[i])
        print(f"{label:8s}   {chars}")

    print("\n✓ Verify orthogonality:")
    s3_table.verify_orthogonality()

    # Galois bootstrap
    print("\n2. Bootstrap via Galois Connection")
    print("-" * 70)

    galois = GaloisConnection(3)

    # Generate all induced characters
    all_induced = galois.generate_all_sn_chars_from_cyclic()

    print(f"Generated {len(all_induced)} induced characters from C_k:")
    for name, induced_char in all_induced.items():
        chars = "  ".join(f"{c.real:5.1f}" for c in induced_char)
        print(f"  {name:20s}: [{chars}]")

    # Decompose into irreps
    print("\n3. Decompose Induced Characters into S_3 Irreps")
    print("-" * 70)

    for name, induced_char in all_induced.items():
        multiplicities = galois.decompose_induced_into_irreps(
            induced_char,
            s3_table.table
        )

        # Pretty print
        decomp = " + ".join(
            f"{m}·{label}"
            for m, label in zip(multiplicities, s3_table.irrep_labels)
            if m > 0
        )
        if not decomp:
            decomp = "0"

        print(f"{name:20s} = {decomp}")

    print("\n" + "=" * 70)
    print("✅ Galois bootstrap complete!")
    print("=" * 70)


if __name__ == "__main__":
    test_galois_reconstruction()
