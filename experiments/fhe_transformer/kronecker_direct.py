"""
Direct Kronecker Coefficient Computation

The formula:
    g^ν_{λμ} = ⟨χ_λ ⊗ χ_μ, χ_ν⟩
             = (1/|G|) Σ_g χ_λ(g) χ_μ(g) χ_ν(g)
             = (1/n!) Σ_{classes C} |C| χ_λ(C) χ_μ(C) χ_ν(C)

For S_3, this is trivial to compute once we have the character table.
"""

import numpy as np
from s3_character_table import S3CharacterTable


class KroneckerCoefficientComputer:
    """Compute Kronecker coefficients for S_n."""

    def __init__(self, char_table: S3CharacterTable):
        self.char_table = char_table
        self.n = 3  # For S_3

    def compute(self, lambda_idx: int, mu_idx: int, nu_idx: int) -> int:
        """
        Compute Kronecker coefficient g^ν_{λμ}.

        Args:
            lambda_idx, mu_idx, nu_idx: Indices of irreps in character table

        Returns:
            Multiplicity of ν in λ ⊗ μ
        """
        n_factorial = 6
        num_classes = len(self.char_table.class_sizes)

        total = 0
        for k in range(num_classes):
            class_size = self.char_table.class_sizes[k]
            chi_lambda = self.char_table.table[lambda_idx, k]
            chi_mu = self.char_table.table[mu_idx, k]
            chi_nu = self.char_table.table[nu_idx, k]

            total += class_size * chi_lambda * chi_mu * np.conj(chi_nu)

        coefficient = total / n_factorial

        return int(np.round(coefficient.real))

    def compute_all_for_s3(self):
        """Compute all Kronecker coefficients for S_3."""
        num_irreps = 3
        results = {}

        for lam in range(num_irreps):
            for mu in range(num_irreps):
                tensor_product_decomp = []

                for nu in range(num_irreps):
                    coeff = self.compute(lam, mu, nu)
                    if coeff > 0:
                        label = self.char_table.irrep_labels[nu]
                        tensor_product_decomp.append(f"{coeff if coeff > 1 else ''}{'·' if coeff > 1 else ''}{label}")

                lam_label = self.char_table.irrep_labels[lam]
                mu_label = self.char_table.irrep_labels[mu]
                decomp_str = " + ".join(tensor_product_decomp) if tensor_product_decomp else "0"

                results[(lam, mu)] = decomp_str

        return results


def test_kronecker_coefficients():
    """Test direct Kronecker coefficient computation."""
    print("=" * 70)
    print("Kronecker Coefficients for S_3 (Direct Formula)")
    print("=" * 70)

    s3_table = S3CharacterTable()
    kronecker = KroneckerCoefficientComputer(s3_table)

    print("\n✓ Character table loaded")
    print()

    # Compute all tensor products
    all_results = kronecker.compute_all_for_s3()

    print("Tensor Product Decompositions:")
    print("-" * 70)

    for (lam, mu), decomp in all_results.items():
        lam_label = s3_table.irrep_labels[lam]
        mu_label = s3_table.irrep_labels[mu]
        print(f"{lam_label:8s} ⊗ {mu_label:8s} = {decomp}")

    # Highlight the famous one: [2,1] ⊗ [2,1]
    print("\n" + "=" * 70)
    print("The Classic Test Case:")
    print("=" * 70)

    lam_idx = 1  # [2,1] standard rep
    mu_idx = 1

    print(f"\n{s3_table.irrep_labels[lam_idx]} ⊗ {s3_table.irrep_labels[mu_idx]} = ", end="")

    decomp = []
    for nu_idx in range(3):
        coeff = kronecker.compute(lam_idx, mu_idx, nu_idx)
        if coeff > 0:
            label = s3_table.irrep_labels[nu_idx]
            if coeff > 1:
                decomp.append(f"{coeff}·{label}")
            else:
                decomp.append(label)

    print(" + ".join(decomp))

    print("\n✅ All Kronecker coefficients computed via direct formula!")


if __name__ == "__main__":
    test_kronecker_coefficients()
