"""
Computes and validates ALL Kronecker coefficients for S_4.

This script loops through all 125 possible coefficients, stores the non-zero
ones, and performs a comprehensive validation of the results.
"""

import numpy as np
import math
from itertools import product

from character_tables_known import get_s4_character_table

def compute_kronecker_coefficient(char_table, class_sizes, n, lambda_idx, mu_idx, nu_idx):
    """
    Computes the Kronecker coefficient g^ν_{λμ} for S_n.
    """
    n_factorial = math.factorial(n)
    chi_lambda = char_table[lambda_idx]
    chi_mu = char_table[mu_idx]
    chi_nu = char_table[nu_idx]
    kronecker_sum = np.sum(class_sizes * chi_lambda * chi_mu * np.conj(chi_nu))
    return round(kronecker_sum.real / n_factorial)

def main():
    """
    Main function to compute and validate all S4 Kronecker coefficients.
    """
    print("="*70)
    print("S_4 Full Kronecker Coefficient Computation and Validation")
    print("="*70)

    # 1. Get validated S4 data
    s4_char_table = get_s4_character_table()
    s4_class_sizes = np.array([1, 6, 3, 8, 6])
    n = 4
    num_irreps = s4_char_table.shape[0]

    partitions_list = ["4", "3,1", "2,2", "2,1,1", "1,1,1,1"]
    
    # 2. Compute all 125 Kronecker coefficients
    print("Computing all 125 S4 Kronecker coefficients...")
    kronecker_coeffs = np.zeros((num_irreps, num_irreps, num_irreps), dtype=int)
    non_zero_coeffs = []

    for lam_idx, mu_idx, nu_idx in product(range(num_irreps), repeat=3):
        g = compute_kronecker_coefficient(s4_char_table, s4_class_sizes, n, lam_idx, mu_idx, nu_idx)
        if g != 0:
            kronecker_coeffs[lam_idx, mu_idx, nu_idx] = g
            lam_name = partitions_list[lam_idx]
            mu_name = partitions_list[mu_idx]
            nu_name = partitions_list[nu_idx]
            non_zero_coeffs.append(f"g([{nu_name}])_([{lam_name}],[{mu_name}]) = {g}")

    print(f"Found {len(non_zero_coeffs)} non-zero coefficients.")
    # for coeff_str in sorted(non_zero_coeffs):
    #     print(f"  {coeff_str}")

    # 3. Perform Validation Checks
    print("\n" + "="*70)
    print("Validation Checks")
    print("="*70)

    # Validation 1: Symmetry (g^ν_{λμ} = g^ν_{μλ})
    symmetry_passed = True
    for lam_idx, mu_idx, nu_idx in product(range(num_irreps), repeat=3):
        if kronecker_coeffs[lam_idx, mu_idx, nu_idx] != kronecker_coeffs[mu_idx, lam_idx, nu_idx]:
            symmetry_passed = False
            print(f"  ✗ Symmetry failed for (λ,μ,ν) = ({lam_idx},{mu_idx},{nu_idx})")
            break
    if symmetry_passed:
        print("  ✓ Symmetry property holds for all coefficients.")

    # Validation 2: Unit Property ([4] ⊗ λ = λ)
    unit_passed = True
    unit_idx = partitions_list.index("4")
    for lam_idx in range(num_irreps):
        # Check that g^λ_{([4]),λ} = 1
        if kronecker_coeffs[unit_idx, lam_idx, lam_idx] != 1:
            unit_passed = False
            print(f"  ✗ Unit property failed: g^[{partitions_list[lam_idx]}]_([4],[{partitions_list[lam_idx]}]) != 1")
        # Check that all other coefficients are zero
        for nu_idx in range(num_irreps):
            if nu_idx != lam_idx and kronecker_coeffs[unit_idx, lam_idx, nu_idx] != 0:
                unit_passed = False
                print(f"  ✗ Unit property failed: g^[{partitions_list[nu_idx]}]_([4],[{partitions_list[lam_idx]}]) != 0")
    if unit_passed:
        print("  ✓ Unit property ([4] ⊗ λ = λ) holds for all λ.")

    # Validation 3: Dimension Formula (Σ_ν g^ν_{λμ} * d_ν = d_λ * d_μ)
    dims = s4_char_table[:, 0].real
    dim_formula_passed = True
    for lam_idx, mu_idx in product(range(num_irreps), repeat=2):
        actual_dim_sum = 0
        for nu_idx in range(num_irreps):
            g = kronecker_coeffs[lam_idx, mu_idx, nu_idx]
            actual_dim_sum += g * dims[nu_idx]
        
        expected_dim_sum = dims[lam_idx] * dims[mu_idx]
        
        if not math.isclose(actual_dim_sum, expected_dim_sum):
            dim_formula_passed = False
            lam_name = partitions_list[lam_idx]
            mu_name = partitions_list[mu_idx]
            print(f"  ✗ Dimension formula failed for λ=[{lam_name}], μ=[{mu_name}]")
            print(f"    Expected sum: {expected_dim_sum}, Actual sum: {actual_dim_sum}")

    if dim_formula_passed:
        print("  ✓ Dimension formula holds for all (λ,μ) pairs.")

    print("\n" + "="*70)
    if symmetry_passed and unit_passed and dim_formula_passed:
        print("✅ All S4 Kronecker coefficient validation checks passed!")
    else:
        print("✗ Some S4 validation checks failed.")
    print("="*70)


if __name__ == "__main__":
    main()
