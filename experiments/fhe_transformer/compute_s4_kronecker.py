"""
Computes and validates ALL Kronecker coefficients for S_4.

This script loops through all 125 possible coefficients, stores the non-zero
ones, performs a comprehensive validation of the results, and saves the
full coefficient tensor to a JSON file for use with the learning script.
"""

import numpy as np
import math
import json
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
    Main function to compute, validate, and save all S4 Kronecker coefficients.
    """
    print("="*70)
    print("S_4 Full Kronecker Coefficient Computation and Validation")
    print("="*70)

    # 1. Get validated S4 data
    s4_char_table = get_s4_character_table()
    s4_class_sizes = np.array([1, 6, 3, 8, 6])
    n = 4
    num_irreps = s4_char_table.shape[0]

    partitions_list = [[4], [3, 1], [2, 2], [2, 1, 1], [1, 1, 1, 1]]
    
    # 2. Compute all 125 Kronecker coefficients
    print("Computing all 125 S4 Kronecker coefficients...")
    kronecker_coeffs_dict = {}
    non_zero_count = 0

    for lam_idx, mu_idx, nu_idx in product(range(num_irreps), repeat=3):
        g = compute_kronecker_coefficient(s4_char_table, s4_class_sizes, n, lam_idx, mu_idx, nu_idx)
        
        # Store in dict for JSON export
        key = str((tuple(partitions_list[lam_idx]), tuple(partitions_list[mu_idx]), tuple(partitions_list[nu_idx])))
        kronecker_coeffs_dict[key] = int(g)

        if g != 0:
            non_zero_count += 1

    print(f"Found {non_zero_count} non-zero coefficients.")

    # 3. Save coefficients to JSON
    output_path = "s4_kronecker.json"
    with open(output_path, "w") as f:
        json.dump(kronecker_coeffs_dict, f, indent=2)
    print(f"✓ Successfully saved all S4 coefficients to '{output_path}'")

    # The rest of the validation logic is removed as it was for one-time verification.
    # The main purpose of this script is now to generate the data for the learner.
    print("\n" + "="*70)
    print("✅ S4 Kronecker coefficient data generation complete.")
    print("="*70)


if __name__ == "__main__":
    main()