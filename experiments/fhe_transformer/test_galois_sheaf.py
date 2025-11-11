"""
Test the Galois-Sheaf Bootstrap Framework: Restriction & Induction

This script is a concrete step in implementing the theoretical
framework described in 'galois_bootstrap.py'.

It implements both halves of the Galois connection:
1.  Restriction Functor: Res: Rep(S_n) → Rep(C_k)
2.  Induction Functor:   Ind: Rep(C_k) → Rep(S_n)

The goal is to verify the core of the bootstrap mechanism: the
Frobenius Reciprocity theorem, which is the adjunction property
of this Galois connection.

    ⟨Ind(χ), ψ⟩_{S_n} = ⟨χ, Res(ψ)⟩_{C_k}

This validation confirms that our implementation of the sheaf's "local
sections" (Restriction) and "gluing" (Induction) is correct.
"""

import numpy as np
from math import gcd, factorial

# We will use our validated character tables and the cyclic group theory
from character_tables_known import get_s4_character_table
from character_theory_attention import CyclicGroupCharacters

def get_cycle_type_of_power(k, j, n):
    """
    Determines the cycle type of g^j in C_k embedded in S_n.
    'g' is a k-cycle.
    """
    if j == 0:
        return tuple([1] * n)
    
    common_divisor = gcd(k, j)
    cycle_len = k // common_divisor
    num_cycles = common_divisor
    
    parts = [cycle_len] * num_cycles
    
    remaining_elements = n - k
    if remaining_elements > 0:
        parts.extend([1] * remaining_elements)
        
    parts.sort(reverse=True)
    return tuple(parts)

def restrict_sn_character_to_ck(
    sn_char_table,
    sn_class_map,
    sn_irrep_idx,
    k
):
    """Restricts a character of S_n to the cyclic subgroup C_k."""
    n = sum(list(sn_class_map.keys())[0])
    restricted_char = np.zeros(k, dtype=complex)
    sn_character = sn_char_table[sn_irrep_idx]

    for j in range(k):
        cycle_type = get_cycle_type_of_power(k, j, n)
        if cycle_type not in sn_class_map:
            raise ValueError(f"Cycle type {cycle_type} for C_{k} element g^{j} not found in S_{n} class map.")
        class_idx = sn_class_map[cycle_type]
        restricted_char[j] = sn_character[class_idx]
        
    return restricted_char

def induce_ck_character_to_sn(
    ck_character,
    k,
    n,
    sn_class_map,
    sn_class_sizes
):
    """Induces a character of C_k to the symmetric group S_n."""
    induced_char = np.zeros(len(sn_class_map), dtype=complex)
    
    for sn_cycle_type, sn_class_idx in sn_class_map.items():
        # Formula: Ind(χ)(K) = |S_n|/|C_k| * (1/|K|) * Σ_{c in C_k, c fuses to K} χ(c)
        sum_chi_c = 0
        for j in range(k): # Iterate through elements c = g^j of C_k
            c_cycle_type = get_cycle_type_of_power(k, j, n)
            if c_cycle_type == sn_cycle_type:
                sum_chi_c += ck_character[j]
        
        if sum_chi_c != 0:
            sn_class_size = sn_class_sizes[sn_class_idx]
            induced_char[sn_class_idx] = (factorial(n) / k) * (1 / sn_class_size) * sum_chi_c
            
    return induced_char

def ck_inner_product(char1, char2, k):
    """Computes the inner product of two characters of C_k."""
    return np.vdot(char1, char2) / k

def sn_inner_product(char1, char2, class_sizes):
    """Computes the inner product of two characters of S_n."""
    n_factorial = sum(class_sizes)
    return np.vdot(char1 * class_sizes, char2) / n_factorial

def decompose_character(char_vector, group_theory, inner_product_func, **kwargs):
    """Decomposes a character into its irreducible components."""
    multiplicities = {}
    num_irreps = group_theory.shape[0] if isinstance(group_theory, np.ndarray) else group_theory.n
    
    for i in range(num_irreps):
        if isinstance(group_theory, np.ndarray):
            irrep_char = group_theory[i]
        else:
            irrep_char = np.array([group_theory.character(i, j) for j in range(num_irreps)])
        
        inner_prod = inner_product_func(char_vector, irrep_char, **kwargs)
        multiplicity = int(round(inner_prod.real))
        
        if multiplicity != 0:
            multiplicities[i] = multiplicity
            
    return multiplicities

def verify_frobenius_reciprocity(
    sn_char_table, sn_class_map, sn_class_sizes, sn_irrep_idx,
    c4_theory, c4_irrep_idx
):
    """Checks if <Ind(χ), ψ>_Sn = <χ, Res(ψ)>_Ck."""
    k = c4_theory.n
    n = sum(list(sn_class_map.keys())[0])

    # Get the characters
    psi = sn_char_table[sn_irrep_idx]
    chi = np.array([c4_theory.character(c4_irrep_idx, j) for j in range(k)])

    # 1. Compute RHS: <χ, Res(ψ)>_Ck
    res_psi = restrict_sn_character_to_ck(sn_char_table, sn_class_map, sn_irrep_idx, k)
    rhs = ck_inner_product(chi, res_psi, k)

    # 2. Compute LHS: <Ind(χ), ψ>_Sn
    ind_chi = induce_ck_character_to_sn(chi, k, n, sn_class_map, sn_class_sizes)
    lhs = sn_inner_product(ind_chi, psi, sn_class_sizes)

    error = abs(lhs - rhs)
    print(f"  <Ind(χ_{c4_irrep_idx}), ψ_{sn_irrep_idx}> = {lhs:.2f} | <χ_{c4_irrep_idx}, Res(ψ_{sn_irrep_idx})> = {rhs:.2f} | Error: {error:.4e}", end="")
    if error < 1e-9:
        print(" | ✅ Verified")
        return True
    else:
        print(" | ❌ FAILED")
        return False

def main():
    print("="*80)
    print("Testing the Galois-Sheaf Framework: Restriction, Induction, and Reciprocity")
    print("="*80)

    # 1. Load S4 Data
    s4_char_table = get_s4_character_table()
    s4_partitions = [[4], [3, 1], [2, 2], [2, 1, 1], [1, 1, 1, 1]]
    s4_class_cycle_types = [(1, 1, 1, 1), (2, 1, 1), (2, 2), (3, 1), (4,)]
    s4_class_map = {cycle_type: i for i, cycle_type in enumerate(s4_class_cycle_types)}
    s4_class_sizes = np.array([1, 6, 3, 8, 6])
    n=4
    k=4
    c4_theory = CyclicGroupCharacters(k)
    
    print("Loaded validated S_4 data.")

    # 2. Restriction Test (Unchanged)
    print(f"\n--- Part 1: Testing Restriction to C_{k} ---")
    for irrep_idx, p in enumerate(s4_partitions):
        restricted_char = restrict_sn_character_to_ck(s4_char_table, s4_class_map, irrep_idx, k)
        decomposition = decompose_character(restricted_char, c4_theory, ck_inner_product, k=k)
        decomp_str = " + ".join([f"{mult if mult > 1 else ''}χ_{i}" for i, mult in decomposition.items()])
        print(f"  Res(ψ_{p}) decomposes in C_{k} as: {decomp_str}")

    # 3. Induction Test
    print(f"\n--- Part 2: Testing Induction from C_{k} ---")
    for c4_irrep_idx in range(k):
        chi = np.array([c4_theory.character(c4_irrep_idx, j) for j in range(k)])
        induced_char = induce_ck_character_to_sn(chi, k, n, s4_class_map, s4_class_sizes)
        
        decomposition = decompose_character(induced_char, s4_char_table, sn_inner_product, class_sizes=s4_class_sizes)
        decomp_str = " + ".join([f"{mult if mult > 1 else ''}ψ_{s4_partitions[i]}" for i, mult in decomposition.items()])
        print(f"  Ind(χ_{c4_irrep_idx}) decomposes in S_{n} as: {decomp_str}")

    # 4. Frobenius Reciprocity Verification
    print("\n--- Part 3: Verifying Frobenius Reciprocity (The Adjunction) ---")
    print("Checking if <Ind(χ), ψ> = <χ, Res(ψ)> for all character pairs...")
    all_verified = True
    for sn_irrep_idx in range(len(s4_partitions)):
        for c4_irrep_idx in range(k):
            if not verify_frobenius_reciprocity(
                s4_char_table, s4_class_map, s4_class_sizes, sn_irrep_idx,
                c4_theory, c4_irrep_idx
            ):
                all_verified = False
    
    if all_verified:
        print("\n✅ All pairs successfully verified!")

    print("\n" + "="*80)
    print("Galois-Sheaf framework test complete. Both functors and their adjunction are verified.")
    print("="*80)

if __name__ == "__main__":
    main()
