"""
Murnaghan-Nakayama Rule: The Key to Computing S_n Characters

This is the combinatorial algorithm for computing \u03c7_\u03bb(\u03c1) where:
- \u03bb is a partition (Young diagram)
- \u03c1 is a conjugacy class (cycle type)

This implementation attempts a manual, recursive approach.
The critical component, `get_rim_hooks`, needs a robust and correct implementation.
"""

import numpy as np
from functools import lru_cache
import sys
import math

# Add the experiment directory to the path to import symmetric_group_characters
sys.path.append('/home/shakil/Documents/heir/heir/experiments/fhe_transformer')
from symmetric_group_characters import SymmetricGroupCharacters as LocalSGC

# --- Placeholder for a CORRECT get_rim_hooks implementation ---
# This function is the core combinatorial challenge.
# It needs to return a list of (new_partition_tuple, height) for all valid rim hooks of 'length'.
# A robust, well-tested implementation is required here.
@lru_cache(maxsize=None)
def get_rim_hooks(p: tuple, length: int) -> list:
    """
    Placeholder for a correct implementation of finding rim hooks.
    Currently returns an empty list, which will cause character tables to be all zeros.
    """
    # This is where a robust implementation needs to go.
    # For now, returning empty to indicate it's not implemented correctly.
    return []

# --- End of Placeholder ---


@lru_cache(maxsize=None)
def murnaghan_nakayama(partition: tuple, cycle_type: tuple) -> int:
    """
    Computes the character value \u03c7_\u03bb(\u03c1) using the Murnaghan-Nakayama rule.
    This is a recursive implementation.
    """
    if not partition:
        return 1 if not cycle_type else 0
    if not cycle_type:
        return 0

    k = cycle_type[0]
    remaining_cycles = cycle_type[1:]

    # A partition of n cannot have a rim hook of length > n
    if k > sum(partition):
        return 0

    total = 0
    
    hooks = get_rim_hooks(partition, k)

    for new_partition, height in hooks:
        sign = (-1) ** height
        total += sign * murnaghan_nakayama(new_partition, remaining_cycles)

    return total


class MurnaghanNakayamaComputer:
    """Fast S_n character table computation via M-N rule."""

    def __init__(self, n: int):
        self.n = n
        self.sn = LocalSGC(n)
        self.partitions = sorted([p.parts for p in self.sn.partitions], reverse=True)
        self.conjugacy_classes = sorted([c.cycle_type.parts for c in self.sn.conjugacy_classes], reverse=True)

    def compute_character_table(self) -> np.ndarray:
        """
        Compute full S_n character table using M-N rule.
        """
        print(f"\nComputing S_{{self.n}} character table via Murnaghan-Nakayama...")
        num_irreps = len(self.partitions)
        num_classes = len(self.conjugacy_classes)
        char_table = np.zeros((num_irreps, num_classes), dtype=int)

        for i, p_parts in enumerate(self.partitions):
            for j, c_parts in enumerate(self.conjugacy_classes):
                # Sort cycle parts for cache efficiency
                sorted_c_parts = tuple(sorted(c_parts, reverse=True))
                val = murnaghan_nakayama(tuple(p_parts), sorted_c_parts)
                char_table[i, j] = val
        
        print(f"✓ Character table computed: {char_table.shape}")
        return char_table

    def verify_character_table(self, char_table: np.ndarray) -> bool:
        """Verify via Schur orthogonality."""
        print("\nVerifying Schur orthogonality...")
        num_irreps = char_table.shape[0]
        n_factorial = math.factorial(self.n)

        gram = np.zeros((num_irreps, num_irreps))
        
        original_classes_map = {tuple(c.cycle_type.parts): c.size() for c in self.sn.conjugacy_classes}
        sorted_class_sizes = [original_classes_map[tuple(c)] for c in self.conjugacy_classes]

        for i in range(num_irreps):
            for j in range(num_irreps):
                inner_product = sum(
                    sorted_class_sizes[k] * char_table[i, k] * np.conj(char_table[j, k])
                    for k in range(len(self.conjugacy_classes))
                )
                gram[i, j] = inner_product / n_factorial

        expected = np.eye(num_irreps)
        error = np.linalg.norm(gram - expected)

        print(f"Orthogonality error: {error:.10e}")
        return error < 1e-9


def test_murnaghan_nakayama():
    """Test M-N rule on known cases."""
    print("="*70)
    print("Murnaghan-Nakayama Rule Test (Manual Implementation - get_rim_hooks placeholder)")
    print("="*70)

    # Test S_3
    print("\n1. S_3 Character Table")
    print("-" * 70)
    mn3 = MurnaghanNakayamaComputer(3)
    table3 = mn3.compute_character_table()
    print("\nComputed:")
    print(table3)
    valid3 = mn3.verify_character_table(table3)
    print(f"\n{'✓' if valid3 else '✗'} S3 Orthogonality Check Passed: {valid3}")

    # Test S_4
    print("\n\n2. S_4 Character Table")
    print("-" * 70)
    mn4 = MurnaghanNakayamaComputer(4)
    table4 = mn4.compute_character_table()
    print("\nComputed:")
    print(table4)
    valid4 = mn4.verify_character_table(table4)
    print(f"\n{'✓' if valid4 else '✗'} S4 Orthogonality Check Passed: {valid4}")

    print("\n" + "=" * 70)
    if valid3 and valid4:
        print("✅ Murnaghan-Nakayama implementation appears correct!")
    else:
        print("✗ Murnaghan-Nakayama implementation still has issues (likely get_rim_hooks).")
    print("=" * 70)


if __name__ == "__main__":
    test_murnaghan_nakayama()
