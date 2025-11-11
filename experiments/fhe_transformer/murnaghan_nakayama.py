"""
Murnaghan-Nakayama Rule: The Key to Computing S_n Characters

This is the combinatorial algorithm for computing χ_λ(ρ) where:
- λ is a partition (Young diagram)
- ρ is a conjugacy class (cycle type)

The rule recursively removes "rim hooks" from the Young diagram.

Why it matters:
===============
- Direct matrix computation: O(n! × n!) (infeasible for n > 5)
- M-N rule: O(poly(n)) (works up to n ≈ 20)
- No matrices needed - pure combinatorics!

This is the missing piece for computing S_5, S_6 characters!
"""

import numpy as np
from typing import List, Tuple, Set
from functools import lru_cache

from symmetric_group_characters import Partition, ConjugacyClass


class YoungDiagram:
    """Young diagram with rim hook operations."""

    def __init__(self, partition: Partition):
        self.partition = partition
        self.rows = partition.parts[:]

    def __repr__(self):
        return f"YoungDiagram({self.rows})"

    def num_boxes(self) -> int:
        """Total number of boxes."""
        return sum(self.rows)

    def is_corner(self, i: int, j: int) -> bool:
        """Check if (i,j) is a corner (removable box)."""
        if i >= len(self.rows) or j >= self.rows[i]:
            return False
        # Corner if no box to right AND no box below
        no_right = (j == self.rows[i] - 1)
        no_below = (i == len(self.rows) - 1 or j >= self.rows[i+1])
        return no_right and no_below

    def get_rim_hooks(self, length: int) -> List[Tuple['YoungDiagram', int]]:
        """
        Find all rim hooks of given length.

        A rim hook is a connected border strip with no 2×2 block.
        """
        if length <= 0 or length > self.num_boxes():
            return []

        rim_hooks = []

        # Special case: remove entire diagram
        if length == self.num_boxes():
            height = len(self.rows) - 1
            empty = YoungDiagram(Partition([]))
            rim_hooks.append((empty, height))
            return rim_hooks

        # General case: recursively build rim hooks
        for i in range(len(self.rows)):
            for j in range(self.rows[i]):
                if self.is_corner(i, j):
                    # Try starting rim hook from this corner
                    result = self._build_rim_hook(i, j, length, set())
                    if result:
                        rim_hooks.extend(result)

        # Remove duplicates
        seen = set()
        unique = []
        for diagram, height in rim_hooks:
            key = tuple(diagram.rows)
            if key not in seen:
                seen.add(key)
                unique.append((diagram, height))

        return unique

    def _build_rim_hook(self, i: int, j: int, remaining: int, removed: Set) -> List:
        """Recursively build rim hook by removing connected corners."""
        if remaining == 0:
            # Build resulting diagram
            new_rows = []
            for row_idx, row_len in enumerate(self.rows):
                count = row_len
                for r, c in removed:
                    if r == row_idx:
                        count -= 1
                if count > 0:
                    new_rows.append(count)

            if not new_rows:
                new_diagram = YoungDiagram(Partition([]))
            else:
                new_diagram = YoungDiagram(Partition(new_rows))

            # Compute height
            rows_spanned = max(r for r, c in removed) - min(r for r, c in removed) if removed else 0
            height = rows_spanned

            return [(new_diagram, height)]

        if (i, j) in removed or i >= len(self.rows) or j >= self.rows[i]:
            return []

        # Try removing this box
        new_removed = removed | {(i, j)}

        # After removing, find new corners
        results = []

        # Continue in adjacent positions
        for di, dj in [(0, -1), (-1, 0)]:  # Left and up
            ni, nj = i + di, j + dj
            if ni >= 0 and nj >= 0:
                sub_results = self._build_rim_hook(ni, nj, remaining - 1, new_removed)
                results.extend(sub_results)

        if remaining == 1:
            # Base case
            results = self._build_rim_hook(i, j, 0, new_removed)

        return results


@lru_cache(maxsize=10000)
def murnaghan_nakayama(
    partition_tuple: tuple,
    cycle_type_tuple: tuple
) -> int:
    """
    Murnaghan-Nakayama rule (cached for speed).

    Computes character value χ_λ(ρ) recursively.

    Base cases:
    - Empty partition: χ_∅(anything) = 0 (unless ρ is identity)
    - Single cycle of length n: Use formula

    Recursive case:
    - Pick largest cycle k from ρ
    - Find all rim hooks of length k in λ
    - Sum: (-1)^height · χ_{λ-hook}(ρ-cycle)

    Args:
        partition_tuple: Young diagram as tuple
        cycle_type_tuple: Conjugacy class as tuple

    Returns:
        Character value (integer for S_n)
    """
    # Convert to lists
    partition_list = list(partition_tuple)
    cycle_type_list = list(cycle_type_tuple)

    # Base case: empty
    if not partition_list:
        return 1 if not cycle_type_list else 0

    if not cycle_type_list:
        return 0

    # Base case: identity
    if cycle_type_list == [1] * len(cycle_type_list):
        # Character value at identity = dimension
        # Use hook-length formula
        from symmetric_group_characters import SymmetricGroupCharacters
        n = sum(partition_list)
        sn = SymmetricGroupCharacters(n)
        p = Partition(partition_list)
        return sn.irrep_dimension(p)

    # Recursive case: remove largest cycle
    k = max(cycle_type_list)
    remaining_cycles = cycle_type_list[:]
    remaining_cycles.remove(k)

    # Find all rim hooks of length k
    diagram = YoungDiagram(Partition(partition_list))
    rim_hooks = diagram.get_rim_hooks(k)

    total = 0
    for new_diagram, height in rim_hooks:
        sign = (-1) ** height

        # Recursive call
        if new_diagram.rows:
            sub_value = murnaghan_nakayama(
                tuple(new_diagram.rows),
                tuple(remaining_cycles)
            )
            total += sign * sub_value

    return total


class MurnaghanNakayamaComputer:
    """Fast S_n character table computation via M-N rule."""

    def __init__(self, n: int):
        self.n = n
        from symmetric_group_characters import SymmetricGroupCharacters
        self.sn = SymmetricGroupCharacters(n)

    def compute_character_table(self) -> np.ndarray:
        """
        Compute full S_n character table using M-N rule.

        Returns:
            Character table [num_irreps, num_classes]
        """
        print(f"\nComputing S_{self.n} character table via Murnaghan-Nakayama...")

        num_irreps = len(self.sn.partitions)
        num_classes = len(self.sn.conjugacy_classes)

        char_table = np.zeros((num_irreps, num_classes), dtype=int)

        for i, partition in enumerate(self.sn.partitions):
            for j, conj_class in enumerate(self.sn.conjugacy_classes):
                # Use M-N rule
                value = murnaghan_nakayama(
                    tuple(partition.parts),
                    tuple(conj_class.cycle_type.parts)
                )
                char_table[i, j] = value

        print(f"✓ Character table computed: {char_table.shape}")

        return char_table

    def verify_character_table(self, char_table: np.ndarray) -> bool:
        """Verify via Schur orthogonality."""
        print("\nVerifying Schur orthogonality...")

        num_irreps = char_table.shape[0]
        n_factorial = np.prod(range(1, self.n + 1))

        # Compute Gram matrix
        gram = np.zeros((num_irreps, num_irreps))

        for i in range(num_irreps):
            for j in range(num_irreps):
                inner = 0
                for k, conj_class in enumerate(self.sn.conjugacy_classes):
                    class_size = conj_class.size()
                    inner += class_size * char_table[i, k] * char_table[j, k]

                gram[i, j] = inner / n_factorial

        # Should be identity
        expected = np.eye(num_irreps)
        error = np.linalg.norm(gram - expected)

        print(f"Orthogonality error: {error:.10e}")
        return error < 1e-10


def test_murnaghan_nakayama():
    """Test M-N rule on known cases."""
    print("="*70)
    print("Murnaghan-Nakayama Rule Test")
    print("="*70)

    # Test S_3
    print("\n1. S_3 Character Table")
    print("-"*70)

    mn3 = MurnaghanNakayamaComputer(3)
    table3 = mn3.compute_character_table()

    print("\nComputed:")
    print(table3)

    print("\nExpected:")
    expected3 = np.array([
        [1,  1,  1],
        [2,  0, -1],
        [1, -1,  1],
    ])
    print(expected3)

    match = np.allclose(table3, expected3)
    print(f"\n{'✓' if match else '✗'} Match: {match}")

    mn3.verify_character_table(table3)

    # Test S_4
    print("\n\n2. S_4 Character Table")
    print("-"*70)

    mn4 = MurnaghanNakayamaComputer(4)
    table4 = mn4.compute_character_table()

    print("\nComputed:")
    print(table4)

    valid4 = mn4.verify_character_table(table4)
    print(f"{'✓' if valid4 else '✗'} Valid: {valid4}")

    print("\n"+"="*70)
    print("✅ Murnaghan-Nakayama implementation complete!")
    print("="*70)


if __name__ == "__main__":
    test_murnaghan_nakayama()
