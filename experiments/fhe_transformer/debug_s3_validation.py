"""
S_3 Validation: Hand-Check Everything

S_3 is simple enough to verify by hand. This is our ground truth test.

S_3 structure:
- 3 irreps: [3], [2,1], [1,1,1]
- 3 conjugacy classes (cycle types): [3], [2,1], [1,1,1]
- 3! = 6 elements

Character table (from Serre):
         [3]  [2,1]  [1,1,1]
[3]       1     1      1      (trivial)
[2,1]     2     0     -1      (standard)
[1,1,1]   1    -1      1      (sign)

Class sizes:
- [3] (identity): 1 element
- [2,1] (transpositions): 3 elements
- [1,1,1] (3-cycle): 2 elements
Total: 1+3+2 = 6 ✓
"""

import numpy as np
from symmetric_group_characters import Partition, ConjugacyClass
from character_tables_known import get_s3_character_table


def verify_s3_structure():
    """Verify basic S_3 structure."""
    print("=" * 70)
    print("S_3 Structure Verification")
    print("=" * 70)

    # Partitions
    partitions = Partition.generate_partitions(3)
    print(f"\nPartitions of 3: {[p.parts for p in partitions]}")
    print(f"Expected: [[3], [2,1], [1,1,1]]")

    # Conjugacy classes
    conj_classes = ConjugacyClass.all_conjugacy_classes(3)
    print(f"\nConjugacy classes:")
    for cc in conj_classes:
        print(f"  Cycle type {cc.cycle_type.parts}: size = {cc.size()}")

    # Expected class sizes
    expected_sizes = {
        (3,): 2,           # Two 3-cycles: (123) and (132)
        (2, 1): 3,         # Three transpositions: (12), (13), (23)
        (1, 1, 1): 1       # Identity
    }

    print(f"\nClass size verification:")
    all_correct = True
    for cc in conj_classes:
        cycle = tuple(cc.cycle_type.parts)
        computed = cc.size()
        expected = expected_sizes[cycle]
        status = "✓" if computed == expected else "✗"
        print(f"  {cycle}: computed={computed}, expected={expected} {status}")
        if computed != expected:
            all_correct = False

    # Total elements
    total = sum(cc.size() for cc in conj_classes)
    print(f"\nTotal elements: {total} (should be 6) {'✓' if total == 6 else '✗'}")

    return all_correct and total == 6


def verify_s3_character_table():
    """Verify S_3 character table."""
    print("\n" + "=" * 70)
    print("S_3 Character Table Verification")
    print("=" * 70)

    char_table = get_s3_character_table()
    print(f"\nCharacter table shape: {char_table.shape}")
    print(f"\nCharacter table:")
    print(char_table)

    # Expected (from Serre)
    expected = np.array([
        [1,  1,  1],    # [3] trivial
        [2,  0, -1],    # [2,1] standard
        [1, -1,  1],    # [1,1,1] sign
    ])

    print(f"\nExpected (from Serre):")
    print(expected)

    matches = np.allclose(char_table, expected)
    print(f"\nMatches expected: {'✓' if matches else '✗'}")

    # Dimensions (first column)
    dims = char_table[:, 0].real
    print(f"\nDimensions: {dims}")
    dim_sum = np.sum(dims ** 2)
    print(f"Σ d_λ² = {dim_sum} (should be 6) {'✓' if dim_sum == 6 else '✗'}")

    # Now check: WHICH column corresponds to which conjugacy class?
    partitions = Partition.generate_partitions(3)
    conj_classes = ConjugacyClass.all_conjugacy_classes(3)

    print(f"\nConjugacy class ordering:")
    print(f"  Partitions (our code): {[p.parts for p in partitions]}")
    print(f"  Conj classes (cycle types): {[cc.cycle_type.parts for cc in conj_classes]}")

    # They should be the SAME for S_3
    class_order_matches = all(
        partitions[i].parts == conj_classes[i].cycle_type.parts
        for i in range(3)
    )
    print(f"  Order matches: {'✓' if class_order_matches else '✗'}")

    return matches and dim_sum == 6 and class_order_matches


def verify_s3_orthogonality():
    """Verify Schur orthogonality relations."""
    print("\n" + "=" * 70)
    print("S_3 Schur Orthogonality")
    print("=" * 70)

    char_table = get_s3_character_table()

    # Class sizes (must match column order!)
    # Columns: [3], [2,1], [1,1,1] → sizes: [2, 3, 1]
    # Wait - identity should be column 0!

    # Let me check what the actual column order is
    # If column 0 is identity, all values should be dimensions
    col0 = char_table[:, 0]
    print(f"\nColumn 0: {col0} (should be dimensions)")

    # If this is identity, class size is 1
    # Let's try different orderings

    # Possibility 1: columns are [[1,1,1], [2,1], [3]]
    class_sizes_attempt1 = np.array([1, 3, 2])
    print(f"\nAttempt 1 - Class sizes: {class_sizes_attempt1}")
    print(f"  Order: [[1,1,1], [2,1], [3]]")
    gram1 = compute_gram_matrix(char_table, class_sizes_attempt1)
    error1 = np.linalg.norm(gram1 - np.eye(3))
    print(f"  Orthogonality error: {error1:.6f}")

    # Possibility 2: columns are [[3], [2,1], [1,1,1]]
    class_sizes_attempt2 = np.array([2, 3, 1])
    print(f"\nAttempt 2 - Class sizes: {class_sizes_attempt2}")
    print(f"  Order: [[3], [2,1], [1,1,1]]")
    gram2 = compute_gram_matrix(char_table, class_sizes_attempt2)
    error2 = np.linalg.norm(gram2 - np.eye(3))
    print(f"  Orthogonality error: {error2:.6f}")

    # Check which is correct
    if error1 < 1e-10:
        print(f"\n✓ Column order is: [[1,1,1], [2,1], [3]]")
        correct_sizes = class_sizes_attempt1
        correct_order = "[[1,1,1], [2,1], [3]]"
    elif error2 < 1e-10:
        print(f"\n✓ Column order is: [[3], [2,1], [1,1,1]]")
        correct_sizes = class_sizes_attempt2
        correct_order = "[[3], [2,1], [1,1,1]]"
    else:
        print(f"\n✗ Neither ordering works!")
        correct_sizes = None
        correct_order = None

    return correct_sizes, correct_order


def compute_gram_matrix(char_table, class_sizes):
    """Compute Gram matrix for orthogonality check."""
    n_irreps = char_table.shape[0]
    gram = np.zeros((n_irreps, n_irreps))

    for i in range(n_irreps):
        for j in range(n_irreps):
            inner = np.sum(class_sizes * char_table[i] * np.conj(char_table[j]))
            gram[i, j] = (inner / 6).real  # Normalize by |G|, take real part

    return gram


def compute_s3_kronecker_by_hand():
    """Compute S_3 Kronecker coefficients by hand to verify."""
    print("\n" + "=" * 70)
    print("S_3 Kronecker Coefficients (Hand Computation)")
    print("=" * 70)

    char_table = get_s3_character_table()

    # First determine correct class sizes
    correct_sizes, correct_order = verify_s3_orthogonality()

    if correct_sizes is None:
        print("\n✗ Cannot compute - orthogonality failed!")
        return None

    print(f"\nUsing class sizes: {correct_sizes}")
    print(f"Column order: {correct_order}")

    # Compute ALL 27 coefficients
    partitions = [[3], [2, 1], [1, 1, 1]]
    print(f"\nComputing all Kronecker coefficients:")

    results = {}
    for lam_idx in range(3):
        for mu_idx in range(3):
            for nu_idx in range(3):
                # Direct formula
                chi_lam = char_table[lam_idx]
                chi_mu = char_table[mu_idx]
                chi_nu = char_table[nu_idx]

                total = np.sum(
                    correct_sizes * chi_lam * chi_mu * np.conj(chi_nu)
                )
                coeff = int(np.round((total / 6).real))

                lam = partitions[lam_idx]
                mu = partitions[mu_idx]
                nu = partitions[nu_idx]

                if coeff > 0:
                    print(f"  g^{nu}_{{{lam},{mu}}} = {coeff}")
                    results[(tuple(lam), tuple(mu), tuple(nu))] = coeff

    return results


def validate_s3_properties(coefficients):
    """Validate S_3 Kronecker coefficients."""
    print("\n" + "=" * 70)
    print("S_3 Validation Tests")
    print("=" * 70)

    if coefficients is None:
        print("No coefficients to validate!")
        return False

    # Test 1: Unit property
    print("\n1. Unit property: [3] ⊗ λ = λ")
    unit_pass = True
    for lam in [[3], [2, 1], [1, 1, 1]]:
        key = ((3,), tuple(lam), tuple(lam))
        expected = 1
        actual = coefficients.get(key, 0)
        status = "✓" if actual == expected else "✗"
        print(f"  [3] ⊗ {lam} → {lam}: {actual} (expected {expected}) {status}")
        if actual != expected:
            unit_pass = False

    # Test 2: Symmetry
    print("\n2. Symmetry: g^ν_{λμ} = g^ν_{μλ}")
    sym_pass = True
    checked = set()
    for (lam, mu, nu), coeff in coefficients.items():
        if (lam, mu) in checked or (mu, lam) in checked:
            continue
        checked.add((lam, mu))
        checked.add((mu, lam))

        sym_coeff = coefficients.get((mu, lam, nu), 0)
        if coeff != sym_coeff:
            print(f"  ✗ g^{nu}_{{{lam},{mu}}} = {coeff} ≠ {sym_coeff} = g^{nu}_{{{mu},{lam}}}")
            sym_pass = False

    if sym_pass:
        print(f"  ✓ All checked pairs symmetric")

    # Test 3: Dimension formula
    print("\n3. Dimension formula: Σ g^ν_{λμ} d_ν = d_λ · d_μ")
    dims = {(3,): 1, (2, 1): 2, (1, 1, 1): 1}
    dim_pass = True

    for lam in [[3], [2, 1], [1, 1, 1]]:
        for mu in [[3], [2, 1], [1, 1, 1]]:
            expected = dims[tuple(lam)] * dims[tuple(mu)]
            computed = sum(
                coefficients.get((tuple(lam), tuple(mu), nu), 0) * dims[nu]
                for nu in [(3,), (2, 1), (1, 1, 1)]
            )
            status = "✓" if computed == expected else "✗"
            if computed != expected:
                print(f"  {lam} ⊗ {mu}: computed={computed}, expected={expected} {status}")
                dim_pass = False

    if dim_pass:
        print(f"  ✓ All dimension formulas correct")

    print(f"\nOverall: {'✓ ALL TESTS PASSED' if (unit_pass and sym_pass and dim_pass) else '✗ SOME TESTS FAILED'}")
    return unit_pass and sym_pass and dim_pass


if __name__ == "__main__":
    print("\n" + "🔍" * 35)
    print("S_3 GROUND TRUTH VALIDATION")
    print("🔍" * 35)

    # Phase 1: Structure
    structure_ok = verify_s3_structure()

    # Phase 2: Character table
    char_table_ok = verify_s3_character_table()

    # Phase 3: Orthogonality (determines column order!)
    # This is done inside compute_s3_kronecker_by_hand

    # Phase 4: Compute coefficients
    coeffs = compute_s3_kronecker_by_hand()

    # Phase 5: Validate
    all_pass = validate_s3_properties(coeffs)

    print("\n" + "=" * 70)
    print("S_3 Validation Summary")
    print("=" * 70)
    print(f"  Structure: {'✓' if structure_ok else '✗'}")
    print(f"  Character table: {'✓' if char_table_ok else '✗'}")
    print(f"  Kronecker validation: {'✓' if all_pass else '✗'}")

    if all_pass:
        print("\n✅ S_3 VALIDATION COMPLETE - Ground truth established!")
    else:
        print("\n⚠️  S_3 validation failed - fix this before proceeding to S_6!")
