# BUG REPORT: Character Table Column Ordering

## Date
2025-11-10

## Severity
**CRITICAL** - All S_6 Kronecker coefficients are incorrect

## Root Cause
Character table columns (conjugacy classes) are in **reverse order** compared to partition ordering from `Partition.generate_partitions(n)`.

## Evidence from S_3

### Partition Order (Rows)
```
[3], [2,1], [1,1,1]
```

### Column Order (Conjugacy Classes)
```
[[1,1,1], [2,1], [3]]  ← REVERSED!
```

### Proof
Orthogonality check shows:
- With class sizes `[1, 3, 2]` (order: identity, transpositions, 3-cycles): **error = 0.000000** ✓
- With class sizes `[2, 3, 1]` (order: 3-cycles, transpositions, identity): **error = 1.118034** ✗

## Impact on S_6

### Current (Incorrect) Setup
- Partition order: `[[6], [5,1], [4,2], [4,1,1], [3,3], [3,2,1], [3,1,1,1], [2,2,2], [2,2,1,1], [2,1,1,1,1], [1,1,1,1,1,1]]`
- Class sizes used: `[120, 144, 90, 120, 40, 90, 15, 40, 90, 144, 1]`
- Column order (assumed): same as partition order ← **WRONG!**

### Actual Column Order (Likely)
Reversed: `[[1,1,1,1,1,1], [2,1,1,1,1], [2,2,1,1], [2,2,2], [3,1,1,1], [3,2,1], [3,3], [4,1,1], [4,2], [5,1], [6]]`

### Validation Failures Explained
1. **Unit property**: [6]⊗[5,1]→[5,1] = 6 instead of 1
   - We're using wrong column for [5,1] conjugacy class
   - Getting character values from wrong column → wrong coefficient

2. **Dimension formula**: All fail because class sizes don't match columns
   - Example: [6]⊗[6] should sum to 1, we get 83
   - Inner product formula uses wrong class sizes for each column

3. **Symmetry violations**: 594/709
   - Only half the coefficients computed (mu >= lam)
   - Never stored symmetric pairs

## Fix Strategy

### Option 1: Reverse Class Sizes (Simplest)
```python
# S_6 class sizes in REVERSE order
class_sizes_s6 = [1, 144, 90, 40, 90, 120, 40, 90, 144, 120][::-1]
```

### Option 2: Determine Exact Column Order
1. Test all 11! possible orderings (impractical)
2. Use orthogonality to find correct order (computational)
3. Consult literature for canonical ordering

### Option 3: Get Character Table from Trusted Source
- Use SageMath to generate S_6 character table with documented ordering
- Store with explicit column labels

## Recommended Action

1. **Immediate**: Test Option 1 (reverse class sizes) on S_6
2. **Validate**: Check if orthogonality error becomes ~0
3. **Recompute**: All S_6 Kronecker coefficients with correct ordering
4. **Verify**: Unit property, dimension formula, symmetry

## Lesson Learned

**ALWAYS document index conventions explicitly!**

Character tables need:
- Row labels (partition for each irrep)
- Column labels (cycle type for each conjugacy class)
- Class sizes matching column order
- References for where table came from

Not just "here's a matrix from SageMath" with no metadata.
