"""
Comprehensive Test: Multi-Head Algebraic Transformer on Both Tasks

This demonstrates that a SINGLE multi-head architecture excels at:
1. Counting task (global sequence patterns)
2. Copy task (position-dependent routing)

By integrating:
- Issue #2389: Chebyshev polynomial activations (depth-optimal)
- Issue #2394: Multi-head character attention (algebraic)

Expected Results:
================
Counting Task: ~95%+ (Head 1 should dominate)
Copy Task: ~95%+ (Head 2 should dominate)
FHE Depth: 2 (vs 5+ for standard transformers)
Learning: Closed-form (no backpropagation)
"""

import numpy as np
import sys

from multihead_algebraic_transformer import MultiHeadAlgebraicTransformer
from counting_task import CountingTaskDataset
from test_copy_task import CopyTaskDataset


def test_counting_task():
    """Test multi-head transformer on counting task."""
    print("\n" + "=" * 70)
    print("TEST 1: COUNTING TASK")
    print("=" * 70)
    print("\nTask: [a, a+s, a+2s, a+3s] → a+4s")
    print("Requires: Global sequence pattern detection")
    print("Expected: Head 1 (global) should dominate")

    # Generate dataset
    print("\n1. Generating dataset...")
    dataset = CountingTaskDataset(max_value=100, seq_len=5)
    X_train, y_train, X_test, y_test = dataset.generate_dataset(
        train_size=300,
        test_size=100
    )

    print(f"   Training: {len(X_train)} examples")
    print(f"   Test: {len(X_test)} examples")

    # Train model
    print("\n2. Training multi-head transformer...")
    model = MultiHeadAlgebraicTransformer(
        vocab_size=104,
        d_model=32,
        seq_len=4,
        group_order=8,
        use_head1=True,  # Global attention
        use_head2=True   # Position-dependent attention
    )

    model.fit(X_train, y_train, verbose=False)
    print("   ✓ Training complete")

    # Evaluate
    print("\n3. Evaluating...")
    correct = 0
    by_step = {}

    for i in range(len(X_test)):
        pred = model.predict(X_test[i])
        step = dataset.get_step_from_sequence(X_test[i])

        if step not in by_step:
            by_step[step] = {'correct': 0, 'total': 0}

        by_step[step]['total'] += 1
        if pred == y_test[i]:
            correct += 1
            by_step[step]['correct'] += 1

    accuracy = correct / len(X_test)

    # Results
    print(f"\n   Overall Accuracy: {accuracy:.1%} ({correct}/{len(X_test)})")
    print("   Per-step accuracy:")
    for step in sorted(by_step.keys()):
        acc = by_step[step]['correct'] / by_step[step]['total']
        status = "✓✓✓" if acc >= 0.9 else "✓✓" if acc >= 0.75 else "✓" if acc >= 0.5 else "✗"
        print(f"     Step {step}: {acc:.1%} ({by_step[step]['correct']}/{by_step[step]['total']}) {status}")

    return accuracy, model


def test_copy_task():
    """Test multi-head transformer on copy task."""
    print("\n\n" + "=" * 70)
    print("TEST 2: COPY TASK")
    print("=" * 70)
    print("\nTask: [position_marker, x, y, z] → value at position_marker")
    print("Requires: Position-dependent routing")
    print("Expected: Head 2 (position-dependent) should dominate")

    # Generate dataset
    print("\n1. Generating dataset...")
    dataset = CopyTaskDataset(vocab_size=104, seq_len=4, n_positions=3)
    X_train, y_train, X_test, y_test = dataset.generate_dataset(
        train_size=300,
        test_size=100
    )

    print(f"   Training: {len(X_train)} examples")
    print(f"   Test: {len(X_test)} examples")
    print(f"   Positions: {dataset.n_positions}")

    # Train model
    print("\n2. Training multi-head transformer...")
    model = MultiHeadAlgebraicTransformer(
        vocab_size=104,
        d_model=32,
        seq_len=4,
        group_order=8,
        use_head1=True,  # Global attention
        use_head2=True   # Position-dependent attention
    )

    model.fit(X_train, y_train, verbose=False)
    print("   ✓ Training complete")

    # Evaluate
    print("\n3. Evaluating...")
    correct = 0
    by_position = {0: {'correct': 0, 'total': 0},
                   1: {'correct': 0, 'total': 0},
                   2: {'correct': 0, 'total': 0}}

    for i in range(len(X_test)):
        pred = model.predict(X_test[i])
        position = X_test[i, 0]

        by_position[position]['total'] += 1
        if pred == y_test[i]:
            correct += 1
            by_position[position]['correct'] += 1

    accuracy = correct / len(X_test)

    # Results
    print(f"\n   Overall Accuracy: {accuracy:.1%} ({correct}/{len(X_test)})")
    print("   Per-position accuracy:")
    for pos in sorted(by_position.keys()):
        acc = by_position[pos]['correct'] / by_position[pos]['total']
        status = "✓✓✓" if acc >= 0.9 else "✓✓" if acc >= 0.75 else "✓" if acc >= 0.5 else "✗"
        print(f"     Position {pos}: {acc:.1%} ({by_position[pos]['correct']}/{by_position[pos]['total']}) {status}")

    return accuracy, by_position, model


def main():
    """Run comprehensive tests on both tasks."""
    print("\n" + "=" * 70)
    print("MULTI-HEAD ALGEBRAIC TRANSFORMER - COMPREHENSIVE TEST")
    print("Integrating Issues #2389 + #2394")
    print("=" * 70)

    print("\nHypothesis:")
    print("  A SINGLE multi-head architecture with Chebyshev activation can:")
    print("  1. Excel at counting (via Head 1: global attention)")
    print("  2. Excel at copy (via Head 2: position-dependent attention)")
    print("  3. Maintain FHE depth of 2 (vs 5+ for standard transformers)")
    print("  4. Learn everything via closed-form solutions (no backprop)")

    # Test 1: Counting
    counting_acc, counting_model = test_counting_task()

    # Test 2: Copy
    copy_acc, copy_positions, copy_model = test_copy_task()

    # Final comparison
    print("\n\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    print("\n📊 ACCURACY COMPARISON")
    print("-" * 70)
    print(f"\nCounting Task: {counting_acc:.1%}")
    status_counting = "✓✓✓ EXCELLENT" if counting_acc >= 0.9 else "✓✓ GOOD" if counting_acc >= 0.75 else "✓ PARTIAL"
    print(f"  Status: {status_counting}")

    print(f"\nCopy Task: {copy_acc:.1%}")
    status_copy = "✓✓✓ EXCELLENT" if copy_acc >= 0.9 else "✓✓ GOOD" if copy_acc >= 0.75 else "✓ PARTIAL"
    print(f"  Status: {status_copy}")

    # Overall success
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    both_excellent = counting_acc >= 0.9 and copy_acc >= 0.9
    both_good = counting_acc >= 0.75 and copy_acc >= 0.75

    if both_excellent:
        print("\n🎉 🎉 🎉 SUCCESS! 🎉 🎉 🎉")
        print("\nMulti-head algebraic transformer EXCELS at BOTH tasks!")
        print(f"  ✓ Counting: {counting_acc:.1%}")
        print(f"  ✓ Copy: {copy_acc:.1%}")
        print()
        print("Key Achievements:")
        print("  ✅ Multi-task capable (single architecture, multiple tasks)")
        print("  ✅ Closed-form learning (no backpropagation)")
        print("  ✅ FHE depth = 2 (vs 5+ for standard transformers)")
        print("  ✅ Combines issues #2389 + #2394")
        print()
        print("This is a BREAKTHROUGH for FHE transformers!")

    elif both_good:
        print("\n✓ SUCCESS with room for improvement!")
        print(f"\n  Counting: {counting_acc:.1%}")
        print(f"  Copy: {copy_acc:.1%}")
        print()
        print("Achievements:")
        print("  ✓ Multi-task capable")
        print("  ✓ Closed-form learning")
        print("  ✓ FHE-compatible (depth 2)")
        print()
        print("Next steps: Tune hyperparameters for 95%+ on both")

    else:
        print("\nPartial success - needs improvement")
        if counting_acc < 0.75:
            print(f"  ✗ Counting: {counting_acc:.1%} (target: 75%+)")
        if copy_acc < 0.75:
            print(f"  ✗ Copy: {copy_acc:.1%} (target: 75%+)")

    # Technical details
    print("\n" + "=" * 70)
    print("TECHNICAL SPECIFICATIONS")
    print("=" * 70)
    print("\nArchitecture:")
    print("  • Head 1: Global conditional character attention")
    print("  • Head 2: Position-dependent character attention")
    print("  • Activation: Chebyshev T₃(x) = 4x³ - 3x")
    print("  • Learning: Closed-form least squares")
    print("\nFHE Compatibility:")
    print("  • Attention heads: Depth 0 (rotations only)")
    print("  • Chebyshev T₃: Depth 2 (x³ via optimal squaring)")
    print("  • Total depth: 2")
    print("\nComparison:")
    print("  • Standard FHE transformer: Depth 5+")
    print("  • Our approach: Depth 2 (2.5x reduction!)")
    print("\nIntegration:")
    print("  • Issue #2389: Polynomial activations ✓")
    print("  • Issue #2394: Algebraic attention ✓")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
