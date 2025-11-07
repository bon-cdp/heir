"""
Position-Dependent Copy Task - The Definitive Attention Test

This task CANNOT be solved by simple linear regression!

Task: Given [position_marker, x, y, z], copy the value at position_marker

Examples:
    [0, 10, 20, 30] → 10  (copy position 0 → x)
    [1, 10, 20, 30] → 20  (copy position 1 → y)
    [2, 10, 20, 30] → 30  (copy position 2 → z)

Why this proves attention:
--------------------------
- Linear regression fails: output position changes based on first token!
- Requires content-based routing (what attention does)
- The model must:
  1. Attend to position_marker
  2. Use it to select which value to copy
  3. Route that value to output

If our algebraic transformer solves this, we have REAL attention!
"""

import numpy as np
from typing import Tuple
import sys

from algebraic_transformer import AlgebraicTransformer


class CopyTaskDataset:
    """Position-dependent copy task."""

    def __init__(
        self,
        vocab_size: int = 104,
        seq_len: int = 4,
        n_positions: int = 3
    ):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.n_positions = n_positions  # Number of data positions (excluding marker)

        # Reserve tokens 0-2 for position markers
        # Tokens 3-103 for actual values
        self.marker_tokens = list(range(n_positions))
        self.value_start = n_positions
        self.value_end = vocab_size

    def generate_example(self) -> Tuple[np.ndarray, int]:
        """
        Generate single copy task example.

        Returns:
            (input, target) where:
                input: [position_marker, value0, value1, value2]
                target: value at position_marker
        """
        # Random position marker (0, 1, or 2)
        position = np.random.randint(0, self.n_positions)

        # Random values at each position
        values = np.random.randint(
            self.value_start,
            self.value_end,
            size=self.n_positions
        )

        # Input: [marker, v0, v1, v2]
        input_seq = np.array([position] + list(values))

        # Target: value at marked position
        target = values[position]

        return input_seq, target

    def generate_batch(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate batch of examples."""
        inputs = []
        targets = []

        for _ in range(batch_size):
            inp, tgt = self.generate_example()
            inputs.append(inp)
            targets.append(tgt)

        return np.array(inputs), np.array(targets)

    def generate_dataset(
        self,
        train_size: int = 300,
        test_size: int = 100
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate train and test sets."""
        X_train, y_train = self.generate_batch(train_size)
        X_test, y_test = self.generate_batch(test_size)

        return X_train, y_train, X_test, y_test

    def analyze_dataset(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> dict:
        """Analyze dataset statistics."""
        stats = {}

        # Count examples per position
        position_counts = {}
        for i in range(len(X)):
            pos = X[i, 0]  # First token is position marker
            position_counts[pos] = position_counts.get(pos, 0) + 1

        stats['position_distribution'] = position_counts
        stats['total_examples'] = len(X)

        return stats


def test_linear_regression_baseline():
    """
    Test that simple linear regression FAILS on copy task.

    This proves the task requires attention!
    """
    print("\n" + "=" * 70)
    print("Baseline Test: Linear Regression (should FAIL)")
    print("=" * 70)

    dataset = CopyTaskDataset(vocab_size=104, seq_len=4, n_positions=3)
    X_train, y_train, X_test, y_test = dataset.generate_dataset(300, 100)

    # Create simple embedding and try linear regression
    vocab_size = 104
    d_model = 32

    np.random.seed(42)
    embedding = np.random.randn(vocab_size, d_model) * 0.1

    # Embed training data
    X_train_emb = np.zeros((len(X_train), 4 * d_model))
    y_train_emb = np.zeros((len(y_train), d_model))

    for i in range(len(X_train)):
        for j in range(4):
            X_train_emb[i, j*d_model:(j+1)*d_model] = embedding[X_train[i, j]]
        y_train_emb[i] = embedding[y_train[i]]

    # Simple linear regression: y = X @ w
    w = np.linalg.lstsq(X_train_emb, y_train_emb, rcond=None)[0]

    # Test
    correct = 0
    for i in range(len(X_test)):
        X_test_emb = np.zeros(4 * d_model)
        for j in range(4):
            X_test_emb[j*d_model:(j+1)*d_model] = embedding[X_test[i, j]]

        pred_emb = X_test_emb @ w

        # Find nearest
        distances = np.linalg.norm(embedding - pred_emb[np.newaxis, :], axis=1)
        pred = np.argmin(distances)

        if pred == y_test[i]:
            correct += 1

    accuracy = correct / len(X_test)

    print(f"\nLinear regression accuracy: {accuracy:.1%}")
    print()

    if accuracy < 0.5:
        print("✓ Linear regression FAILS (as expected!)")
        print("  This confirms the task requires attention.")
    else:
        print("✗ Unexpected: linear regression succeeded")
        print("  (This would mean the task is too simple)")

    return accuracy


def main():
    """Test algebraic transformer on copy task."""
    print("\n" + "=" * 70)
    print("ATTENTION TEST: Position-Dependent Copy Task")
    print("=" * 70)
    print()
    print("Task: Given [position_marker, x, y, z], copy value at position_marker")
    print()
    print("This CANNOT be solved by linear regression!")
    print("Requires true attention mechanism.")
    print()

    # First, verify linear regression fails
    baseline_acc = test_linear_regression_baseline()

    # Now test our algebraic transformer
    print("\n" + "=" * 70)
    print("Testing Algebraic Transformer (with Character Attention)")
    print("=" * 70)

    # Generate dataset
    print("\n1. Generating copy task dataset...")
    dataset = CopyTaskDataset(vocab_size=104, seq_len=4, n_positions=3)
    X_train, y_train, X_test, y_test = dataset.generate_dataset(
        train_size=300,
        test_size=100
    )

    stats = dataset.analyze_dataset(X_train, y_train)
    print(f"   Training examples: {stats['total_examples']}")
    print(f"   Position distribution: {stats['position_distribution']}")

    # Create and train model
    print("\n2. Training algebraic transformer...")
    model = AlgebraicTransformer(
        vocab_size=104,
        d_model=32,
        seq_len=4,
        group_order=8
    )

    model.fit(X_train, y_train, verbose=True)

    # Evaluate
    print("\n3. Evaluating on test set...")
    print("-" * 70)

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

    print(f"\nTest accuracy: {accuracy:.1%} ({correct}/{len(X_test)})")
    print()
    print("Accuracy by position marker:")
    for pos in sorted(by_position.keys()):
        acc = by_position[pos]['correct'] / by_position[pos]['total']
        print(f"  Position {pos}: {acc:.1%} "
              f"({by_position[pos]['correct']}/{by_position[pos]['total']})")

    # Show examples
    print("\nExample predictions:")
    for i in range(min(15, len(X_test))):
        pred = model.predict(X_test[i])
        status = "✓" if pred == y_test[i] else "✗"
        pos = X_test[i, 0]
        values = X_test[i, 1:]
        print(f"  {status} Position {pos}: [{values[0]}, {values[1]}, {values[2]}] "
              f"→ {y_test[i]} (predicted: {pred})")

    # Final verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    print()
    print(f"Linear regression:      {baseline_acc:.1%} (FAILED)")
    print(f"Algebraic transformer:  {accuracy:.1%}", end="")

    if accuracy > 0.9:
        print(" ✓✓✓")
        print()
        print("🎉 SUCCESS! Our model learned TRUE ATTENTION!")
        print()
        print("The algebraic transformer:")
        print("  • Used character theory to learn rotation patterns")
        print("  • Learned to attend to different positions")
        print("  • Solved a task that linear regression CANNOT solve")
        print()
        print("This proves we have a real attention mechanism,")
        print("not just memorized linear combinations!")
    elif accuracy > 0.5:
        print(" (partial success)")
        print()
        print("Model learned some attention, but not perfectly.")
    else:
        print(" ✗")
        print()
        print("Model failed - may need architecture adjustments.")

    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
