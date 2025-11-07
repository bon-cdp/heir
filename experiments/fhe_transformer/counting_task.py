"""
Counting Task Dataset Generator

Generate arithmetic sequence completion tasks to test FHE transformer.

Task: Given a sequence [start, start+step, start+2*step, ...], predict next number

Examples:
    Input: [2, 4, 6, 8]    → Output: 10  (count by 2s)
    Input: [3, 6, 9, 12]   → Output: 15  (count by 3s)
    Input: [1, 3, 5, 7]    → Output: 9   (odd numbers)
    Input: [5, 10, 15, 20] → Output: 25  (count by 5s)

Why this task is perfect for testing:
1. Requires attention: Model must attend to previous tokens to detect pattern
2. Simple enough: Small vocab (0-100), short sequences (4-8 tokens)
3. Verifiable: Easy to check if learned correctly
4. Interpretable: Can analyze which rotations are learned

Vocabulary:
- Tokens 0-100: Numbers
- Token 101: Padding token
- Token 102: Start token
- Token 103: End token
"""

import numpy as np
from typing import List, Tuple, Dict


class CountingTaskDataset:
    """
    Dataset for arithmetic sequence completion.

    Attributes:
        vocab_size: Size of vocabulary (default 104: 0-100 + special tokens)
        max_value: Maximum number in sequences (default 100)
        seq_len: Sequence length (default 5: 4 input + 1 output)
        pad_token: Padding token ID
        start_token: Start token ID
        end_token: End token ID
    """

    def __init__(
        self,
        max_value: int = 100,
        seq_len: int = 5,
        vocab_size: int = 104
    ):
        self.max_value = max_value
        self.seq_len = seq_len
        self.vocab_size = vocab_size

        # Special tokens
        self.pad_token = 101
        self.start_token = 102
        self.end_token = 103

    def generate_sequence(
        self,
        start: int,
        step: int,
        length: int
    ) -> Tuple[List[int], int]:
        """
        Generate a single arithmetic sequence.

        Args:
            start: Starting number
            step: Step size
            length: Sequence length (number of elements)

        Returns:
            (sequence, target) where sequence is input and target is next number
        """
        sequence = [start + i * step for i in range(length)]
        target = start + length * step

        return sequence, target

    def generate_batch(
        self,
        batch_size: int = 32,
        min_start: int = 0,
        max_start: int = 20,
        steps: List[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a batch of counting tasks.

        Args:
            batch_size: Number of examples to generate
            min_start: Minimum starting value
            max_start: Maximum starting value
            steps: List of possible step sizes (default [1,2,3,5])

        Returns:
            (inputs, targets) where:
                inputs: [batch_size, seq_len-1] token IDs
                targets: [batch_size] target token IDs
        """
        if steps is None:
            steps = [1, 2, 3, 5]

        inputs = []
        targets = []

        for _ in range(batch_size):
            # Random start and step
            start = np.random.randint(min_start, max_start + 1)
            step = np.random.choice(steps)

            # Generate sequence
            seq, target = self.generate_sequence(start, step, self.seq_len - 1)

            # Check if target is in vocabulary
            if target > self.max_value:
                # Retry with smaller start/step
                start = np.random.randint(0, 10)
                step = np.random.choice([1, 2, 3])
                seq, target = self.generate_sequence(start, step, self.seq_len - 1)

            # Skip if still out of bounds
            if target > self.max_value:
                continue

            inputs.append(seq)
            targets.append(target)

        return np.array(inputs), np.array(targets)

    def generate_dataset(
        self,
        train_size: int = 500,
        test_size: int = 100,
        steps: List[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate train and test datasets.

        Args:
            train_size: Number of training examples
            test_size: Number of test examples
            steps: List of possible step sizes

        Returns:
            (X_train, y_train, X_test, y_test)
        """
        if steps is None:
            steps = [1, 2, 3, 5]

        X_train, y_train = self.generate_batch(train_size, steps=steps)
        X_test, y_test = self.generate_batch(test_size, steps=steps)

        return X_train, y_train, X_test, y_test

    def get_step_from_sequence(self, sequence: List[int]) -> int:
        """Get the step size from a sequence."""
        if len(sequence) < 2:
            return 0
        return sequence[1] - sequence[0]

    def analyze_dataset(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict:
        """
        Analyze dataset statistics.

        Args:
            X: Input sequences [N, seq_len-1]
            y: Target values [N]

        Returns:
            Dictionary of statistics
        """
        stats = {}

        # Count examples per step size
        step_counts = {}
        for i in range(len(X)):
            seq = X[i]
            step = self.get_step_from_sequence(seq)
            step_counts[step] = step_counts.get(step, 0) + 1

        stats['step_distribution'] = step_counts
        stats['total_examples'] = len(X)
        stats['min_value'] = int(X.min())
        stats['max_value'] = int(max(X.max(), y.max()))
        stats['avg_sequence_start'] = float(X[:, 0].mean())

        return stats


def test_counting_task():
    """Test the counting task dataset generator."""
    print("=" * 70)
    print("Testing Counting Task Dataset")
    print("=" * 70)

    # Create dataset
    dataset = CountingTaskDataset(max_value=100, seq_len=5)

    # Test 1: Generate single sequence
    print("\n1. Single sequence generation")
    print("-" * 70)

    seq, target = dataset.generate_sequence(start=2, step=2, length=4)
    print(f"Sequence: {seq}")
    print(f"Target: {target}")
    print(f"Expected: [2, 4, 6, 8] → 10")
    assert seq == [2, 4, 6, 8], "Sequence generation failed"
    assert target == 10, "Target calculation failed"
    print("✓ Correct!")

    # Test 2: Generate batch
    print("\n2. Batch generation")
    print("-" * 70)

    X, y = dataset.generate_batch(batch_size=10)
    print(f"Input shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Expected: (10, 4) and (10,)")
    assert X.shape == (10, 4), "Batch input shape incorrect"
    assert y.shape == (10,), "Batch target shape incorrect"
    print("✓ Correct shapes!")

    # Show examples
    print("\nFirst 5 examples:")
    for i in range(min(5, len(X))):
        step = dataset.get_step_from_sequence(X[i])
        print(f"  {list(X[i])} → {y[i]} (step={step})")

    # Test 3: Generate full dataset
    print("\n3. Full dataset generation")
    print("-" * 70)

    X_train, y_train, X_test, y_test = dataset.generate_dataset(
        train_size=500,
        test_size=100
    )

    print(f"Train set: {X_train.shape}, {y_train.shape}")
    print(f"Test set: {X_test.shape}, {y_test.shape}")
    assert X_train.shape[0] == 500, "Train size incorrect"
    assert X_test.shape[0] == 100, "Test size incorrect"
    print("✓ Correct sizes!")

    # Test 4: Analyze dataset
    print("\n4. Dataset analysis")
    print("-" * 70)

    stats = dataset.analyze_dataset(X_train, y_train)
    print(f"Total examples: {stats['total_examples']}")
    print(f"Value range: [{stats['min_value']}, {stats['max_value']}]")
    print(f"Average starting value: {stats['avg_sequence_start']:.2f}")
    print("\nStep distribution:")
    for step, count in sorted(stats['step_distribution'].items()):
        percentage = 100 * count / stats['total_examples']
        print(f"  Step {step}: {count} examples ({percentage:.1f}%)")

    # Test 5: Verify sequences are valid
    print("\n5. Sequence validation")
    print("-" * 70)

    all_valid = True
    for i in range(len(X_train)):
        seq = X_train[i]
        target = y_train[i]
        step = dataset.get_step_from_sequence(seq)

        # Verify sequence is arithmetic
        for j in range(1, len(seq)):
            expected = seq[0] + j * step
            if seq[j] != expected:
                print(f"Invalid sequence at index {i}: {seq}")
                all_valid = False
                break

        # Verify target
        expected_target = seq[-1] + step
        if target != expected_target:
            print(f"Invalid target at index {i}: {seq} → {target} (expected {expected_target})")
            all_valid = False

    if all_valid:
        print(f"✓ All {len(X_train)} sequences valid!")
    else:
        print("✗ Some sequences invalid")

    print("\n" + "=" * 70)
    print("✅ Counting task dataset tests passed!")
    print("=" * 70)

    return dataset, X_train, y_train, X_test, y_test


if __name__ == "__main__":
    test_counting_task()
