"""
Algebraic Training via Character Theory

Instead of backpropagation, use representation theory of cyclic groups!

Key Insight:
-----------
Rotation attention learns weights over Galois group Gal(ℚ(ζ_n)/ℚ) ≅ C_n.

By Maschke's theorem, every representation decomposes into irreducible
characters χ_0, χ_1, ..., χ_{n-1}.

Learning rotation weights = learning character coefficients!

This is a LINEAR problem - solve with least squares (NO gradient descent!).

Mathematical Framework:
----------------------
1. Regular representation: R = ⊕_{j=0}^{n-1} χ_j
2. Decompose V: V = Σ_j Proj_{χ_j}(V)
3. Learn coefficients: Find c_j such that Σ_j c_j · Proj_{χ_j}(V) ≈ target
4. Solve: c = (A^T A)^{-1} A^T b (closed-form!)

Connection to Counting Task:
----------------------------
For counting by step s, the model needs to attend to:
- Previous element (rotation by -1)
- Detect the step size s

In Fourier (character) space:
- Low-frequency characters = global patterns
- High-frequency characters = local differences

The step detection should naturally emerge in certain characters!
"""

import numpy as np
from typing import Tuple, List
import sys

from counting_task import CountingTaskDataset
from character_theory_attention import CyclicGroupCharacters
from fhe_transformer import FHETransformer


class CharacterAttentionLearner:
    """
    Learn attention via character theory (algebraic approach).

    This replaces gradient descent with orthogonal projections
    and least squares - purely algebraic!
    """

    def __init__(
        self,
        seq_len: int = 8,
        d_model: int = 32,
        group_order: int = 8
    ):
        self.seq_len = seq_len
        self.d_model = d_model

        # Character theory for cyclic group
        self.group = CyclicGroupCharacters(group_order)
        self.n_characters = group_order

        # Learned character coefficients
        self.character_weights = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        embedding_matrix: np.ndarray,
        position_encoding: np.ndarray
    ) -> np.ndarray:
        """
        Learn character weights via least squares.

        For each training example:
        1. Embed and add position encoding
        2. Decompose into characters
        3. Learn which character combination predicts target

        Args:
            X_train: Training inputs [N, seq_len]
            y_train: Training targets [N]
            embedding_matrix: [vocab_size, d_model]
            position_encoding: [seq_len, d_model]

        Returns:
            Character weights [n_characters]
        """
        n_samples = len(X_train)

        print(f"Learning character weights from {n_samples} examples...")
        print(f"  Group order: {self.group.n}")
        print(f"  Number of characters: {self.n_characters}")

        # Build linear system: A·c = b
        # where c are character coefficients

        all_projections = []
        all_targets = []

        for i in range(n_samples):
            # 1. Embed input sequence
            V = np.zeros((self.seq_len, self.d_model))
            for j in range(len(X_train[i])):
                V[j] = embedding_matrix[X_train[i][j]]

            # Add position encoding
            V += position_encoding[:self.seq_len]

            # 2. Decompose into characters
            projs = self.group.decompose_into_characters(V)

            # 3. Store projections
            all_projections.append(projs)

            # 4. Target: next token embedding
            target_embedding = embedding_matrix[y_train[i]]
            all_targets.append(target_embedding)

        # Build design matrix A
        # Each row corresponds to one training example
        # Each column corresponds to one character

        n_features = self.d_model  # Simplified: use last position only

        A = np.zeros((n_samples * n_features, self.n_characters), dtype=complex)
        b = np.zeros(n_samples * n_features, dtype=complex)

        for i in range(n_samples):
            projs = all_projections[i]
            target = all_targets[i]

            # Use projection at last position (where we predict)
            # Note: len(projs) may be min(seq_len, n_characters)
            for j in range(min(len(projs), self.n_characters)):
                proj_last = projs[j][-1]  # Last position
                A[i*n_features:(i+1)*n_features, j] = proj_last

            b[i*n_features:(i+1)*n_features] = target

        # Solve least squares: c = (A^H A)^{-1} A^H b
        print("\nSolving least squares...")
        c, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)

        self.character_weights = c

        if len(residuals) > 0:
            print(f"  Residual: {residuals[0]:.6f}")
        else:
            print(f"  Residual: N/A")
        print(f"  Matrix rank: {rank}/{self.n_characters}")
        print(f"\nLearned character weights (magnitude):")
        for j in range(min(8, self.n_characters)):
            print(f"  χ_{j}: {np.abs(c[j]):8.5f}")

        return c

    def predict(
        self,
        X: np.ndarray,
        embedding_matrix: np.ndarray,
        position_encoding: np.ndarray
    ) -> np.ndarray:
        """
        Predict using learned character weights.

        Args:
            X: Input sequence [seq_len]
            embedding_matrix: [vocab_size, d_model]
            position_encoding: [seq_len, d_model]

        Returns:
            Predicted embedding [d_model]
        """
        if self.character_weights is None:
            raise ValueError("Must call fit() first!")

        # Embed input
        V = np.zeros((self.seq_len, self.d_model))
        for j in range(len(X)):
            V[j] = embedding_matrix[X[j]]

        # Add position encoding
        V += position_encoding[:self.seq_len]

        # Decompose into characters
        projs = self.group.decompose_into_characters(V)

        # Weighted combination of character projections
        prediction = np.zeros(self.d_model, dtype=complex)
        for j in range(min(len(projs), self.n_characters)):
            projection_last = projs[j][-1]  # Last position
            prediction += self.character_weights[j] * projection_last

        # Return real part (should be mostly real anyway)
        return prediction.real


def main():
    """Main training script using character theory."""
    print("=" * 70)
    print("Algebraic Training via Character Theory")
    print("=" * 70)
    print()

    # 1. Generate dataset
    print("1. Generating counting task dataset...")
    dataset = CountingTaskDataset(max_value=100, seq_len=5)
    X_train, y_train, X_test, y_test = dataset.generate_dataset(
        train_size=200,
        test_size=50
    )

    stats = dataset.analyze_dataset(X_train, y_train)
    print(f"   Training examples: {stats['total_examples']}")
    print(f"   Step distribution: {stats['step_distribution']}")

    # 2. Create embeddings
    print("\n2. Initializing embeddings...")
    vocab_size = 104
    d_model = 16  # Smaller for faster computation
    seq_len = 4  # Input sequence length

    np.random.seed(42)
    embedding_matrix = np.random.randn(vocab_size, d_model) * 0.1

    # Cyclotomic position encoding
    from cyclotomic_encoding import cyclotomic_encoding_matrix
    position_encoding = cyclotomic_encoding_matrix(seq_len, d_model, seq_len)

    print(f"   Vocabulary size: {vocab_size}")
    print(f"   Model dimension: {d_model}")
    print(f"   Sequence length: {seq_len}")

    # 3. Learn via character theory!
    print("\n3. Learning via character theory (NO backprop!)...")
    print("-" * 70)

    learner = CharacterAttentionLearner(
        seq_len=seq_len,
        d_model=d_model,
        group_order=8  # C_8
    )

    character_weights = learner.fit(
        X_train[:100],  # Use subset for speed
        y_train[:100],
        embedding_matrix,
        position_encoding
    )

    # 4. Evaluate
    print("\n4. Evaluating learned model...")
    print("-" * 70)

    correct = 0
    total = 0

    for i in range(len(X_test)):
        # Predict next token embedding
        pred_embedding = learner.predict(
            X_test[i],
            embedding_matrix,
            position_encoding
        )

        # Find closest embedding
        distances = np.linalg.norm(
            embedding_matrix - pred_embedding[np.newaxis, :],
            axis=1
        )
        pred_token = np.argmin(distances)

        if pred_token == y_test[i]:
            correct += 1
        total += 1

    accuracy = correct / total

    print(f"\nTest accuracy: {accuracy:.3f} ({correct}/{total})")

    # Show example predictions
    print("\nExample predictions:")
    for i in range(min(10, len(X_test))):
        pred_embedding = learner.predict(
            X_test[i],
            embedding_matrix,
            position_encoding
        )
        distances = np.linalg.norm(
            embedding_matrix - pred_embedding[np.newaxis, :],
            axis=1
        )
        pred_token = np.argmin(distances)

        step = dataset.get_step_from_sequence(X_test[i])
        status = "✓" if pred_token == y_test[i] else "✗"
        print(f"  {status} {list(X_test[i])} → {y_test[i]} "
              f"(predicted: {pred_token}, step={step})")

    print("\n" + "=" * 70)
    print("✅ Algebraic training complete!")
    print("=" * 70)
    print()
    print("Key insights:")
    print("  • Learning via character theory (representation theory!)")
    print("  • NO backpropagation - just least squares")
    print("  • Closed-form solution via orthogonal projections")
    print("  • Character weights = Fourier coefficients")
    print("  • This is how FHE naturally works (NTT)!")
    print()


if __name__ == "__main__":
    main()
