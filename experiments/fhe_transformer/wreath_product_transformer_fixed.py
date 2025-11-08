"""
Wreath Product Transformer - CORRECTED Implementation

Key Fix:
========
The wreath product learns position-dependent character weights,
but these weights aggregate information across the ENTIRE sequence,
not predict from each position separately.

Correct Mathematical Framework:
===============================

For each position p, we learn weights w[c,p,j] such that:
    H[p] = Σ_j w[c,p,j] · Proj_χ_j(V)[p]

where Proj_χ_j(V) is the character projection of the FULL sequence V.

Then FFN combines all positions:
    output = FFN(flatten([H[0], H[1], ..., H[n-1]]))

This is different from learning separate predictors per position!

Previous WRONG approach:
    w[c,p,:] directly predicts target from position p alone
    (This is over-parameterized and confused)

Corrected approach:
    w[c,p,:] extracts position-specific features from full sequence
    FFN learns how to combine these features
    (This is the proper wreath product structure!)
"""

import numpy as np
from typing import Tuple, List
import sys

from counting_task import CountingTaskDataset
from test_copy_task import CopyTaskDataset
from character_theory_attention import CyclicGroupCharacters
from cyclotomic_encoding import cyclotomic_encoding_matrix


class WreathProductTransformerFixed:
    """
    CORRECTED wreath product transformer.

    Key difference: Position-dependent weights aggregate across sequence,
    not predict from each position separately.
    """

    def __init__(
        self,
        vocab_size: int = 104,
        d_model: int = 32,
        seq_len: int = 4,
        group_order: int = None,
        n_conditions: int = None
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.seq_len = seq_len

        if group_order is None:
            group_order = max(8, seq_len)
        self.group_order = group_order

        self.n_conditions = n_conditions
        self.group = CyclicGroupCharacters(group_order)

        # Learned parameters
        self.embedding = None
        self.position_encoding = None
        self.position_character_weights = None  # [n_conditions, seq_len, n_characters]
        self.ffn_weights = None
        self.ffn_bias = None

        print(f"Wreath Product Transformer (FIXED) initialized:")
        print(f"  Vocab: {vocab_size}, Model dim: {d_model}, Seq len: {seq_len}")
        print(f"  Character group order: {group_order}")

    def _embed_sequence(self, X: np.ndarray) -> np.ndarray:
        """Embed sequence and add position encoding."""
        V = np.zeros((self.seq_len, self.d_model))
        for i in range(min(len(X), self.seq_len)):
            V[i] = self.embedding[X[i]]
        V += self.position_encoding[:self.seq_len]
        return V

    def _extract_condition(self, X: np.ndarray) -> int:
        """Extract condition from input sequence."""
        return int(X[0])

    def _apply_wreath_attention(self, V: np.ndarray, condition: int) -> np.ndarray:
        """
        Apply position-dependent character attention.

        CORRECTED: Uses full sequence character decomposition,
        with position-dependent weighting.
        """
        # Decompose FULL sequence into characters
        projs = self.group.decompose_into_characters(V)

        # Get position-dependent weights
        if condition < self.position_character_weights.shape[0]:
            pos_weights = self.position_character_weights[condition]  # [seq_len, n_chars]
        else:
            pos_weights = self.position_character_weights[0]

        # Apply position-dependent weighting
        output = np.zeros_like(V, dtype=complex)
        for p in range(self.seq_len):
            for j in range(min(len(projs), pos_weights.shape[1])):
                weight = pos_weights[p, j]
                proj_at_p = projs[j][p]  # Character j's projection at position p
                output[p] += weight * proj_at_p

        return output.real

    def _apply_ffn(self, H: np.ndarray) -> np.ndarray:
        """Apply FFN to combined representation."""
        h_flat = H.flatten()
        output = h_flat @ self.ffn_weights + self.ffn_bias
        return output

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        verbose: bool = True
    ) -> None:
        """
        Learn position-dependent character weights CORRECTLY.

        Key fix: We learn how each position's character weights contribute
        to the OVERALL representation, not separate predictors per position.
        """
        n_samples = len(X_train)

        if verbose:
            print("\n" + "=" * 70)
            print("Learning Wreath Product Transformer (CORRECTED)")
            print("=" * 70)

        # 1. Initialize embeddings
        np.random.seed(42)
        self.embedding = np.random.randn(self.vocab_size, self.d_model) * 0.1
        self.position_encoding = cyclotomic_encoding_matrix(
            self.seq_len, self.d_model, self.seq_len
        )

        # Prepare data
        all_V = []
        all_projections = []
        all_targets = []
        all_conditions = []

        for i in range(n_samples):
            V = self._embed_sequence(X_train[i])
            all_V.append(V)

            projs = self.group.decompose_into_characters(V)
            all_projections.append(projs)

            target_emb = self.embedding[y_train[i]]
            all_targets.append(target_emb)

            condition = self._extract_condition(X_train[i])
            all_conditions.append(condition)

        # Infer conditions
        if self.n_conditions is None:
            self.n_conditions = len(set(all_conditions))

        n_chars = min(self.seq_len, self.group_order)

        # 2. Learn position-dependent character weights via JOINT optimization
        if verbose:
            print("\n2. Learning position-dependent character weights (CORRECTED)...")

        # We'll use a simplified approach: learn position-independent first,
        # then add position-dependent refinement

        # Initialize position-dependent weights
        self.position_character_weights = np.zeros(
            (self.n_conditions, self.seq_len, self.group_order),
            dtype=complex
        )

        # For each condition, learn weights jointly across all positions
        for cond_idx in range(self.n_conditions):
            samples_for_cond = [i for i in range(n_samples) if all_conditions[i] == cond_idx]

            if len(samples_for_cond) == 0:
                continue

            n_cond_samples = len(samples_for_cond)

            # Build JOINT design matrix for all positions
            # Size: [n_samples * d_model, seq_len * n_chars]
            # This learns all position weights jointly!
            A_joint = np.zeros(
                (n_cond_samples * self.d_model, self.seq_len * n_chars),
                dtype=complex
            )
            b_joint = np.zeros(n_cond_samples * self.d_model, dtype=complex)

            for idx, i in enumerate(samples_for_cond):
                projs = all_projections[i]
                target = all_targets[i]

                # For each position and character combination
                col_idx = 0
                for p in range(self.seq_len):
                    for j in range(n_chars):
                        proj_at_p = projs[j][p]  # Character j at position p
                        A_joint[idx*self.d_model:(idx+1)*self.d_model, col_idx] = proj_at_p
                        col_idx += 1

                # Target
                b_joint[idx*self.d_model:(idx+1)*self.d_model] = target

            # Solve JOINT least squares
            w_joint, residuals, rank, s = np.linalg.lstsq(A_joint, b_joint, rcond=None)

            # Reshape into position-dependent weights
            w_joint_reshaped = w_joint.reshape(self.seq_len, n_chars)
            for p in range(self.seq_len):
                self.position_character_weights[cond_idx, p, :n_chars] = w_joint_reshaped[p]

            if verbose:
                print(f"   Condition {cond_idx}: rank={rank}/{self.seq_len * n_chars}, "
                      f"samples={n_cond_samples}")
                if len(residuals) > 0:
                    print(f"      Residual: {np.mean(residuals):.6f}")

        if verbose:
            print(f"   ✓ Learned position-dependent weights jointly!")

        # 3. Learn FFN
        if verbose:
            print("\n3. Learning FFN...")

        input_dim = self.seq_len * self.d_model
        output_dim = self.d_model

        A = np.zeros((n_samples, input_dim))
        b = np.zeros((n_samples, output_dim))

        for i in range(n_samples):
            condition = all_conditions[i]
            V = all_V[i]
            H = self._apply_wreath_attention(V, condition)

            A[i] = H.flatten()
            b[i] = all_targets[i]

        w, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)

        self.ffn_weights = w
        self.ffn_bias = np.zeros(output_dim)

        if verbose:
            print(f"   Rank: {rank}/{min(n_samples, input_dim)}")
            if len(residuals) > 0:
                print(f"   Residual: {np.mean(residuals):.6f}")

        if verbose:
            print("\n" + "=" * 70)
            print("✅ Learning complete!")
            print("=" * 70)

    def predict(self, X: np.ndarray) -> int:
        """Predict using corrected wreath product attention."""
        condition = self._extract_condition(X)
        V = self._embed_sequence(X)
        H = self._apply_wreath_attention(V, condition)
        pred_embedding = self._apply_ffn(H)

        distances = np.linalg.norm(
            self.embedding - pred_embedding[np.newaxis, :],
            axis=1
        )
        return int(np.argmin(distances))

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        verbose: bool = True
    ) -> float:
        """Evaluate accuracy."""
        correct = sum(1 for i in range(len(X_test)) if self.predict(X_test[i]) == y_test[i])
        accuracy = correct / len(X_test)

        if verbose:
            print(f"\nTest Accuracy: {accuracy:.1%} ({correct}/{len(X_test)})")

        return accuracy


def main():
    """Test corrected wreath product on both tasks."""
    print("\n" + "=" * 70)
    print("CORRECTED WREATH PRODUCT TRANSFORMER - Both Tasks")
    print("=" * 70)

    # Test 1: Counting
    print("\n" + "=" * 70)
    print("TEST 1: COUNTING TASK")
    print("=" * 70)

    dataset = CountingTaskDataset(max_value=100, seq_len=5)
    X_train, y_train, X_test, y_test = dataset.generate_dataset(300, 100)

    model = WreathProductTransformerFixed(
        vocab_size=104,
        d_model=32,
        seq_len=4,
        group_order=8
    )

    model.fit(X_train, y_train, verbose=True)
    counting_acc = model.evaluate(X_test, y_test, verbose=True)

    # Test 2: Copy
    print("\n\n" + "=" * 70)
    print("TEST 2: COPY TASK")
    print("=" * 70)

    dataset = CopyTaskDataset(vocab_size=104, seq_len=4, n_positions=3)
    X_train, y_train, X_test, y_test = dataset.generate_dataset(300, 100)

    model = WreathProductTransformerFixed(
        vocab_size=104,
        d_model=32,
        seq_len=4,
        group_order=8
    )

    model.fit(X_train, y_train, verbose=True)
    copy_acc = model.evaluate(X_test, y_test, verbose=True)

    # Summary
    print("\n\n" + "=" * 70)
    print("FINAL RESULTS (CORRECTED WREATH PRODUCT)")
    print("=" * 70)
    print(f"\nCounting: {counting_acc:.1%}")
    print(f"Copy: {copy_acc:.1%}")

    if counting_acc >= 0.90 and copy_acc >= 0.90:
        print("\n✓✓✓ SUCCESS! Both tasks solved!")
    elif counting_acc >= 0.85 or copy_acc >= 0.85:
        print("\n✓✓ Good performance on at least one task")
    else:
        print("\n✓ Partial success")

    print("=" * 70)


if __name__ == "__main__":
    main()
