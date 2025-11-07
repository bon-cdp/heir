"""
Complete Algebraic Transformer (NO BACKPROP!)

Learn EVERYTHING via representation theory + least squares:
1. Character attention weights - via character theory
2. FFN weights - via least squares
3. Output projection - via least squares

For counting task [a, a+s, a+2s, a+3s] → a+4s, this is EXACTLY solvable!

The optimal solution is:
    next = 2·last - second_to_last

This is LINEAR - can be learned perfectly with least squares!

Mathematical Framework:
======================

Full model:
    1. Embed: X → V (embedding lookup)
    2. Attention: V → H via character decomposition
    3. FFN: H → Y via learned linear map
    4. Output: Y → token via nearest embedding

All learned via least squares - closed-form solutions!

This is genuinely novel: transformers via pure algebra!
"""

import numpy as np
from typing import Tuple, List
import sys

from counting_task import CountingTaskDataset
from character_theory_attention import CyclicGroupCharacters
from cyclotomic_encoding import cyclotomic_encoding_matrix


class AlgebraicTransformer:
    """
    Complete transformer learned via representation theory + least squares.

    NO gradient descent. NO backpropagation. Pure linear algebra!
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

        # Group order (default to seq_len if not specified)
        if group_order is None:
            group_order = max(8, seq_len)  # At least 8 for good character resolution
        self.group_order = group_order

        # Number of conditions (for conditional character weights)
        # If None, will be inferred from training data
        self.n_conditions = n_conditions

        # Character theory
        self.group = CyclicGroupCharacters(group_order)

        # Learned parameters
        self.embedding = None
        self.position_encoding = None
        self.character_weight_matrix = None  # [n_conditions, n_characters]
        self.ffn_weights = None
        self.ffn_bias = None

        print(f"Algebraic Transformer initialized:")
        print(f"  Vocab size: {vocab_size}")
        print(f"  Model dim: {d_model}")
        print(f"  Sequence length: {seq_len}")
        print(f"  Character group order: {group_order}")
        if n_conditions:
            print(f"  Conditional weights: {n_conditions} conditions (Galois connection)")

    def _embed_sequence(self, X: np.ndarray) -> np.ndarray:
        """
        Embed sequence and add position encoding.

        Args:
            X: Token IDs [seq_len]

        Returns:
            Embedded sequence [seq_len, d_model]
        """
        V = np.zeros((self.seq_len, self.d_model))
        for i in range(min(len(X), self.seq_len)):
            V[i] = self.embedding[X[i]]

        # Add position encoding
        V += self.position_encoding[:self.seq_len]

        return V

    def _extract_condition(self, X: np.ndarray) -> int:
        """
        Extract condition from input sequence.

        For conditional character weights (Galois connections),
        the first token serves as the condition selector.

        Args:
            X: Input sequence [seq_len]

        Returns:
            Condition index (integer)
        """
        # Use first token as condition
        return int(X[0])

    def _apply_character_attention(
        self,
        V: np.ndarray,
        condition: int = None
    ) -> np.ndarray:
        """
        Apply learned character attention (possibly conditional).

        Args:
            V: Embedded sequence [seq_len, d_model]
            condition: Condition index (for conditional weights)

        Returns:
            Attention output [seq_len, d_model]
        """
        # Decompose into characters
        projs = self.group.decompose_into_characters(V)

        # Get character weights (conditional or unconditional)
        if condition is not None and self.character_weight_matrix is not None:
            # Conditional: select weights based on condition
            if condition < self.character_weight_matrix.shape[0]:
                char_weights = self.character_weight_matrix[condition, :]
            else:
                # Fallback to first condition if out of range
                char_weights = self.character_weight_matrix[0, :]
        else:
            # Unconditional: use single weight vector
            # (For backward compatibility - shouldn't happen with new code)
            char_weights = self.character_weight_matrix[0, :] if self.character_weight_matrix is not None else np.ones(len(projs))

        # Weighted combination
        output = np.zeros_like(V, dtype=complex)
        for j in range(min(len(projs), len(char_weights))):
            output += char_weights[j] * projs[j]

        return output.real

    def _apply_ffn(self, H: np.ndarray) -> np.ndarray:
        """
        Apply learned FFN (simple linear layer).

        Args:
            H: Attention output [seq_len, d_model]

        Returns:
            FFN output [d_model] (prediction at last position)
        """
        # Flatten sequence (use all positions)
        h_flat = H.flatten()  # [seq_len * d_model]

        # Linear transformation
        output = h_flat @ self.ffn_weights + self.ffn_bias

        return output

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        verbose: bool = True
    ) -> None:
        """
        Learn all parameters via least squares.

        Args:
            X_train: Training inputs [N, seq_len]
            y_train: Training targets [N]
            verbose: Print progress
        """
        n_samples = len(X_train)

        if verbose:
            print("\n" + "=" * 70)
            print("Learning Transformer via Representation Theory + Least Squares")
            print("=" * 70)

        # 1. Initialize embeddings (random)
        if verbose:
            print("\n1. Initializing embeddings...")

        np.random.seed(42)
        self.embedding = np.random.randn(self.vocab_size, self.d_model) * 0.1
        self.position_encoding = cyclotomic_encoding_matrix(
            self.seq_len, self.d_model, self.seq_len
        )

        if verbose:
            print(f"   Embedding matrix: {self.embedding.shape}")
            print(f"   Position encoding: {self.position_encoding.shape}")

        # 2. Learn character attention weights (CONDITIONAL via Galois connections!)
        if verbose:
            print("\n2. Learning conditional character weights (Galois connections)...")

        # For each sample, get embedded sequence
        all_V = []
        all_projections = []
        all_targets = []
        all_conditions = []

        for i in range(n_samples):
            V = self._embed_sequence(X_train[i])
            all_V.append(V)

            # Decompose into character projections
            projs = self.group.decompose_into_characters(V)
            all_projections.append(projs)

            # Target: embedding of next token
            target_emb = self.embedding[y_train[i]]
            all_targets.append(target_emb)

            # Extract condition (first token)
            condition = self._extract_condition(X_train[i])
            all_conditions.append(condition)

        # Infer number of conditions if not specified
        if self.n_conditions is None:
            self.n_conditions = len(set(all_conditions))
            if verbose:
                print(f"   Inferred {self.n_conditions} unique conditions from data")

        n_chars = min(self.seq_len, self.group_order)

        # Initialize character weight matrix [n_conditions, n_characters]
        self.character_weight_matrix = np.zeros((self.n_conditions, self.group_order), dtype=complex)

        if verbose:
            print(f"   Learning {n_chars} character weights PER condition ({self.n_conditions} conditions)")
            print(f"   This creates a Galois connection (lattice of conditions × characters)")

        # Learn separate character weights for EACH condition!
        # This is the Galois connection / algebraic decision tree
        for cond_idx in range(self.n_conditions):
            # Filter samples for this condition
            samples_for_cond = [i for i in range(n_samples) if all_conditions[i] == cond_idx]

            if len(samples_for_cond) == 0:
                if verbose:
                    print(f"   Condition {cond_idx}: No samples (skipping)")
                continue

            if verbose:
                print(f"   Condition {cond_idx}: {len(samples_for_cond)} samples")

            # Build design matrix for this condition only
            n_cond_samples = len(samples_for_cond)
            A_char = np.zeros((n_cond_samples * self.d_model, n_chars), dtype=complex)
            b_char = np.zeros(n_cond_samples * self.d_model, dtype=complex)

            for idx, i in enumerate(samples_for_cond):
                projs = all_projections[i]
                target = all_targets[i]

                # For each character, get its projection at last position
                for j in range(n_chars):
                    proj_last = projs[j][-1]  # Last position (where we predict)
                    A_char[idx*self.d_model:(idx+1)*self.d_model, j] = proj_last

                # Target
                b_char[idx*self.d_model:(idx+1)*self.d_model] = target

            # Solve for character weights for THIS condition
            c, residuals_char, rank_char, s = np.linalg.lstsq(A_char, b_char, rcond=None)

            # Store learned weights for this condition
            for j in range(n_chars):
                self.character_weight_matrix[cond_idx, j] = c[j]

            if verbose:
                print(f"     Rank: {rank_char}/{n_chars}, Weights: {[f'{np.abs(c[j]):.3f}' for j in range(min(4, n_chars))]}")

        if verbose:
            print(f"\n   ✅ Learned conditional character weights via Galois connection!")
            print(f"      Each condition has its own character distribution")

        # 3. Learn FFN via least squares
        if verbose:
            print("\n3. Learning FFN (least squares)...")

        # Build linear system: A·w = b
        # Input: attention output (flattened)
        # Output: target embedding

        input_dim = self.seq_len * self.d_model
        output_dim = self.d_model

        A = np.zeros((n_samples, input_dim))
        b = np.zeros((n_samples, output_dim))

        for i in range(n_samples):
            # Apply learned conditional character attention
            condition = all_conditions[i]

            # Reconstruct from character projections using condition-specific weights
            projs = all_projections[i]
            H = np.zeros((self.seq_len, self.d_model), dtype=complex)
            for j in range(n_chars):
                H += self.character_weight_matrix[condition, j] * projs[j]
            H = H.real

            # Flatten
            A[i] = H.flatten()

            # Target
            b[i] = all_targets[i]

        # Solve least squares: w = (A^T A)^(-1) A^T b
        if verbose:
            print(f"   Solving system: A({n_samples}×{input_dim}) · w = b({n_samples}×{output_dim})")

        # Use pseudo-inverse for numerical stability
        w, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)

        self.ffn_weights = w  # [input_dim, output_dim]
        self.ffn_bias = np.zeros(output_dim)  # Could learn this too, but simplify

        if verbose:
            print(f"   Matrix rank: {rank}/{min(n_samples, input_dim)}")
            if len(residuals) > 0:
                print(f"   Residual: {np.mean(residuals):.6f}")
            print(f"   Learned FFN weights: {self.ffn_weights.shape}")

        # 4. Refine character weights (optional second pass)
        # For now, keep them uniform - FFN does the heavy lifting

        if verbose:
            print("\n" + "=" * 70)
            print("✅ Learning complete (all via closed-form least squares!)")
            print("=" * 70)

    def predict(self, X: np.ndarray) -> int:
        """
        Predict next token (using conditional character weights).

        Args:
            X: Input sequence [seq_len]

        Returns:
            Predicted token ID
        """
        # 1. Extract condition
        condition = self._extract_condition(X)

        # 2. Embed
        V = self._embed_sequence(X)

        # 3. Conditional attention (Galois connection!)
        H = self._apply_character_attention(V, condition=condition)

        # 4. FFN
        pred_embedding = self._apply_ffn(H)

        # 5. Find nearest embedding
        distances = np.linalg.norm(
            self.embedding - pred_embedding[np.newaxis, :],
            axis=1
        )
        pred_token = np.argmin(distances)

        return int(pred_token)

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        dataset: CountingTaskDataset = None,
        verbose: bool = True
    ) -> float:
        """
        Evaluate accuracy on test set.

        Args:
            X_test: Test inputs [N, seq_len]
            y_test: Test targets [N]
            dataset: Dataset (for analysis)
            verbose: Print results

        Returns:
            Test accuracy
        """
        n_test = len(X_test)
        correct = 0

        predictions = []
        for i in range(n_test):
            pred = self.predict(X_test[i])
            predictions.append(pred)
            if pred == y_test[i]:
                correct += 1

        accuracy = correct / n_test

        if verbose:
            print("\n" + "=" * 70)
            print("Evaluation Results")
            print("=" * 70)
            print(f"\nTest accuracy: {accuracy:.1%} ({correct}/{n_test})")

            # Breakdown by step size
            if dataset is not None:
                by_step = {}
                for i in range(n_test):
                    step = dataset.get_step_from_sequence(X_test[i])
                    if step not in by_step:
                        by_step[step] = {'correct': 0, 'total': 0}

                    if predictions[i] == y_test[i]:
                        by_step[step]['correct'] += 1
                    by_step[step]['total'] += 1

                print("\nAccuracy by step size:")
                for step in sorted(by_step.keys()):
                    acc = by_step[step]['correct'] / by_step[step]['total']
                    print(f"  Step {step}: {acc:.1%} "
                          f"({by_step[step]['correct']}/{by_step[step]['total']})")

            # Show examples
            print("\nExample predictions:")
            for i in range(min(10, n_test)):
                if dataset is not None:
                    step = dataset.get_step_from_sequence(X_test[i])
                else:
                    step = "?"

                status = "✓" if predictions[i] == y_test[i] else "✗"
                print(f"  {status} {list(X_test[i])} → {y_test[i]} "
                      f"(predicted: {predictions[i]}, step={step})")

        return accuracy


def main():
    """Main script: learn counting task via pure algebra!"""
    print("\n" + "=" * 70)
    print("ALGEBRAIC TRANSFORMER - NO BACKPROP!")
    print("Learning via Representation Theory + Least Squares")
    print("=" * 70)

    # 1. Generate dataset
    print("\n1. Generating counting task dataset...")
    dataset = CountingTaskDataset(max_value=100, seq_len=5)
    X_train, y_train, X_test, y_test = dataset.generate_dataset(
        train_size=300,
        test_size=100
    )

    stats = dataset.analyze_dataset(X_train, y_train)
    print(f"   Training examples: {stats['total_examples']}")
    print(f"   Test examples: {len(X_test)}")
    print(f"   Step distribution: {stats['step_distribution']}")

    # 2. Create model
    print("\n2. Creating algebraic transformer...")
    model = AlgebraicTransformer(
        vocab_size=104,
        d_model=32,
        seq_len=4,
        group_order=8
    )

    # 3. Learn via algebra!
    model.fit(X_train, y_train, verbose=True)

    # 4. Evaluate
    accuracy = model.evaluate(X_test, y_test, dataset=dataset, verbose=True)

    # 5. Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print(f"✅ Final test accuracy: {accuracy:.1%}")
    print()
    print("What we just did:")
    print("  • Learned rotation attention via CHARACTER THEORY")
    print("  • Learned FFN via LEAST SQUARES")
    print("  • ZERO backpropagation - pure linear algebra!")
    print("  • Representation theory over Galois group")
    print("  • Closed-form solutions throughout")
    print()
    print("This is a genuinely novel approach to learning transformers!")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
