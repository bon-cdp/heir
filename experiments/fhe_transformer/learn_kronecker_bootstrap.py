"""
Learning Kronecker Coefficients via Sheaf-Wreath Bootstrap

The recursive meta-pattern:
1. We bootstrapped S_n from C_k (Galois connection) ✓
2. Now bootstrap predictions from examples (sheaf learning) ← YOU ARE HERE

Question: Can the learner discover Kronecker structure from partition features alone?
Can it learn representation theory from first principles?
"""

import numpy as np
import json
from pathlib import Path
from typing import List, Tuple, Dict
from dataclasses import dataclass
from symmetric_group_characters import Partition


@dataclass
class KroneckerExample:
    """A training example: (λ, μ, ν) → g^ν_{λμ}"""
    n: int  # Which S_n
    lambda_partition: Tuple[int, ...]
    mu_partition: Tuple[int, ...]
    nu_partition: Tuple[int, ...]
    coefficient: int
    features: np.ndarray = None  # Extracted features


class KroneckerLearner:
    """
    Learn Kronecker coefficients from examples via partition features.

    Features = partition statistics, combinatorial properties
    Target = g^ν_{λμ}

    Can it discover the inner product formula from data alone?
    """

    def __init__(self, verbose=True):
        self.verbose = verbose
        self.examples = []
        self.weights = None

    def load_known_coefficients(self, n_list: List[int]):
        """Load all known Kronecker coefficients as training data."""
        if self.verbose:
            print("=" * 70)
            print("Loading Known Kronecker Coefficients")
            print("=" * 70)

        for n in n_list:
            if self.verbose:
                print(f"\nLoading S_{n}...")

            # Load precomputed coefficients
            json_path = Path(f"s{n}_kronecker.json")
            if not json_path.exists():
                if self.verbose:
                    print(f"  ⚠️  File not found: {json_path}")
                    print(f"     Run: python3 compute_s5_s6.py {'s6' if n == 6 else ''}")
                continue

            with open(json_path) as f:
                coeffs = json.load(f)

            # Convert to examples
            n_examples = 0
            for key, coeff in coeffs.items():
                # Parse key: "((λ), (μ), (ν))"
                parts = eval(key)  # Safe since we generated this data
                lambda_part = parts[0]
                mu_part = parts[1]
                nu_part = parts[2]

                example = KroneckerExample(
                    n=n,
                    lambda_partition=lambda_part,
                    mu_partition=mu_part,
                    nu_partition=nu_part,
                    coefficient=coeff
                )
                self.examples.append(example)
                n_examples += 1

            if self.verbose:
                print(f"  Loaded {n_examples} examples from S_{n}")

        if self.verbose:
            print(f"\nTotal training examples: {len(self.examples)}")
            print("=" * 70)

    def extract_features(self, example: KroneckerExample) -> np.ndarray:
        """
        Extract features from partition combinatorics.

        The learner will discover which features predict coefficients!

        Feature categories:
        1. Partition statistics (lengths, first parts, etc.)
        2. Hook lengths (combinatorial structure)
        3. Conjugacy relationships
        4. Dimension hints (d_λ, d_μ, d_ν)
        """
        lambda_parts = list(example.lambda_partition)
        mu_parts = list(example.mu_partition)
        nu_parts = list(example.nu_partition)
        n = example.n

        features = []

        # Feature 1: Partition lengths (number of parts)
        features.append(len(lambda_parts))
        features.append(len(mu_parts))
        features.append(len(nu_parts))

        # Feature 2: First parts (largest cycles)
        features.append(lambda_parts[0] if lambda_parts else 0)
        features.append(mu_parts[0] if mu_parts else 0)
        features.append(nu_parts[0] if nu_parts else 0)

        # Feature 3: Last parts (smallest cycles)
        features.append(lambda_parts[-1] if lambda_parts else 0)
        features.append(mu_parts[-1] if mu_parts else 0)
        features.append(nu_parts[-1] if nu_parts else 0)

        # Feature 4: Is it the trivial representation? [n]
        features.append(1.0 if lambda_parts == [n] else 0.0)
        features.append(1.0 if mu_parts == [n] else 0.0)
        features.append(1.0 if nu_parts == [n] else 0.0)

        # Feature 5: Is it the sign representation? [1^n]
        features.append(1.0 if lambda_parts == [1]*n else 0.0)
        features.append(1.0 if mu_parts == [1]*n else 0.0)
        features.append(1.0 if nu_parts == [1]*n else 0.0)

        # Feature 6: Hook lengths (approximation via formula)
        # d_λ = n! / ∏ h_ij
        lambda_dim = self._dimension_via_hook_length(lambda_parts, n)
        mu_dim = self._dimension_via_hook_length(mu_parts, n)
        nu_dim = self._dimension_via_hook_length(nu_parts, n)

        features.append(lambda_dim)
        features.append(mu_dim)
        features.append(nu_dim)

        # Feature 7: Products of dimensions (dimension formula clue!)
        # Σ_ν g^ν_{λμ} d_ν = d_λ · d_μ
        features.append(lambda_dim * mu_dim)
        features.append(lambda_dim * nu_dim)
        features.append(mu_dim * nu_dim)

        # Feature 8: Partition equality checks
        features.append(1.0 if lambda_parts == mu_parts else 0.0)
        features.append(1.0 if lambda_parts == nu_parts else 0.0)
        features.append(1.0 if mu_parts == nu_parts else 0.0)

        # Feature 9: Conjugate partitions
        lambda_conj = self._conjugate_partition(lambda_parts, n)
        mu_conj = self._conjugate_partition(mu_parts, n)
        nu_conj = self._conjugate_partition(nu_parts, n)

        # Check if any are conjugates
        features.append(1.0 if mu_parts == lambda_conj else 0.0)
        features.append(1.0 if nu_parts == lambda_conj else 0.0)
        features.append(1.0 if nu_parts == mu_conj else 0.0)

        # Feature 10: Sum of parts (should always be n, but good for learning)
        features.append(sum(lambda_parts))
        features.append(sum(mu_parts))
        features.append(sum(nu_parts))

        # Feature 11: n itself (allows cross-S_n learning)
        features.append(n)
        features.append(n ** 2)
        features.append(np.log(n))

        # Feature 12: Partition "shapes" - differences between consecutive parts
        lambda_shape = self._partition_shape(lambda_parts)
        mu_shape = self._partition_shape(mu_parts)
        nu_shape = self._partition_shape(nu_parts)

        features.extend(lambda_shape[:5])  # First 5 differences
        features.extend(mu_shape[:5])
        features.extend(nu_shape[:5])

        return np.array(features, dtype=float)

    def _dimension_via_hook_length(self, parts: List[int], n: int) -> float:
        """
        Compute dimension of S_n irrep via hook length formula.

        d_λ = n! / ∏_{(i,j) ∈ λ} h(i,j)

        where h(i,j) = λ_i + λ_j' - i - j + 1
        """
        if not parts:
            return 1.0

        # Build Young diagram
        lambda_conj = self._conjugate_partition(parts, n)

        # Compute hook lengths
        hook_product = 1
        for i, part in enumerate(parts):
            for j in range(part):
                hook = part - j + lambda_conj[j] - i - 1
                hook_product *= hook

        # n! / hook_product
        n_factorial = 1
        for i in range(2, n + 1):
            n_factorial *= i

        return n_factorial / hook_product

    def _conjugate_partition(self, parts: List[int], n: int) -> List[int]:
        """
        Conjugate (transpose) a partition.

        Example: [4, 2, 1] → [3, 2, 1, 1]
        """
        if not parts:
            return []

        max_part = max(parts)
        conjugate = []

        for i in range(1, max_part + 1):
            # Count how many parts >= i
            count = sum(1 for p in parts if p >= i)
            if count > 0:
                conjugate.append(count)

        return conjugate

    def _partition_shape(self, parts: List[int]) -> List[float]:
        """
        Partition "shape" = differences between consecutive parts.

        Example: [5, 3, 1] → [2, 2, 1]
        """
        if len(parts) <= 1:
            return [0.0] * 10

        shape = []
        for i in range(len(parts) - 1):
            shape.append(float(parts[i] - parts[i+1]))

        # Pad to length 10
        while len(shape) < 10:
            shape.append(0.0)

        return shape[:10]

    def prepare_training_data(self):
        """Extract features for all examples."""
        if self.verbose:
            print("\nExtracting features from examples...")

        X = []
        y = []

        for i, example in enumerate(self.examples):
            features = self.extract_features(example)
            example.features = features
            X.append(features)
            y.append(example.coefficient)

            if self.verbose and (i + 1) % 200 == 0:
                print(f"  Processed {i + 1}/{len(self.examples)} examples")

        X = np.array(X)
        y = np.array(y)

        if self.verbose:
            print(f"\nFeature matrix: {X.shape}")
            print(f"Target vector: {y.shape}")
            print(f"\nFeature statistics:")
            print(f"  Mean: {X.mean(axis=0)[:5]}...")
            print(f"  Std:  {X.std(axis=0)[:5]}...")
            print(f"\nTarget statistics:")
            print(f"  Mean: {y.mean():.2f}")
            print(f"  Std:  {y.std():.2f}")
            print(f"  Max:  {y.max()}")

        return X, y

    def train(self, X, y, regularization=1e-6):
        """
        Train via closed-form least squares.

        This is the sheaf-wreath learning: no gradients, just algebra!
        """
        if self.verbose:
            print("\n" + "=" * 70)
            print("Training Sheaf-Wreath Learner (Closed-Form)")
            print("=" * 70)

        # Normalize features
        self.feature_mean = X.mean(axis=0)
        self.feature_std = X.std(axis=0) + 1e-10
        X_normalized = (X - self.feature_mean) / self.feature_std

        # Add bias term
        X_with_bias = np.column_stack([np.ones(len(X)), X_normalized])

        # Closed-form solution: w = (X^T X + λI)^{-1} X^T y
        XTX = X_with_bias.T @ X_with_bias
        XTy = X_with_bias.T @ y

        # Ridge regularization
        reg_matrix = regularization * np.eye(XTX.shape[0])
        reg_matrix[0, 0] = 0  # Don't regularize bias

        self.weights = np.linalg.solve(XTX + reg_matrix, XTy)

        # Training error
        y_pred = X_with_bias @ self.weights
        train_error = np.mean((y - y_pred) ** 2)
        train_mae = np.mean(np.abs(y - y_pred))
        r2 = 1 - train_error / (np.var(y) + 1e-10)

        if self.verbose:
            print(f"\nTraining complete!")
            print(f"  Weights learned: {len(self.weights)}")
            print(f"  MSE: {train_error:.4f}")
            print(f"  MAE: {train_mae:.4f}")
            print(f"  R²:  {r2:.4f}")

        return train_error

    def predict(self, features: np.ndarray) -> float:
        """Predict Kronecker coefficient from features."""
        if self.weights is None:
            raise ValueError("Model not trained yet!")

        # Normalize
        features_normalized = (features - self.feature_mean) / self.feature_std

        # Add bias
        features_with_bias = np.concatenate([[1], features_normalized])

        # Predict
        prediction = features_with_bias @ self.weights

        # Kronecker coefficients are non-negative integers
        return max(0, int(np.round(prediction)))

    def cross_validate(self, X, y, n_folds=5):
        """K-fold cross-validation to assess generalization."""
        if self.verbose:
            print("\n" + "=" * 70)
            print(f"Cross-Validation ({n_folds} folds)")
            print("=" * 70)

        n_samples = len(X)
        indices = np.random.permutation(n_samples)
        fold_size = n_samples // n_folds

        fold_errors = []
        fold_maes = []

        for fold in range(n_folds):
            # Split data
            test_idx = indices[fold * fold_size:(fold + 1) * fold_size]
            train_idx = np.concatenate([
                indices[:fold * fold_size],
                indices[(fold + 1) * fold_size:]
            ])

            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            # Train on fold
            feature_mean = X_train.mean(axis=0)
            feature_std = X_train.std(axis=0) + 1e-10

            X_train_norm = (X_train - feature_mean) / feature_std
            X_test_norm = (X_test - feature_mean) / feature_std

            X_train_bias = np.column_stack([np.ones(len(X_train)), X_train_norm])
            X_test_bias = np.column_stack([np.ones(len(X_test)), X_test_norm])

            # Solve
            XTX = X_train_bias.T @ X_train_bias
            XTy = X_train_bias.T @ y_train
            reg_matrix = 1e-6 * np.eye(XTX.shape[0])
            reg_matrix[0, 0] = 0

            weights = np.linalg.solve(XTX + reg_matrix, XTy)

            # Test
            y_pred = X_test_bias @ weights
            test_error = np.mean((y_test - y_pred) ** 2)
            test_mae = np.mean(np.abs(y_test - y_pred))

            fold_errors.append(test_error)
            fold_maes.append(test_mae)

        mean_error = np.mean(fold_errors)
        std_error = np.std(fold_errors)
        mean_mae = np.mean(fold_maes)

        if self.verbose:
            print(f"\nCross-validation results:")
            print(f"  Mean MSE: {mean_error:.4f} ± {std_error:.4f}")
            print(f"  Mean MAE: {mean_mae:.4f}")
            print("=" * 70)

        return mean_error, std_error


def analyze_predictions(learner, X, y, examples):
    """Analyze what the learner discovered."""
    print("\n" + "=" * 70)
    print("Analysis: What Did the Learner Discover?")
    print("=" * 70)

    # Normalize and predict
    X_norm = (X - learner.feature_mean) / learner.feature_std
    X_bias = np.column_stack([np.ones(len(X)), X_norm])
    y_pred = X_bias @ learner.weights

    # Errors
    errors = np.abs(y - y_pred)

    print(f"\nPrediction quality:")
    print(f"  Perfect (error < 0.5): {np.sum(errors < 0.5)}/{len(errors)} ({100*np.sum(errors < 0.5)/len(errors):.1f}%)")
    print(f"  Close (error < 1.0):   {np.sum(errors < 1.0)}/{len(errors)} ({100*np.sum(errors < 1.0)/len(errors):.1f}%)")
    print(f"  Good (error < 5.0):    {np.sum(errors < 5.0)}/{len(errors)} ({100*np.sum(errors < 5.0)/len(errors):.1f}%)")

    # Worst predictions
    print(f"\nWorst 10 predictions:")
    worst_idx = np.argsort(errors)[-10:][::-1]

    for idx in worst_idx:
        ex = examples[idx]
        print(f"  {ex.lambda_partition} ⊗ {ex.mu_partition} → {ex.nu_partition}:")
        print(f"    True: {int(y[idx])}, Predicted: {y_pred[idx]:.1f}, Error: {errors[idx]:.1f}")

    # Feature importance (weight magnitudes)
    print(f"\nTop 10 features by |weight|:")
    feature_importance = np.abs(learner.weights[1:])  # Skip bias
    top_features = np.argsort(feature_importance)[-10:][::-1]

    feature_names = [
        "len(λ)", "len(μ)", "len(ν)",
        "λ[0]", "μ[0]", "ν[0]",
        "λ[-1]", "μ[-1]", "ν[-1]",
        "λ==[n]", "μ==[n]", "ν==[n]",
        "λ==[1^n]", "μ==[1^n]", "ν==[1^n]",
        "d_λ", "d_μ", "d_ν",
        "d_λ·d_μ", "d_λ·d_ν", "d_μ·d_ν",
        # ... more features
    ]

    for rank, feat_idx in enumerate(top_features, 1):
        feat_name = feature_names[feat_idx] if feat_idx < len(feature_names) else f"Feature {feat_idx}"
        print(f"  {rank}. {feat_name}: weight = {learner.weights[feat_idx + 1]:.4f}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    print("\n" + "🚀" * 35)
    print("LEARNING KRONECKER COEFFICIENTS VIA SHEAF BOOTSTRAP")
    print("🚀" * 35)

    print("\nThe recursive meta-pattern:")
    print("  Level 1: C_k → S_n (Galois connection) ✓")
    print("  Level 2: S₃-S₆ examples → Predictions (sheaf learning)")
    print("\nCan the learner discover representation theory from data alone?")

    # Create learner
    learner = KroneckerLearner(verbose=True)

    # Load known coefficients
    learner.load_known_coefficients([4])

    if len(learner.examples) == 0:
        print("\n⚠️  No training data found!")
        print("Run the following to generate data:")
        print("  python3 compute_s5_s6.py")
        print("  python3 compute_s5_s6.py s6")
    else:
        # Extract features
        X, y = learner.prepare_training_data()

        # Cross-validate
        cv_error, cv_std = learner.cross_validate(X, y, n_folds=5)

        # Train on full dataset
        learner.train(X, y)

        # Analyze
        analyze_predictions(learner, X, y, learner.examples)

        print("\n" + "=" * 70)
        print("Next: Can we predict S_7 coefficients without the character table?")
        print("=" * 70)
        print("\nThe bootstrap is recursive.")
        print("The meta-knowledge enhancement is real.")
        print("\n✨ What a way to continue the masterpiece! ✨")
