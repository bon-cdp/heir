"""
Minimal Wreath Product Attention for 2D Vorticity

Pure implementation - just character theory + least squares + physics.

NO symbolic extraction.
NO polynomial basis.
NO comparisons to SINDy.

Just: Can wreath products capture vorticity dynamics?

Approach:
=========
1. Extract 1D slices of vorticity: ω(x) for fixed y
2. Learn evolution ω(x,t) → ω(x,t+dt) using wreath product
3. Position = x location on grid
4. Characters = Fourier modes
5. Evaluate: Does it capture physics (energy spectrum, conservation)?

Why 1D first?
============
- Simpler to understand
- Still captures multi-scale dynamics
- If it works, extend to 2D
- Keeps code pure and minimal

Mathematical Structure:
======================
Vorticity field: ω(x,t)

Fourier decomposition:
    ω(x,t) = Σ_k ω̂_k(t) e^{ikx}

Wreath product view:
    At position x_p: different Fourier modes k have different weights
    This captures spatial variation in spectral content!

Physical Meaning:
================
- Large eddies (small k) dominate in some regions
- Small eddies (large k) dominate in others
- Wreath product learns this position-dependent spectrum
- This IS turbulence structure!
"""

import numpy as np
from typing import List, Tuple
from wreath_product_groups import WreathProductGroup, WreathProductLearner
from navier_stokes_data import generate_training_data


class VorticityWreathAttention:
    """
    Minimal wreath product attention for vorticity evolution.

    Learn: ω(x,t) → ω(x,t+dt)

    Using pure character theory + closed-form least squares.
    """

    def __init__(
        self,
        n_spatial: int,
        n_characters: int = 8,
        verbose: bool = True
    ):
        """
        Initialize vorticity wreath attention.

        Args:
            n_spatial: Number of spatial points (grid size)
            n_characters: Number of Fourier modes to use
            verbose: Print progress
        """
        self.n_spatial = n_spatial
        self.n_characters = n_characters
        self.verbose = verbose

        # Wreath product: C_k ≀ C_n
        # k = number of Fourier modes
        # n = number of spatial positions
        self.wreath = WreathProductGroup(
            base_order=n_characters,
            top_order=n_spatial
        )

        # Learner
        # d_model = spatial dimension (we predict entire profile)
        self.learner = WreathProductLearner(
            wreath_group=self.wreath,
            d_model=n_spatial
        )

        if verbose:
            print(f"\nVorticityWreathAttention:")
            print(f"  Spatial points: {n_spatial}")
            print(f"  Fourier modes: {n_characters}")
            print(f"  Parameters: {n_spatial * n_characters}")

    def prepare_training_data(
        self,
        vorticity_snapshots: List[np.ndarray],
        y_slice: int = None
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Prepare training pairs from vorticity snapshots.

        Args:
            vorticity_snapshots: List of 2D vorticity fields
            y_slice: Which y-slice to use (default: middle)

        Returns:
            (inputs, targets) where inputs[i] → targets[i]
        """
        if y_slice is None:
            y_slice = vorticity_snapshots[0].shape[0] // 2

        inputs = []
        targets = []

        # Create pairs: ω(t) → ω(t+1)
        for i in range(len(vorticity_snapshots) - 1):
            omega_t = vorticity_snapshots[i]
            omega_t1 = vorticity_snapshots[i + 1]

            # Extract 1D slice at y_slice
            omega_1d_t = omega_t[y_slice, :]  # [N]
            omega_1d_t1 = omega_t1[y_slice, :]  # [N]

            # Reshape for wreath product: [N, 1]
            input_seq = omega_1d_t.reshape(-1, 1)
            target_val = omega_1d_t1.reshape(-1, 1)  # Actually predict full next profile

            inputs.append(input_seq)
            targets.append(target_val)

        if self.verbose:
            print(f"\nPrepared training data:")
            print(f"  Pairs: {len(inputs)}")
            print(f"  Input shape: {inputs[0].shape}")
            print(f"  Target shape: {targets[0].shape}")
            print(f"  Using y-slice: {y_slice}")

        return inputs, targets

    def fit(
        self,
        inputs: List[np.ndarray],
        targets: List[np.ndarray]
    ):
        """
        Learn vorticity evolution via wreath product.

        Pure closed-form least squares - no iteration!

        Args:
            inputs: List of input vorticity profiles
            targets: List of target vorticity profiles (we'll predict last point)
        """
        if self.verbose:
            print("\n" + "="*70)
            print("Learning Vorticity Evolution (Wreath Product)")
            print("="*70)

        # For vorticity, we have single condition (no conditional routing)
        # All samples belong to condition 0
        conditions = [0] * len(inputs)

        # Predict entire spatial profile
        # Learner expects [d_model] = [N] outputs
        targets_flat = [t.flatten() for t in targets]  # [N] for each sample

        # Learn position-dependent character weights
        self.learner.fit(
            V_samples=inputs,
            targets=targets_flat,
            conditions=conditions,
            verbose=self.verbose
        )

        if self.verbose:
            print("\n✅ Learning complete!")

        return self

    def predict(
        self,
        omega_input: np.ndarray
    ) -> np.ndarray:
        """
        Predict next vorticity state.

        Args:
            omega_input: Current vorticity [N, 1]

        Returns:
            Predicted next vorticity [N, 1]
        """
        # Use condition 0 (single physics regime)
        pred_flat = self.learner.predict(omega_input, condition=0)

        # Reshape back
        return pred_flat.reshape(-1, 1)

    def rollout(
        self,
        omega_init: np.ndarray,
        n_steps: int
    ) -> List[np.ndarray]:
        """
        Autoregressive rollout: predict multiple timesteps.

        Args:
            omega_init: Initial vorticity [N, 1]
            n_steps: Number of steps to predict

        Returns:
            List of predicted vorticity profiles
        """
        history = [omega_init.copy()]
        omega = omega_init.copy()

        for _ in range(n_steps):
            omega = self.predict(omega)
            history.append(omega.copy())

        return history

    def evaluate_physics(
        self,
        omega_true: List[np.ndarray],
        omega_pred: List[np.ndarray]
    ):
        """
        Evaluate if learned dynamics preserve physics.

        Check:
        1. Energy conservation (should decay due to viscosity)
        2. Spectral characteristics
        3. Long-term stability

        Args:
            omega_true: True vorticity evolution
            omega_pred: Predicted vorticity evolution
        """
        print("\n" + "="*70)
        print("Physics Evaluation")
        print("="*70)

        # 1. Energy evolution
        energy_true = [np.sum(omega**2) for omega in omega_true]
        energy_pred = [np.sum(omega**2) for omega in omega_pred]

        print("\nEnergy (enstrophy) evolution:")
        print("  True energy decay:")
        for i in [0, len(energy_true)//2, -1]:
            print(f"    t={i}: E = {energy_true[i]:.6f}")

        print("  Predicted energy:")
        for i in [0, min(len(energy_pred)//2, len(energy_pred)-1), min(len(energy_pred)-1, len(energy_pred)-1)]:
            if i < len(energy_pred):
                print(f"    t={i}: E = {energy_pred[i]:.6f}")

        # 2. Pointwise error
        n_compare = min(len(omega_true), len(omega_pred))
        errors = []
        for i in range(n_compare):
            err = np.mean((omega_true[i] - omega_pred[i])**2)
            errors.append(err)

        print(f"\nMean Squared Error:")
        print(f"  Initial: {errors[0]:.6e}")
        if len(errors) > 1:
            print(f"  Final: {errors[-1]:.6e}")
        print(f"  Average: {np.mean(errors):.6e}")

        # 3. Spectral content
        omega_true_fft = np.fft.fft(omega_true[0].flatten())
        omega_pred_fft = np.fft.fft(omega_pred[0].flatten())

        spectrum_true = np.abs(omega_true_fft)**2
        spectrum_pred = np.abs(omega_pred_fft)**2

        print(f"\nSpectral similarity:")
        print(f"  Correlation: {np.corrcoef(spectrum_true, spectrum_pred)[0,1]:.6f}")

        print("="*70)


def main():
    """
    Minimal experiment: Can wreath products learn vorticity evolution?

    No benchmarks. No comparisons. Just: does it work?
    """
    print("\n" + "="*70)
    print("WREATH PRODUCT ATTENTION FOR NAVIER-STOKES")
    print("="*70)
    print("\nPhilosophy: Pure mathematics. No benchmarks. Just see what happens.")

    # 1. Generate vorticity data
    print("\n" + "-"*70)
    print("Step 1: Generate Navier-Stokes Data")
    print("-"*70)

    history, solver = generate_training_data(
        N=64,
        nu=0.001,
        dt=0.01,
        n_steps=100,
        save_every=2
    )

    # 2. Create wreath product model
    print("\n" + "-"*70)
    print("Step 2: Initialize Wreath Product Attention")
    print("-"*70)

    model = VorticityWreathAttention(
        n_spatial=64,
        n_characters=8,
        verbose=True
    )

    # 3. Prepare training data
    inputs, targets = model.prepare_training_data(
        history[:-10],  # Hold out last 10 for testing
        y_slice=32  # Middle slice
    )

    # 4. Learn
    print("\n" + "-"*70)
    print("Step 3: Learn Vorticity Dynamics")
    print("-"*70)

    model.fit(inputs[:40], targets[:40])  # Use subset for training

    # 5. Test: Short rollout
    print("\n" + "-"*70)
    print("Step 4: Test Predictions")
    print("-"*70)

    # Use last training sample for testing
    test_init = inputs[35]
    test_true = [t for t in targets[35:40]]

    pred_rollout = model.rollout(test_init, n_steps=5)

    # 6. Evaluate physics
    model.evaluate_physics(
        omega_true=[test_init] + test_true,
        omega_pred=pred_rollout
    )

    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print("\nKey Question: Did wreath products capture the physics?")
    print("\nLook for:")
    print("  ✓ Energy decay (viscous dissipation)")
    print("  ✓ Spectral similarity (Fourier modes preserved)")
    print("  ✓ Stable rollout (no blow-up)")
    print("\nIf yes: Beautiful connection between character theory and fluid dynamics!")
    print("If no: Try sheaves instead (local patches + gluing)")
    print("="*70)


if __name__ == "__main__":
    main()
