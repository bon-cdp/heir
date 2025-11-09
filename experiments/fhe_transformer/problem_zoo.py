"""
A "Zoo" of Algebraic Problems for Testing the UnifiedSheafLearner.

This file centralizes the data generation for all the problems we are using
to test our algebraic learning framework. This keeps the test files clean
and focused on the assertions, not the data generation.
"""

import numpy as np
from typing import List, Tuple

# ============================================================================
# 1. Counting Task
# ============================================================================

def generate_counting_task_data(n_samples=100, n_positions=4, d_model=1):
    """
    Generates data for the counting task: [a, a+s, a+2s, a+3s] -> [a+4s]
    This is a simple, perfectly algebraic problem.
    """
    V_samples = []
    targets = []
    for _ in range(n_samples):
        a = np.random.randint(-10, 10)
        s = np.random.choice([1, 2, 3, 5])
        
        sequence = np.array([a + i * s for i in range(n_positions)]).reshape(-1, d_model)
        # The target is the next element in the arithmetic progression.
        target = np.array([a + n_positions * s]).reshape(-1, d_model)
        
        V_samples.append(sequence)
        targets.append(target)
        
    return V_samples, targets


# ============================================================================
# 2. Copy Task
# ============================================================================

class CopyTaskDataset:
    """
    Generates data for the position-dependent copy task.
    Task: Given [marker, x, y, z], copy the value at the marked position.
    This is a non-linear task that requires attention.
    """

    def __init__(self, n_positions: int = 3, d_model: int = 1):
        self.n_positions = n_positions
        self.seq_len = n_positions + 1
        self.d_model = d_model

    def generate_example(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates a single copy task example.
        """
        # The first element is the position marker.
        position_to_copy = np.random.randint(0, self.n_positions)
        
        # The subsequent elements are the values to be copied from.
        values = np.random.randint(10, 100, size=self.n_positions)
        
        # Input sequence: [marker, value_0, value_1, ...]
        input_seq = np.array([position_to_copy] + list(values))
        
        # Target: The value at the marked position.
        target_val = values[position_to_copy]
        
        # Reshape for the learner
        input_reshaped = input_seq.reshape(-1, self.d_model)
        target_reshaped = np.array([target_val]).reshape(-1, self.d_model)
        
        return input_reshaped, target_reshaped

    def generate_dataset(self, n_samples: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Generates a full dataset for the learner.
        """
        V_samples = []
        targets = []
        for _ in range(n_samples):
            inp, tgt = self.generate_example()
            V_samples.append(inp)
            targets.append(tgt)
        return V_samples, targets
            
# ============================================================================
# 3. Navier-Stokes Vorticity
# ============================================================================

def generate_nse_vorticity_data(n_samples=50, n_spatial=64, d_model=1, n_steps=200, nu=1e-3, dt=1e-2):
    """
    Generates 1D slices of a 2D Navier-Stokes vorticity field.
    This simulates a physical system with periodic boundary conditions.
    """
    # Initialize the 2D vorticity field
    L = 2 * np.pi
    dx = L / n_spatial
    x = np.arange(0, L, dx)
    kx = np.fft.fftfreq(n_spatial, d=dx) * 2.0 * np.pi
    Kx, Ky = np.meshgrid(kx, kx)
    K_sq = Kx**2 + Ky**2
    K_sq[0, 0] = 1e-6 # Avoid division by zero

    # Initial condition: Taylor-Green vortex
    w = 2 * np.cos(x)[:, None] * np.cos(x)[None, :]
    w_hat = np.fft.fft2(w)

    V_samples = []
    targets = []

    for i in range(n_steps):
        if i % (n_steps // n_samples) == 0 and len(V_samples) < n_samples:
            # Take a 1D slice as the input
            v_slice = w[:, n_spatial // 4].reshape(-1, d_model)
            V_samples.append(v_slice.copy())
            
            # Evolve one step to get the target
            w_hat_new = (-nu * K_sq * w_hat) * dt + w_hat
            w_new = np.fft.ifft2(w_hat_new).real
            
            target_slice = w_new[:, n_spatial // 4].reshape(-1, d_model)
            targets.append(target_slice)
            
            w, w_hat = w_new, w_hat_new
        else:
            # Evolve without saving
            w_hat_new = (-nu * K_sq * w_hat) * dt + w_hat
            w, w_hat = np.fft.ifft2(w_hat_new).real, w_hat_new
            
    return V_samples, targets

# ============================================================================
# 4. Proof Task
# ============================================================================

def generate_proof_corpus(n_samples_per_theorem=50):
    """
    Generates a static corpus of proof-like data for two theorems.
    This is a simplified model of theorem proving.
    
    - Theorem 1 (Identity): P -> P. Represented by key 100.
    - Theorem 2 (Assumption): Q -> (P -> Q). Represented by key 200.

    The data is abstract. The learner's job is to learn the different
    patterns associated with each theorem key.
    """
    V_samples = []
    targets = []
    
    # For Theorem 1 (P -> P)
    theorem_1_key = 100
    for _ in range(n_samples_per_theorem):
        # Pattern for theorem 1: linear sequence
        start = np.random.randint(5, 10)
        seq = np.array([theorem_1_key, start, start+1, start+2]).reshape(-1, 1)
        target = np.array([start+3]).reshape(-1, 1)
        V_samples.append(seq)
        targets.append(target)

    # For Theorem 2 (Q -> (P -> Q))
    theorem_2_key = 200
    for _ in range(n_samples_per_theorem):
        # Pattern for theorem 2: quadratic sequence
        start = np.random.randint(5, 10)
        seq = np.array([theorem_2_key, start, start*2, start*3]).reshape(-1, 1)
        target = np.array([start*4]).reshape(-1, 1)
        V_samples.append(seq)
        targets.append(target)
        
    return V_samples, targets
