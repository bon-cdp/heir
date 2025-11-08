"""Quick robustness tests for wreath product vorticity"""
import numpy as np
from navier_stokes_data import generate_training_data
from vorticity_wreath import VorticityWreathAttention

print("="*70)
print("WREATH PRODUCT VORTICITY: ROBUSTNESS TESTS")
print("="*70)

# Generate data
history, _ = generate_training_data(N=64, nu=0.001, dt=0.01, n_steps=200, save_every=2)

model = VorticityWreathAttention(n_spatial=64, n_characters=8, verbose=False)

# Prepare ALL data
inputs, targets = model.prepare_training_data(history, y_slice=32)

# TEST 1: Held-out validation
print("\n1. HELD-OUT VALIDATION TEST")
print("-"*70)
model.fit(inputs[:60], targets[:60])

# Test on truly unseen data
test_init = inputs[80]
test_true = targets[80:85]
pred_rollout = model.rollout(test_init, n_steps=5)

errors = [np.mean((test_true[i] - pred_rollout[i+1])**2) for i in range(len(test_true))]
print(f"Held-out MSE: {np.mean(errors):.6e}")
print("✓ PASS" if np.mean(errors) < 1e-10 else "✗ FAIL")

# TEST 2: Long rollout
print("\n2. LONG ROLLOUT STABILITY TEST")
print("-"*70)
long_rollout = model.rollout(inputs[60], n_steps=20)
energies = [np.sum(omega**2) for omega in long_rollout]
print(f"Energy at t=0: {energies[0]:.6f}")
print(f"Energy at t=20: {energies[-1]:.6f}")
print(f"Energy decay: {(energies[0]-energies[-1])/energies[0]*100:.2f}%")
print("✓ PASS (stable)" if not np.isnan(energies[-1]) else "✗ FAIL (blowup)")

# TEST 3: Random turbulent IC
print("\n3. TURBULENT INITIAL CONDITION TEST")
print("-"*70)
from navier_stokes_data import NavierStokes2D
solver = NavierStokes2D(N=64)
omega_turb = solver.random_vorticity(k_max=20, alpha=2.0)
omega_turb_1d = omega_turb[32, :].reshape(-1, 1)

pred_turb = model.predict(omega_turb_1d)
print(f"Turbulent prediction norm: {np.linalg.norm(pred_turb):.6f}")
print("✓ PASS (no NaN)" if not np.isnan(pred_turb).any() else "✗ FAIL (NaN)")

print("\n" + "="*70)
print("✅ ALL TESTS COMPLETE")
print("="*70)
