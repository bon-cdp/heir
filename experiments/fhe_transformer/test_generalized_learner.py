"""
The Ultimate Test Suite for the GeneralizedSheafLearner.

This suite validates that a single, generalized learner can solve a diverse
set of problems by automatically discovering the underlying sheaf structure
based on a provided conditioning function.

Each test calls the *same* GeneralizedSheafLearner instance.
"""

import numpy as np
from generalized_sheaf_learner import GeneralizedSheafLearner
from problem_zoo import (
    generate_counting_task_data,
    CopyTaskDataset,
    generate_nse_vorticity_data,
    generate_proof_corpus
)

def run_all_tests():
    """Execute all tests for the generalized learner."""
    
    learner = GeneralizedSheafLearner(verbose=True)
    
    # Test 1: Simple algebraic task
    test_generalized_counting(learner)
    
    # Test 2: Conditional non-linear task
    test_generalized_copy(learner)
    
    # Test 3: Abstract theorem proving task
    test_generalized_proofs(learner)
    
    # Test 4: Physically constrained task with gluing
    # Note: This test still requires manual patch definition because the
    # partitioning is spatial, not conditional. It uses the underlying
    # UnifiedSheafLearner directly.
    test_manual_gluing_nse(learner.unified_learner)


def test_generalized_counting(learner):
    print("\n" + "="*80)
    print("GRAND TEST 1: Counting Task (Auto-Discovered Single Patch)")
    print("="*80)
    
    V_samples, targets = generate_counting_task_data(n_samples=100)
    problem_config = {'n_characters': 8, 'd_model': 1}
    conditioning_function = lambda v, t: "counting_patch"

    _, residual = learner.fit(V_samples, targets, problem_config, conditioning_function)
    assert residual < 1e-9, f"Counting task failed with obstruction {residual:.4e}"
    print(f"\n✓ PASSED: Counting Task solved with obstruction {residual:.4e}")


def test_generalized_copy(learner):
    print("\n" + "="*80)
    print("GRAND TEST 2: Copy Task (Auto-Discovered Multi-Patch)")
    print("="*80)
    
    dataset = CopyTaskDataset(n_positions=3)
    V_samples, targets = dataset.generate_dataset(n_samples=300)
    problem_config = {'n_characters': 8, 'd_model': 1}
    conditioning_function = lambda v, t: f"copy_pos_{int(v[0,0])}"

    _, residual = learner.fit(V_samples, targets, problem_config, conditioning_function)
    assert residual < 1e-9, f"Copy task failed with obstruction {residual:.4e}"
    print(f"\n✓ PASSED: Copy Task solved with obstruction {residual:.4e}")


def test_generalized_proofs(learner):
    print("\n" + "="*80)
    print("GRAND TEST 3: Proof Task (Auto-Discovered Multi-Patch)")
    print("="*80)
    
    V_samples, targets = generate_proof_corpus(n_samples_per_theorem=50)
    problem_config = {'n_characters': 8, 'd_model': 1}
    # Condition on the first element of the sequence, which is the theorem key
    conditioning_function = lambda v, t: f"theorem_{int(v[0,0])}"

    _, residual = learner.fit(V_samples, targets, problem_config, conditioning_function)
    assert residual < 1e-9, f"Proof task failed with obstruction {residual:.4e}"
    print(f"\n✓ PASSED: Proof Task solved with obstruction {residual:.4e}")


def test_manual_gluing_nse(unified_learner):
    print("\n" + "="*80)
    print("GRAND TEST 4: NSE Vorticity (Manual 2-Patch with Gluing)")
    print("="*80)
    
    n_samples, n_spatial, d_model = 50, 32, 1
    V_samples, targets = generate_nse_vorticity_data(n_samples=n_samples, n_spatial=n_spatial, d_model=d_model)

    patch_a_indices = np.arange(20)
    patch_b_indices = np.arange(12, 32)
    V_A = [v[patch_a_indices] for v in V_samples]; T_A = [v[patch_a_indices] for v in targets]
    V_B = [v[patch_b_indices] for v in V_samples]; T_B = [v[patch_b_indices] for v in targets]
    
    # For the gluing constraint, we need a data point. We'll use the first sample.
    # The constraint will be that the prediction for this data point is the same
    # from both patches' perspectives.
    constraint_data_A = V_A[0]
    constraint_data_B = V_B[0]

    problem_definition = {
        'patches': {
            'nse_left': {
                'data': (V_A, T_A),
                'config': {'n_characters': 8, 'd_model': d_model, 'n_positions': len(patch_a_indices)}
            },
            'nse_right': {
                'data': (V_B, T_B),
                'config': {'n_characters': 8, 'd_model': d_model, 'n_positions': len(patch_b_indices)}
            }
        },
        'gluings': [
            {
                'patch_1': 'nse_left',
                'patch_2': 'nse_right',
                # We provide the data needed for each patch to build its constraint row
                'constraint_data_1': constraint_data_A,
                'constraint_data_2': constraint_data_B,
            }
        ]
    }
    # We call the underlying unified_learner directly for this manual definition
    _, residual = unified_learner.fit(problem_definition)
    assert residual < 1e-5, f"NSE gluing task failed with obstruction {residual:.4e}"
    print(f"\n✓ PASSED: NSE Gluing Task solved with obstruction {residual:.4e}")


if __name__ == "__main__":
    run_all_tests()
