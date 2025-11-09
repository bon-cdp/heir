"""
A Unified Algebraic Learner Based on Sheaf Theory and Wreath Products.

This model solves a diverse set of problems by constructing a single, global
linear system that enforces both local accuracy and global consistency.

The core idea is to represent a problem as a "sheaf" of local learning tasks
(the "patches") connected by linear "gluing" constraints on their overlaps.

The resulting global linear system is solved in a single, closed-form step,
yielding both the optimal parameters and a measure of the "cohomological
obstruction" (the residual error), which quantifies the problem's inherent
unlearnability.
"""

import numpy as np
from scipy.linalg import block_diag

# We will reuse the character theory from our previous work.
from character_theory_attention import CyclicGroupCharacters

class UnifiedSheafLearner:
    """
    Learns by solving a single global linear system representing a sheaf.
    """
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.solution = None
        self.residual_error = None
        self.problem_definition = None

    def fit(self, problem_definition):
        """
        Fits the model to a problem defined as a sheaf of learning tasks.

        Args:
            problem_definition (dict): A dictionary describing the sheaf structure.
                {
                    'patches': {
                        'patch_name_1': {
                            'data': (V_samples, targets),
                            'config': {'n_characters': k, 'd_model': d}
                        }, ...
                    },
                    'gluings': [
                        {
                            'patch_1': 'patch_name_1',
                            'patch_2': 'patch_name_2',
                            'constraint_points': (points_1, points_2),
                        }, ...
                    ]
                }
        """
        self.problem_definition = problem_definition
        if self.verbose:
            print("="*80)
            print("Fitting Unified Sheaf Learner")
            print(f"Found {len(problem_definition['patches'])} patches.")
            print(f"Found {len(problem_definition['gluings'])} gluing constraints.")
            print("="*80)

        # 1. Build the block-diagonal matrix for local data fitting
        local_matrices, local_targets, patch_info = self._build_local_systems()
        A_local = block_diag(*local_matrices)
        b_local = np.concatenate(local_targets)

        # 2. Build the matrix for global gluing constraints
        A_gluing, b_gluing = self._build_gluing_systems(patch_info)

        # 3. Assemble the final global system
        # A_sheaf has shape (total_data_points + total_constraint_points, total_weights)
        A_sheaf = np.vstack([A_local, A_gluing])
        b_sheaf = np.concatenate([b_local, b_gluing])

        if self.verbose:
            print(f"\nAssembled Global System 'A_sheaf':")
            print(f"  - Shape: {A_sheaf.shape}")
            print(f"  - Local data rows (accuracy): {A_local.shape[0]}")
            print(f"  - Gluing rows (consistency): {A_gluing.shape[0]}")

        # 4. Solve the single global least-squares problem
        try:
            # Using Ridge regression for stability, as the matrix can be ill-conditioned
            lambda_ridge = 1e-8
            A_H_A = A_sheaf.T.conj() @ A_sheaf + lambda_ridge * np.eye(A_sheaf.shape[1])
            A_H_b = A_sheaf.T.conj() @ b_sheaf
            
            w_solution = np.linalg.solve(A_H_A, A_H_b)
            
            residuals = np.linalg.norm(A_sheaf @ w_solution - b_sheaf)**2
            
            self.solution = self._unpack_solution(w_solution, patch_info)
            self.residual_error = residuals if residuals > 1e-12 else 0.0

            if self.verbose:
                print(f"\nGlobal System Solved:")
                print(f"  - Total weights learned: {len(w_solution)}")
                print(f"  - Final Residual (Obstruction): {self.residual_error:.4e}")

        except np.linalg.LinAlgError as e:
            print(f"\nERROR: The global system is singular and could not be solved. {e}")
            self.solution = None
            self.residual_error = float('inf')

        return self.solution, self.residual_error

    def _get_feature_row(self, V, config):
        """
        Computes a single row of the design matrix for a given sample V.
        This represents the features for the wreath product attention model.
        """
        n_positions = config['n_positions']
        n_characters = config['n_characters']
        
        group = CyclicGroupCharacters(n=n_positions)
        projs = group.decompose_into_characters(V)
        
        feature_row = np.zeros(n_positions * n_characters, dtype=complex)
        for p in range(n_positions):
            for j in range(n_characters):
                if j >= len(projs): continue
                col_idx = p * n_characters + j
                scalar_feature = projs[j][p, 0]
                feature_row[col_idx] = scalar_feature
        return feature_row

    def _build_local_systems(self):
        """
        Constructs the block-diagonal part of the system for local accuracy.
        """
        local_matrices = []
        local_targets = []
        patch_info = {}
        current_col_offset = 0

        if self.verbose:
            print("\nBuilding Local Systems (Patches):")

        for name, patch in self.problem_definition['patches'].items():
            V_samples, targets = patch['data']
            config = patch['config']
            d_model = config['d_model']
            n_samples = len(V_samples)
            n_weights_for_patch = config['n_positions'] * config['n_characters']
            
            if d_model != 1:
                raise NotImplementedError("d_model > 1 not yet supported.")

            A_patch = np.zeros((n_samples, n_weights_for_patch), dtype=complex)
            b_patch = np.zeros(n_samples, dtype=complex)

            for i in range(n_samples):
                A_patch[i, :] = self._get_feature_row(V_samples[i], config)
                b_patch[i] = targets[i].flatten()[0]

            if self.verbose:
                print(f"  - Patch '{name}': A_patch shape {A_patch.shape}, b_patch shape {b_patch.shape}")

            local_matrices.append(A_patch)
            local_targets.append(b_patch)
            
            patch_info[name] = {
                'col_offset': current_col_offset,
                'n_weights': n_weights_for_patch,
                'config': config
            }
            current_col_offset += n_weights_for_patch
            
        return local_matrices, local_targets, patch_info

    def _build_gluing_systems(self, patch_info):
        """
        Constructs the constraint part of the system for global consistency.
        """
        gluing_matrices = []
        total_weights = sum(info['n_weights'] for info in patch_info.values())

        if self.verbose and self.problem_definition['gluings']:
            print("\nBuilding Gluing Systems (Constraints):")

        for i, gluing in enumerate(self.problem_definition['gluings']):
            p1_name = gluing['patch_1']
            p2_name = gluing['patch_2']
            
            config1 = patch_info[p1_name]['config']
            config2 = patch_info[p2_name]['config']

            # We need data points to build the feature rows for the constraint.
            constraint_V1 = gluing['constraint_data_1']
            constraint_V2 = gluing['constraint_data_2']

            # This logic assumes we are constraining a single point prediction.
            # A_constraint_row will enforce the constraint.
            A_constraint_row = np.zeros((1, total_weights), dtype=complex)
            
            # Feature row for the first patch
            feature_row1 = self._get_feature_row(constraint_V1, config1)
            offset1 = patch_info[p1_name]['col_offset']
            n_weights1 = patch_info[p1_name]['n_weights']
            A_constraint_row[0, offset1 : offset1 + n_weights1] = feature_row1

            # Feature row for the second patch (subtracting)
            feature_row2 = self._get_feature_row(constraint_V2, config2)
            offset2 = patch_info[p2_name]['col_offset']
            n_weights2 = patch_info[p2_name]['n_weights']
            A_constraint_row[0, offset2 : offset2 + n_weights2] = -feature_row2
            
            if self.verbose:
                print(f"  - Gluing {i+1} ('{p1_name}' <-> '{p2_name}'): Constraint row shape {A_constraint_row.shape}")

            gluing_matrices.append(A_constraint_row)

        if not gluing_matrices:
            return np.empty((0, total_weights)), np.empty(0)

        A_gluing = np.vstack(gluing_matrices)
        b_gluing = np.zeros(A_gluing.shape[0]) # Constraints are of the form ... = 0
        
        return A_gluing, b_gluing

    def _unpack_solution(self, w_solution, patch_info):
        """
        Unpacks the flat solution vector into a structured dictionary.
        """
        solution = {}
        for name, info in patch_info.items():
            start = info['col_offset']
            end = start + info['n_weights']
            config = info['config']
            n_positions = config.get('n_positions', -1)
            n_characters = config['n_characters']
            
            weights_flat = w_solution[start:end]
            
            # Handle case where n_positions might not be in config if patch is empty
            if n_positions != -1:
                solution[name] = {
                    'weights': weights_flat.reshape((n_positions, n_characters)),
                    'config': config
                }
            else:
                solution[name] = {
                    'weights': weights_flat,
                    'config': config
                }
        return solution
