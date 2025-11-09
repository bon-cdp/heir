"""
The Grand Finale: A Generalized Sheaf Learner

This learner automates the discovery of sheaf structure by using a 
conditioning function to partition data into patches before solving.
"""

from unified_sheaf_learner import UnifiedSheafLearner

class GeneralizedSheafLearner:
    """
    A higher-level learner that automates patch discovery.
    """
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.unified_learner = UnifiedSheafLearner(verbose=verbose)
        self.solution = None
        self.residual = None

    def fit(self, V_samples, targets, problem_config, conditioning_function):
        """
        Fits the model by first partitioning data into patches and then solving.

        Args:
            V_samples (list): List of all input samples.
            targets (list): List of all target samples.
            problem_config (dict): Config specifying n_characters, d_model, etc.
                                   and optionally gluing information.
            conditioning_function (callable): A function that takes a sample V
                                              and its target T and returns a
                                              key to identify its patch.
        """
        # 1. Conditioning Step: Partition data into patches
        patches_data = {}
        for v, t in zip(V_samples, targets):
            key = conditioning_function(v, t)
            if key not in patches_data:
                patches_data[key] = ([], [])
            patches_data[key][0].append(v)
            patches_data[key][1].append(t)

        # 2. Sheaf Construction Step: Build the problem definition
        problem_definition = {'patches': {}, 'gluings': problem_config.get('gluings', [])}
        for key, (v_patch, t_patch) in patches_data.items():
            patch_name = str(key)
            if not v_patch: continue

            problem_definition['patches'][patch_name] = {
                'data': (v_patch, t_patch),
                'config': {
                    'n_characters': problem_config['n_characters'],
                    'd_model': problem_config['d_model'],
                    'n_positions': v_patch[0].shape[0]
                }
            }
        
        # 3. Solver Step: Use the unified learner
        self.solution, self.residual = self.unified_learner.fit(problem_definition)
        
        return self.solution, self.residual

