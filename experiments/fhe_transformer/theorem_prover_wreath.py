"""
Wreath Product Attention for Automated Theorem Proving

Learn to prove theorems via character theory!

Architecture:
- Position p = proof step
- Character j = inference rule type
- Weight w_{p,j} = "at step p, use rule j"

This IS the wreath product structure of proof search!
"""

import numpy as np
from typing import List, Tuple, Dict
from proof_system import *
from wreath_product_groups import WreathProductGroup, WreathProductLearner


class ProofEncoder:
    """
    Encode proof states as embeddings for wreath product learning.
    """

    def __init__(self, embedding_dim: int = 32):
        """
        Initialize encoder.

        Args:
            embedding_dim: Dimension of formula embeddings
        """
        self.embedding_dim = embedding_dim
        self.vocab = self._build_vocab()

    def _build_vocab(self) -> Dict[str, int]:
        """Build vocabulary of tokens."""
        return {
            '(': 0,
            ')': 1,
            '→': 2,
            'VAR': 3,  # Generic variable token
            'P': 4,
            'Q': 5,
            'R': 6,
            'S': 7,
        }

    def formula_to_tokens(self, formula: Formula) -> List[int]:
        """Convert formula to token sequence."""
        if isinstance(formula, Var):
            if formula.name in self.vocab:
                return [self.vocab[formula.name]]
            else:
                return [self.vocab['VAR']]
        elif isinstance(formula, Implies):
            return ([self.vocab['(']] +
                    self.formula_to_tokens(formula.antecedent) +
                    [self.vocab['→']] +
                    self.formula_to_tokens(formula.consequent) +
                    [self.vocab[')']])
        else:
            return []

    def encode_state(self, state: ProofState) -> np.ndarray:
        """
        Encode proof state as embedding.

        Simple approach: Hash derived formulas into fixed-size vector.

        Returns:
            Embedding [embedding_dim]
        """
        # Create embedding
        emb = np.zeros(self.embedding_dim)

        # Hash each derived formula into embedding
        for i, formula in enumerate(state.derived):
            tokens = self.formula_to_tokens(formula)
            # Simple hash: XOR token indices
            hash_val = sum(tokens) % self.embedding_dim
            emb[hash_val] += 1.0

        # Add step number
        emb[0] = float(state.step)

        # Normalize
        if np.linalg.norm(emb) > 0:
            emb = emb / np.linalg.norm(emb)

        return emb


class TheoremProverWreathAttention:
    """
    Automated theorem prover using wreath product attention.

    Learn: proof_state → next_inference_rule

    Via character theory + closed-form least squares!
    """

    def __init__(
        self,
        n_rules: int = 5,
        max_proof_steps: int = 10,
        embedding_dim: int = 32,
        n_characters: int = 4,
        verbose: bool = True
    ):
        """
        Initialize theorem prover.

        Args:
            n_rules: Number of inference rules
            max_proof_steps: Maximum proof depth
            embedding_dim: Dimension of state embeddings
            n_characters: Number of Fourier modes
            verbose: Print progress
        """
        self.n_rules = n_rules
        self.max_steps = max_proof_steps
        self.embedding_dim = embedding_dim
        self.n_characters = n_characters
        self.verbose = verbose

        # Encoder
        self.encoder = ProofEncoder(embedding_dim)

        # Wreath product: InferenceRules ≀ ProofSteps
        self.wreath = WreathProductGroup(
            base_order=n_characters,
            top_order=max_proof_steps
        )

        # Learner
        self.learner = WreathProductLearner(
            wreath_group=self.wreath,
            d_model=n_rules  # Predict which rule to apply
        )

        self.system = ProofSystem()

        if verbose:
            print(f"\nTheoremProverWreathAttention:")
            print(f"  Inference rules: {n_rules}")
            print(f"  Max proof steps: {max_proof_steps}")
            print(f"  Characters (Fourier modes): {n_characters}")
            print(f"  Parameters: {max_proof_steps * n_characters * n_rules}")

    def prepare_training_data(
        self,
        proofs: List[List[ProofState]]
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Convert proofs to training data.

        For each proof step, create:
            Input: State embedding sequence
            Target: Which rule was applied (one-hot)

        Args:
            proofs: List of proof sequences

        Returns:
            (inputs, targets)
        """
        inputs = []
        targets = []

        for proof in proofs:
            # For each step in proof (except initial state)
            for i in range(1, len(proof)):
                prev_state = proof[i-1]
                next_state = proof[i]

                # Encode states as sequence
                # Use all states up to current as context
                state_sequence = []
                for j in range(i):
                    emb = self.encoder.encode_state(proof[j])
                    state_sequence.append(emb)

                # Pad to max_steps
                while len(state_sequence) < self.max_steps:
                    state_sequence.append(np.zeros(self.embedding_dim))

                # Convert to sequence embedding
                seq_emb = np.array(state_sequence)  # [max_steps, embedding_dim]

                # Determine which rule was applied
                # For now, encode as simple one-hot
                # Rule 0: Axiom K
                # Rule 1: Axiom S
                # Rule 2: Modus Ponens
                rule_applied = self._infer_rule(prev_state, next_state)
                target = np.zeros(self.n_rules)
                if rule_applied is not None:
                    target[rule_applied] = 1.0

                # Reshape for wreath product: [max_steps, 1]
                seq_input = seq_emb[:, 0].reshape(-1, 1)  # Use first dim only for simplicity

                inputs.append(seq_input)
                targets.append(target)

        if self.verbose:
            print(f"\nPrepared training data:")
            print(f"  Proof steps: {len(inputs)}")
            print(f"  Input shape: {inputs[0].shape if inputs else 'N/A'}")
            print(f"  Target shape: {targets[0].shape if targets else 'N/A'}")

        return inputs, targets

    def _infer_rule(
        self,
        prev_state: ProofState,
        next_state: ProofState
    ) -> Optional[int]:
        """
        Infer which rule was applied between states.

        Returns:
            Rule index, or None if unknown
        """
        if len(next_state.derived) != len(prev_state.derived) + 1:
            return None

        new_formula = next_state.derived[-1]

        # Check if it's an axiom K instance
        if isinstance(new_formula, Implies):
            consequent = new_formula.consequent
            if isinstance(consequent, Implies):
                # Could be axiom K: P → (Q → P)
                return 0  # Axiom K

        # Check if it could be axiom S
        # S is more complex, skip for now

        # Otherwise assume Modus Ponens
        return 2

    def fit(
        self,
        proofs: List[List[ProofState]]
    ):
        """
        Learn from proof examples.

        Args:
            proofs: List of proof sequences
        """
        if self.verbose:
            print(f"\n{'='*70}")
            print("Learning Theorem Proving Strategy")
            print('='*70)
            print(f"  Training proofs: {len(proofs)}")

        # Prepare data
        inputs, targets = self.prepare_training_data(proofs)

        if len(inputs) == 0:
            print("WARNING: No training data!")
            return

        # All belong to condition 0 (single proof strategy)
        conditions = [0] * len(inputs)

        # Learn position-dependent rule weights
        self.learner.fit(
            V_samples=inputs,
            targets=targets,
            conditions=conditions,
            verbose=self.verbose
        )

        if self.verbose:
            print("\n✅ Learning complete!")

    def predict_rule(
        self,
        state: ProofState
    ) -> int:
        """
        Predict which rule to apply at current state.

        Args:
            state: Current proof state

        Returns:
            Rule index
        """
        # Encode state
        emb = self.encoder.encode_state(state)

        # Create sequence (pad to max_steps)
        seq = np.zeros((self.max_steps, 1))
        seq[state.step, 0] = emb[0]  # Use first embedding dim

        # Predict
        rule_probs = self.learner.predict(seq, condition=0)

        # Return argmax
        return int(np.argmax(rule_probs))


def main():
    """Test theorem prover on P → P."""

    print(f"\n{'='*70}")
    print("AUTOMATED THEOREM PROVING VIA WREATH PRODUCTS")
    print('='*70)

    # Generate training data: proof of P → P
    proof_pp = prove_identity()

    # Create prover
    prover = TheoremProverWreathAttention(
        n_rules=5,
        max_proof_steps=10,
        embedding_dim=32,
        n_characters=4,
        verbose=True
    )

    # Train on single proof
    prover.fit([proof_pp])

    # Test: Can it predict rules for the proof?
    print(f"\n{'='*70}")
    print("Testing Learned Prover")
    print('='*70)

    for i in range(len(proof_pp) - 1):
        state = proof_pp[i]
        predicted_rule = prover.predict_rule(state)
        print(f"\nStep {i}:")
        print(f"  State: {state}")
        print(f"  Predicted rule: {predicted_rule}")

    print(f"\n{'='*70}")
    print("✅ Wreath product theorem prover created!")
    print('='*70)
    print(f"\nKey insight:")
    print(f"  Proof search = position-dependent rule selection")
    print(f"  This IS wreath product structure!")
    print(f"  InferenceRules ≀ ProofSteps")


if __name__ == "__main__":
    main()
