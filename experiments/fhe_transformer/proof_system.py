"""
Minimal Propositional Logic Proof System

Simple Hilbert-style system with:
- 3 axiom schemas
- 1 inference rule (Modus Ponens)

Goal: Test if wreath product attention can learn to prove theorems!
"""

from dataclasses import dataclass
from typing import List, Set, Optional, Tuple
from enum import Enum


class Formula:
    """Base class for propositional formulas."""
    pass


@dataclass(frozen=True)
class Var(Formula):
    """Propositional variable."""
    name: str

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"Var({self.name})"


@dataclass(frozen=True)
class Implies(Formula):
    """Implication: P → Q."""
    antecedent: Formula
    consequent: Formula

    def __str__(self):
        return f"({self.antecedent} → {self.consequent})"

    def __repr__(self):
        return f"Implies({repr(self.antecedent)}, {repr(self.consequent)})"


# Convenience constructors
def var(name: str) -> Var:
    """Create variable."""
    return Var(name)


def imp(p: Formula, q: Formula) -> Implies:
    """Create implication."""
    return Implies(p, q)


class InferenceRule(Enum):
    """Available inference rules."""
    AXIOM_1 = 0  # P → (Q → P)
    AXIOM_2 = 1  # (P → (Q → R)) → ((P → Q) → (P → R))
    AXIOM_3 = 2  # (¬Q → ¬P) → (P → Q) [if we add negation]
    MODUS_PONENS = 3  # From P and P→Q, derive Q
    ASSUMPTION = 4  # Introduce assumption


@dataclass
class ProofState:
    """
    Current state of proof.

    Contains:
    - goal: What we're trying to prove
    - assumptions: Available assumptions
    - derived: Formulas we've derived so far
    - step: Current proof depth
    """
    goal: Formula
    assumptions: List[Formula]
    derived: List[Formula]
    step: int

    def __str__(self):
        return f"Step {self.step}: Derived {len(self.derived)} formulas, Goal: {self.goal}"

    def is_proven(self) -> bool:
        """Check if goal is proven."""
        return self.goal in self.derived


class ProofSystem:
    """
    Hilbert-style proof system for propositional logic.

    Axioms:
        K: P → (Q → P)
        S: (P → (Q → R)) → ((P → Q) → (P → R))

    Rule:
        MP: From P and P→Q, derive Q
    """

    def __init__(self):
        """Initialize proof system."""
        pass

    def axiom_k(self, p: Formula, q: Formula) -> Formula:
        """
        Axiom K: P → (Q → P)

        "From P, you can derive P regardless of Q"
        """
        return imp(p, imp(q, p))

    def axiom_s(self, p: Formula, q: Formula, r: Formula) -> Formula:
        """
        Axiom S: (P → (Q → R)) → ((P → Q) → (P → R))

        "Distributivity of implication"
        Also called Curry's axiom
        """
        return imp(
            imp(p, imp(q, r)),
            imp(imp(p, q), imp(p, r))
        )

    def modus_ponens(
        self,
        state: ProofState,
        p_idx: int,
        pq_idx: int
    ) -> Optional[ProofState]:
        """
        Apply Modus Ponens.

        From formulas at indices p_idx (which is P) and pq_idx (which is P→Q),
        derive Q.

        Returns new ProofState with Q added, or None if not applicable.
        """
        if p_idx >= len(state.derived) or pq_idx >= len(state.derived):
            return None

        p = state.derived[p_idx]
        pq = state.derived[pq_idx]

        # Check if pq is actually an implication with p as antecedent
        if not isinstance(pq, Implies):
            return None

        if pq.antecedent != p:
            return None

        # Derive consequent
        q = pq.consequent

        # Create new state
        new_derived = state.derived + [q]
        return ProofState(
            goal=state.goal,
            assumptions=state.assumptions,
            derived=new_derived,
            step=state.step + 1
        )

    def apply_axiom_k(
        self,
        state: ProofState,
        p: Formula,
        q: Formula
    ) -> ProofState:
        """
        Add instance of Axiom K to derived formulas.
        """
        axiom = self.axiom_k(p, q)
        new_derived = state.derived + [axiom]
        return ProofState(
            goal=state.goal,
            assumptions=state.assumptions,
            derived=new_derived,
            step=state.step + 1
        )

    def apply_axiom_s(
        self,
        state: ProofState,
        p: Formula,
        q: Formula,
        r: Formula
    ) -> ProofState:
        """
        Add instance of Axiom S to derived formulas.
        """
        axiom = self.axiom_s(p, q, r)
        new_derived = state.derived + [axiom]
        return ProofState(
            goal=state.goal,
            assumptions=state.assumptions,
            derived=new_derived,
            step=state.step + 1
        )


def prove_identity() -> List[ProofState]:
    """
    Prove P → P using axioms K and S.

    Classic proof:
    1. Axiom S with P, P→P, P:  (P → ((P→P) → P)) → ((P → (P→P)) → (P → P))
    2. Axiom K with P, P:        P → ((P→P) → P)
    3. MP on 2,1:                (P → (P→P)) → (P → P)
    4. Axiom K with P, P:        P → (P→P)
    5. MP on 4,3:                P → P ✓

    Returns list of proof states.
    """
    system = ProofSystem()
    P = var("P")

    # Initial state
    state0 = ProofState(
        goal=imp(P, P),
        assumptions=[],
        derived=[],
        step=0
    )

    proof = [state0]

    # Step 1: Axiom K to get P → ((P→P) → P)
    PP = imp(P, P)
    state1 = system.apply_axiom_k(state0, P, PP)
    proof.append(state1)

    # Step 2: Axiom S(P, P→P, P) to get (P → ((P→P) → P)) → ((P → (P→P)) → (P → P))
    state2 = system.apply_axiom_s(state1, P, PP, P)
    proof.append(state2)

    # Step 3: MP on derived[0] (axiom K) and derived[1] (axiom S)
    # This gives us (P → (P→P)) → (P → P)
    state3 = system.modus_ponens(state2, 0, 1)
    if state3 is None:
        print("ERROR: MP failed at step 3")
        return proof
    proof.append(state3)

    # Step 4: Axiom K to get P → (P→P)
    state4 = system.apply_axiom_k(state3, P, P)
    proof.append(state4)

    # Step 5: MP on derived[3] (P → (P→P)) and derived[2] ((P → (P→P)) → (P → P))
    # This gives us P → P!
    state5 = system.modus_ponens(state4, 3, 2)
    if state5 is None:
        print("ERROR: MP failed at step 5")
        return proof
    proof.append(state5)

    return proof


if __name__ == "__main__":
    """Test the proof system."""

    print("="*70)
    print("MINIMAL PROPOSITIONAL PROOF SYSTEM")
    print("="*70)

    system = ProofSystem()
    P, Q, R = var("P"), var("Q"), var("R")

    # Test axioms
    print("\nAxioms:")
    print(f"  K: {system.axiom_k(P, Q)}")
    print(f"  S: {system.axiom_s(P, Q, R)}")

    # Test proving P → P
    print("\n" + "="*70)
    print("Proving: P → P")
    print("="*70)

    proof = prove_identity()

    for i, state in enumerate(proof):
        print(f"\nStep {i}:")
        print(f"  Derived: {len(state.derived)} formulas")
        if state.derived:
            print(f"  Latest: {state.derived[-1]}")
        print(f"  Goal proven: {state.is_proven()}")

    if proof[-1].is_proven():
        print("\n✅ Successfully proved P → P!")
    else:
        print("\n❌ Failed to prove P → P")

    print("\n" + "="*70)
    print("Proof system ready for wreath product learning!")
    print("="*70)
