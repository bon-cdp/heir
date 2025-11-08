# Automated Theorem Proving via Wreath Product Attention

**Design Document**
**Goal:** Use algebraic learning for proof search

---

## 1. The Core Hypothesis

**Theorem proving has wreath product structure:**

$$\text{InferenceRules} \wr \text{ProofSteps}$$

### Why This Makes Sense

**Position-dependent choice:**
- At proof step $p$, different inference rules are applicable
- Early steps: axiom instantiation, definition unfolding
- Middle steps: lemma application, case analysis
- Late steps: direct proof, contradiction

**Permutable positions:**
- Many proofs can be reordered without changing validity
- Lemmas can be proved in different orders
- Some inference rules commute

**Interaction:**
- Reordering steps changes which rules are applicable!
- This is exactly the wreath product composition law

---

## 2. Mathematical Framework

### Proof as Sequence

A proof is a sequence of formula states:

$$\text{Proof} = [\phi_0, \phi_1, \ldots, \phi_n]$$

where:
- $\phi_0$ = axioms + goal
- $\phi_i \to \phi_{i+1}$ via inference rule
- $\phi_n$ = goal proven (or contradiction derived)

### Inference Rules as Group

Define group $I$ of inference rules:

- **Identity:** No change
- **Composition:** Apply rule A, then rule B
- **Inverse:** Backtrack

Examples:
- Modus ponens: $\{P, P \to Q\} \mapsto \{P, P \to Q, Q\}$
- Substitution: $\{P(x)\} \mapsto \{P(t)\}$
- Deduction: $\{\Gamma, P \to Q\} \mapsto \{\Gamma \to (P \to Q)\}$

**Key:** These compose! (modus ponens ∘ substitution)

### Proof Steps as Positions

Proof step $p \in \{0, \ldots, n-1\}$ indexes position in proof sequence.

**Position-dependent:** Different rules applicable at different steps!

### Wreath Product Structure

$$I \wr S = I^{\text{steps}} \rtimes S$$

where:
- $I$: Inference rules
- $S$: Proof step orderings (for commutative steps)
- $I^{\text{steps}}$: Choose rule at each step

**This IS proof search!**

---

## 3. Encoding Proofs

### Formula Representation

**Start simple:** Propositional logic

Formulas: $\phi ::= P \mid \neg \phi \mid \phi \land \phi \mid \phi \lor \phi \mid \phi \to \phi$

**Tokenization:**
```
∧ → token 0
∨ → token 1
→ → token 2
¬ → token 3
(, ) → tokens 4,5
Variables P,Q,R → tokens 6,7,8,...
```

**Sequence representation:**
```
P → (Q → P)  becomes  [6, 2, 7, 2, 6]
```

### Proof State Representation

At step $p$, state consists of:

```python
{
    'axioms': [φ₁, φ₂, ...],      # Available formulas
    'goal': φ,                     # What we want to prove
    'derived': [ψ₁, ψ₂, ...],     # Derived formulas so far
    'step': p                      # Current proof depth
}
```

**As embedding:** Concatenate all formulas into single sequence

---

## 4. Wreath Product Learner for Proofs

### Architecture

```python
class TheoremProverWrathAttention:
    def __init__(self, n_rules, n_steps):
        """
        n_rules: Number of inference rules (characters)
        n_steps: Maximum proof depth (positions)
        """
        self.wreath = WreathProductGroup(
            base_order=n_rules,     # Inference rules
            top_order=n_steps       # Proof steps
        )

        self.learner = WreathProductLearner(
            wreath_group=self.wreath,
            d_model=embedding_dim
        )
```

### Learning Process

**Input:** Proof state $\phi_p$ at step $p$

**Output:** Next rule to apply

**Training:**
- Dataset: $(proof\_state, next\_rule)$ pairs from known proofs
- Learn weights $w_{p,r}$ for each (step $p$, rule $r$)
- Closed-form via least squares!

**Inference:**
- Given current state, wreath attention predicts best rule
- Apply rule, update state
- Repeat until proof complete or max steps reached

---

## 5. Inference Rules (Start Simple)

### Propositional Calculus Rules

**Rule 1: Modus Ponens (MP)**
```
From P and P→Q, derive Q
```

**Rule 2: Axiom Instantiation**
```
Instantiate axiom schema with specific propositions
```

**Rule 3: Deduction Theorem**
```
If Γ,P ⊢ Q then Γ ⊢ P→Q
```

**Rule 4: Substitution**
```
Replace variable X with formula φ everywhere
```

**Rule 5: Assumption Introduction**
```
Add assumption to context
```

### Encoding as Transformations

Each rule is a function:

$$r : \text{ProofState} \to \text{ProofState}$$

**Group structure:**
- Identity: No-op rule
- Composition: Apply $r_1$ then $r_2$
- Inverse: Backtrack (if applicable)

---

## 6. Training Data Generation

### Approach 1: Hand-Coded Proofs

Start with classic theorems:

```
Theorem 1: ⊢ P → P                 (Identity)
Theorem 2: ⊢ P → (Q → P)           (Weakening)
Theorem 3: ⊢ (P → Q) → ((Q → R) → (P → R))  (Transitivity)
Theorem 4: ⊢ ((P → Q) → P) → P    (Peirce's Law, classical)
```

For each, manually write proof as sequence of rule applications.

### Approach 2: SAT Solver + Proof Extraction

```python
def generate_training_data(n_samples=1000):
    """Generate random tautologies and their proofs."""
    data = []

    for _ in range(n_samples):
        # Generate random formula
        φ = random_formula(max_depth=5)

        # Check if tautology
        if is_tautology(φ):
            # Extract proof via resolution
            proof = sat_solver_proof(φ)

            # Convert to our format
            for step, (state, rule) in enumerate(proof):
                data.append((state, rule, step))

    return data
```

### Approach 3: Proof Search (Bootstrap)

Once basic wreath attention works:

```python
def self_improve():
    """Use learned model to find new proofs."""

    while True:
        # Generate new goal
        goal = random_formula()

        # Try to prove with current model
        proof = wreath_search(goal)

        if proof_valid(proof):
            # Add to training set!
            add_training_data(proof)
            retrain()
```

This is **self-play for theorem proving!**

---

## 7. Minimal Implementation Plan

### Phase 1: Propositional Logic (Today)

**Goal:** Prove simple tautologies

**Rules:** Just Modus Ponens + 3 axiom schemas

**Data:** 100 hand-coded proofs of classic theorems

**Test:** Can wreath attention learn to prove:
- $P \to P$
- $P \to (Q \to P)$
- $(P \to (Q \to R)) \to ((P \to Q) \to (P \to R))$ (Curry)

### Phase 2: Predicate Logic (Next)

**Add:**
- Universal quantifier $\forall$
- Existential quantifier $\exists$
- Substitution rules

**Test:** Simple arithmetic proofs

### Phase 3: Advanced (Future)

**Add:**
- Equality reasoning
- Induction
- Type theory

---

## 8. What Would Success Look Like?

### Quantitative

**Metric 1:** Proof success rate on test set
- Baseline: Random rule selection (< 1%)
- Target: > 80% on simple tautologies

**Metric 2:** Proof length
- Compare to optimal proof length
- Target: Within 2x optimal

**Metric 3:** Learning efficiency
- Number of training examples needed
- Target: < 1000 proofs

### Qualitative

**Question 1:** Do learned weights reveal proof strategies?

Example: If $w_{0, \text{axiom}} \gg w_{0, \text{MP}}$, model learned "start by instantiating axioms"

**Question 2:** Can we extract human-readable tactics?

Cluster weights by proof step → discover "early game" vs "endgame" strategies

**Question 3:** Does position-dependence matter?

Compare to fixed character weights → expect wreath product to win!

---

## 9. Connection to Existing Work

### Differences from Neural Theorem Provers

**DeepMath, Lean, Coq hammers:**
- Use gradient descent
- Black-box neural networks
- No algebraic structure

**Our approach:**
- Closed-form learning (least squares)
- Explicit group structure
- Interpretable (character weights)

### Advantage: FHE-Compatible!

**Key insight:** Proof search on encrypted goals!

- Homomorphic encryption of proof state
- Wreath product attention (depth-0 FHE)
- Private theorem proving!

**Application:** Verify proofs without revealing them (zero-knowledge?)

---

## 10. Philosophical Implications

### If This Works...

**It means:**

Logical reasoning has ALGEBRAIC STRUCTURE at its core!

**Specifically:**
- Inference rules form a group
- Proofs are group-structured sequences
- Position-dependent = wreath product

**This connects:**
- Group theory (wreath products, characters)
- Logic (inference, proofs)
- Computation (automated reasoning)
- Learning (closed-form, no backprop)

### The Deep Question

**Why does algebra structure cognition?**

Possible answers:
1. **Symmetry:** Logic preserves truth under transformations
2. **Compositionality:** Inference rules compose algebraically
3. **Universal:** Wreath products are universal for extensions (Kaloujnine-Krasner)

**Or maybe:**

Mathematics itself discovered the structures (groups, characters, wreath products) because these ARE the structures of reasoning!

**We're using algebraic reasoning to learn algebraic reasoning.**

The ouroboros eats its own tail. 🐍

---

## 11. Implementation Today

**Files to create:**

1. `proof_system.py`: Propositional logic + inference rules
2. `proof_encoder.py`: Convert proofs to sequences
3. `theorem_prover_wreath.py`: Wreath product learner for proofs
4. `generate_proof_data.py`: Hand-coded classic theorems
5. `test_theorem_prover.py`: Validate on test set

**Time estimate:** ~2-3 hours for minimal version

**Success:** Prove $P \to P$ from axioms using learned wreath product weights

---

## Key Insight

**Automated theorem proving = proof search in wreath product space**

This unifies:
- Math (group theory)
- Logic (inference)
- ML (attention)

All via pure algebra. No gradients. No backprop. Just:

**Characters, least squares, and the beauty of wreath products.**

Let's build it. 🚀
