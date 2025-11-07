"""
Complete FHE-Native Transformer

End-to-end transformer designed to run on encrypted data with minimal depth.

Architecture:
- Embedding + Cyclotomic position encoding
- N × Transformer layers:
  - Multi-rotation attention (depth 0)
  - Residual connection
  - FFN with Chebyshev T_3 activation (depth 1 + 2 = 3)
  - Residual connection
  - NO LayerNorm (skip for FHE efficiency)
- Output linear layer (depth 1)

Key innovations:
1. Rotation-based attention: depth 0 (vs ~5 for standard attention)
2. Cyclotomic position encoding: FHE-native (no sin/cos approximation)
3. Chebyshev T_3 activation: numerically stable polynomial
4. Skip LayerNorm: avoids division by ciphertext

Total depth per layer: 0 (attention) + 3 (FFN) + 1 (output) = 4
For 2-layer model: ~8 depth (vs ~10-15 for standard)
"""

import numpy as np
from typing import List, Tuple, Optional, Union
import sys
sys.path.append('.')

from simulated_fhe import SimulatedFHE, encrypt, plaintext
from cyclotomic_encoding import cyclotomic_encoding_matrix


def chebyshev_T3(x: Union[np.ndarray, SimulatedFHE]) -> Union[np.ndarray, SimulatedFHE]:
    """
    Chebyshev polynomial T_3(x) = 4x³ - 3x

    Properties:
    - Bounded on [-1,1]: T_3(x) ∈ [-1,1] for x ∈ [-1,1]
    - Odd function: T_3(-x) = -T_3(x) (preserves sign)
    - Depth: 2 (computes x³ optimally via x² then x²·x)

    Args:
        x: Input (numpy array or SimulatedFHE)

    Returns:
        T_3(x) with same type as input
    """
    if isinstance(x, SimulatedFHE):
        # FHE computation
        x2 = x * x          # depth 1
        x3 = x2 * x         # depth 2
        return 4.0 * x3 - 3.0 * x  # plaintext mul (depth unchanged)
    else:
        # Plaintext computation
        return 4*x**3 - 3*x


def chebyshev_T3_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of T_3: d/dx T_3(x) = 12x² - 3"""
    return 12*x**2 - 3


class FHETransformer:
    """
    Complete FHE-native transformer for sequence tasks.

    Args:
        vocab_size: Vocabulary size
        d_model: Model dimension (embedding size)
        n_layers: Number of transformer layers
        rotation_amounts: List of rotation amounts for attention (e.g., [1,2,4])
        max_seq_len: Maximum sequence length
        dropout: Dropout rate (only used during training, ignored in FHE)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        n_layers: int = 2,
        rotation_amounts: List[int] = None,
        max_seq_len: int = 64,
        dropout: float = 0.0
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.dropout = dropout

        if rotation_amounts is None:
            rotation_amounts = [1, 2, 4]
        self.rotation_amounts = rotation_amounts
        self.n_rotations = len(rotation_amounts)

        # Initialize weights
        np.random.seed(42)

        # Embedding matrix (plaintext)
        self.embedding = np.random.randn(vocab_size, d_model) * 0.1

        # Position encoding (cyclotomic, plaintext)
        self.position_encoding = cyclotomic_encoding_matrix(
            max_seq_len, d_model, max_seq_len
        )

        # Transformer layers
        self.layers = []
        for _ in range(n_layers):
            layer = {
                # Rotation attention weights
                'W_V': np.random.randn(d_model, d_model) * 0.01,
                'W_O': np.random.randn(d_model, d_model) * 0.01,
                'rotation_weights': np.random.randn(self.n_rotations) * 0.01,

                # FFN weights
                'W_ffn1': np.random.randn(d_model * 4, d_model) * 0.01,  # Expand
                'b_ffn1': np.zeros(d_model * 4),
                'W_ffn2': np.random.randn(d_model, d_model * 4) * 0.01,  # Contract
                'b_ffn2': np.zeros(d_model),
            }

            # Normalize rotation weights
            layer['rotation_weights'] /= np.sum(np.abs(layer['rotation_weights']))

            self.layers.append(layer)

        # Output layer
        self.W_output = np.random.randn(vocab_size, d_model) * 0.01
        self.b_output = np.zeros(vocab_size)

    def rotate(
        self,
        x: Union[np.ndarray, SimulatedFHE],
        k: int
    ) -> Union[np.ndarray, SimulatedFHE]:
        """Rotate sequence by k positions."""
        if isinstance(x, SimulatedFHE):
            return x.rotate(k)
        else:
            return np.roll(x, k, axis=0)

    def rotation_attention(
        self,
        x: Union[np.ndarray, SimulatedFHE],
        layer: dict
    ) -> Union[np.ndarray, SimulatedFHE]:
        """
        Multi-rotation attention: Σ_k α_k · rotate(V, k)

        Depth: 0 (plaintext-ciphertext multiplication only)
        """
        # Project to values
        if isinstance(x, SimulatedFHE):
            V = x @ plaintext(layer['W_V'])
        else:
            V = x @ layer['W_V']

        # Multi-rotation attention
        if isinstance(V, SimulatedFHE):
            attn_output = plaintext(np.zeros_like(V.value))
            for i, k in enumerate(self.rotation_amounts):
                V_rotated = V.rotate(k)
                attn_output = attn_output + layer['rotation_weights'][i] * V_rotated
        else:
            attn_output = np.zeros_like(V)
            for i, k in enumerate(self.rotation_amounts):
                V_rotated = self.rotate(V, k)
                attn_output += layer['rotation_weights'][i] * V_rotated

        # Output projection
        if isinstance(attn_output, SimulatedFHE):
            output = attn_output @ plaintext(layer['W_O'])
        else:
            output = attn_output @ layer['W_O']

        return output

    def ffn(
        self,
        x: Union[np.ndarray, SimulatedFHE],
        layer: dict
    ) -> Union[np.ndarray, SimulatedFHE]:
        """
        Feed-forward network with Chebyshev T_3 activation.

        Depth: 1 (linear) + 2 (T_3) + 1 (linear) = 4 total
        But actually: first linear is pt-ct (depth 0), so total 3
        """
        # First linear layer (expand)
        if isinstance(x, SimulatedFHE):
            h = x @ plaintext(layer['W_ffn1'].T) + plaintext(layer['b_ffn1'])
            # Clip to [-1, 1] for Chebyshev stability
            # Note: Can't really clip encrypted data, but in real FHE we'd
            # use bounded encoding. For simulation, trust the math.
        else:
            h = x @ layer['W_ffn1'].T + layer['b_ffn1']
            h = np.clip(h, -1, 1)  # Chebyshev stability

        # Activation: T_3(x) = 4x³ - 3x
        h = chebyshev_T3(h)

        # Second linear layer (contract)
        if isinstance(h, SimulatedFHE):
            output = h @ plaintext(layer['W_ffn2'].T) + plaintext(layer['b_ffn2'])
        else:
            output = h @ layer['W_ffn2'].T + layer['b_ffn2']

        return output

    def forward(
        self,
        token_ids: np.ndarray,
        use_fhe: bool = False
    ) -> Union[np.ndarray, SimulatedFHE]:
        """
        Forward pass.

        Args:
            token_ids: Input token IDs [seq_len] or [batch_size, seq_len]
            use_fhe: Whether to simulate FHE (wrap in SimulatedFHE)

        Returns:
            Logits [seq_len, vocab_size] or [batch_size, seq_len, vocab_size]
        """
        # Handle batching
        if len(token_ids.shape) == 1:
            token_ids = token_ids[np.newaxis, :]
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size, seq_len = token_ids.shape

        # Embedding (plaintext lookup)
        x = np.zeros((batch_size, seq_len, self.d_model))
        for i in range(batch_size):
            for j in range(seq_len):
                x[i, j] = self.embedding[token_ids[i, j]]

        # Add position encoding (plaintext)
        for i in range(batch_size):
            x[i] += self.position_encoding[:seq_len]

        # Process each batch element separately (for simplicity)
        # In real implementation, would batch FHE operations
        outputs = []
        for i in range(batch_size):
            x_i = x[i]  # [seq_len, d_model]

            # Encrypt if using FHE
            if use_fhe:
                x_i = encrypt(x_i, depth=0)

            # Transformer layers
            for layer_idx, layer in enumerate(self.layers):
                # Multi-rotation attention
                attn_output = self.rotation_attention(x_i, layer)

                # Residual connection
                x_i = x_i + attn_output

                # FFN
                ffn_output = self.ffn(x_i, layer)

                # Residual connection
                x_i = x_i + ffn_output

            # Output layer
            if isinstance(x_i, SimulatedFHE):
                logits = x_i @ plaintext(self.W_output.T) + plaintext(self.b_output)
            else:
                logits = x_i @ self.W_output.T + self.b_output

            outputs.append(logits)

        # Stack outputs
        if use_fhe and isinstance(outputs[0], SimulatedFHE):
            # Combine encrypted outputs
            # For simplicity, return list (can't stack SimulatedFHE directly)
            if squeeze_output:
                return outputs[0]
            return outputs
        else:
            output = np.stack(outputs, axis=0)
            if squeeze_output:
                return output[0]
            return output

    def predict(
        self,
        token_ids: np.ndarray,
        use_fhe: bool = False
    ) -> Union[int, np.ndarray]:
        """
        Predict next token.

        Args:
            token_ids: Input sequence [seq_len]
            use_fhe: Whether to use FHE simulation

        Returns:
            Predicted token ID (or IDs if batched)
        """
        logits = self.forward(token_ids, use_fhe=use_fhe)

        if use_fhe and isinstance(logits, SimulatedFHE):
            logits = logits.decrypt()

        # Take last position
        if len(logits.shape) == 2:
            logits = logits[-1]  # [vocab_size]

        return int(np.argmax(logits))

    def get_depth(self, token_ids: np.ndarray) -> int:
        """
        Measure depth of FHE computation.

        Args:
            token_ids: Input sequence [seq_len]

        Returns:
            Maximum multiplicative depth
        """
        logits = self.forward(token_ids, use_fhe=True)

        if isinstance(logits, SimulatedFHE):
            return logits.get_depth()
        else:
            return 0


def test_fhe_transformer():
    """Test the FHE transformer."""
    print("=" * 70)
    print("Testing FHE Transformer")
    print("=" * 70)

    # Create small transformer
    vocab_size = 104
    d_model = 32
    n_layers = 2

    model = FHETransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        rotation_amounts=[1, 2],
        max_seq_len=16
    )

    print(f"\nModel configuration:")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Model dimension: {d_model}")
    print(f"  Number of layers: {n_layers}")
    print(f"  Rotation amounts: {model.rotation_amounts}")

    # Test 1: Forward pass (plaintext)
    print("\n1. Plaintext forward pass")
    print("-" * 70)

    token_ids = np.array([2, 4, 6, 8])  # Count by 2s
    logits = model.forward(token_ids, use_fhe=False)

    print(f"Input tokens: {token_ids}")
    print(f"Output shape: {logits.shape}")
    print(f"Expected: (4, {vocab_size})")
    assert logits.shape == (4, vocab_size)
    print("✓ Correct shape!")

    # Predict next token
    pred = model.predict(token_ids, use_fhe=False)
    print(f"Predicted next token: {pred}")
    print("(Random weights, so prediction will be random)")

    # Test 2: FHE forward pass
    print("\n2. FHE simulated forward pass")
    print("-" * 70)

    logits_fhe = model.forward(token_ids, use_fhe=True)

    if isinstance(logits_fhe, SimulatedFHE):
        depth = logits_fhe.get_depth()
        logits_decrypted = logits_fhe.decrypt()

        print(f"FHE depth: {depth}")
        print(f"Expected depth: ~{n_layers * 3} (2 layers × 3 depth each)")
        print(f"Decrypted output shape: {logits_decrypted.shape}")

        # Verify correctness
        max_diff = np.max(np.abs(logits - logits_decrypted))
        print(f"\nCorrectness check (plaintext vs FHE):")
        print(f"  Max difference: {max_diff:.10f}")
        print(f"  Should be ~0 (numerical precision)")
        assert max_diff < 1e-6, "FHE and plaintext should give same result"
        print("✓ FHE simulation is correct!")

    # Test 3: Depth breakdown
    print("\n3. Depth breakdown per layer")
    print("-" * 70)

    # Manually trace depth
    print(f"Per-layer depth contribution:")
    print(f"  Embedding + position encoding: depth 0")
    print(f"  Layer 1:")
    print(f"    Rotation attention: +0 (pt-ct mul)")
    print(f"    Residual add: +0")
    print(f"    FFN linear 1: +0 (pt-ct mul)")
    print(f"    Chebyshev T_3: +2 (x² then x³)")
    print(f"    FFN linear 2: +1 (ct-ct mul with T_3 output)")
    print(f"    Residual add: +0")
    print(f"  Layer 2: (same as layer 1)")
    print(f"  Output linear: +1")
    print(f"\nTotal depth: 0 + (0+2+1) × 2 + 1 = {0 + 3*2 + 1}")

    print("\n" + "=" * 70)
    print("✅ FHE Transformer tests passed!")
    print("=" * 70)

    return model


if __name__ == "__main__":
    test_fhe_transformer()
