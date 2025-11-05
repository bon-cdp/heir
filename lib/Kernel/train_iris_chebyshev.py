#!/usr/bin/env python3
"""
Train Iris classifier with Chebyshev T_3(x) activation.

This matches our HEIR implementation:
- chebyshevPolynomial(input, 3) with optimal depth
- Bounded on [-1,1] for numerical stability
- Odd function (good for classification)
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def chebyshev_T3(x):
    """T_3(x) = 4x³ - 3x"""
    return 4*x**3 - 3*x

def chebyshev_T3_derivative(x):
    """d/dx T_3(x) = 12x² - 3"""
    return 12*x**2 - 3

# Load Iris
iris = load_iris()
X = iris.data
y = iris.target

# Normalize and clip to [-1, 1] for Chebyshev stability
scaler = StandardScaler()
X = scaler.fit_transform(X)
X = np.clip(X / 2.0, -1, 1)  # Scale down to prevent overflow

# One-hot encode
y_onehot = np.zeros((len(y), 3))
for i, label in enumerate(y):
    y_onehot[i, label] = 1

# Split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

y_train_onehot = np.zeros((len(y_train), 3))
for i in range(len(y_train)):
    y_train_onehot[i, y_train[i]] = 1

print("="*70)
print("Training Iris Classifier with Chebyshev T_3(x) Activation")
print("="*70)
print(f"\nTraining samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Activation: T_3(x) = 4x³ - 3x")
print(f"Architecture: 4 → 8 → 3\n")

# Initialize weights (small values for stability)
np.random.seed(42)
W1 = np.random.randn(8, 4) * 0.2
b1 = np.zeros(8)
W2 = np.random.randn(3, 8) * 0.2
b2 = np.zeros(3)

def forward(X, W1, b1, W2, b2):
    # Layer 1: z1 = W1·x + b1, h1 = T_3(z1)
    z1 = X @ W1.T + b1
    z1 = np.clip(z1, -1, 1)
    h1 = chebyshev_T3(z1)

    # Layer 2: z2 = W2·h1 + b2, h2 = T_3(z2)
    z2 = h1 @ W2.T + b2
    z2 = np.clip(z2, -1, 1)
    h2 = chebyshev_T3(z2)

    return z1, h1, z2, h2

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Training loop
learning_rate = 0.005
epochs = 1000
best_acc = 0
best_W1, best_b1, best_W2, best_b2 = None, None, None, None

print("Training progress:")
print("-" * 70)

for epoch in range(epochs):
    # Forward pass
    z1, h1, z2, h2 = forward(X_train, W1, b1, W2, b2)
    y_pred = softmax(h2)

    # Compute loss
    loss = -np.mean(np.sum(y_train_onehot * np.log(y_pred + 1e-8), axis=1))

    # Compute accuracy
    predictions = np.argmax(y_pred, axis=1)
    accuracy = np.mean(predictions == y_train)

    # Track best
    if accuracy > best_acc:
        best_acc = accuracy
        best_W1, best_b1 = W1.copy(), b1.copy()
        best_W2, best_b2 = W2.copy(), b2.copy()

    if epoch % 100 == 0:
        print(f"Epoch {epoch:4d}: Loss={loss:.4f}, Train Acc={accuracy:.4f}, Best={best_acc:.4f}")

    # Backpropagation
    # Output layer
    dL_dy = (y_pred - y_train_onehot) / len(y_train)
    dL_dz2 = dL_dy * chebyshev_T3_derivative(z2)
    dL_dW2 = dL_dz2.T @ h1
    dL_db2 = np.sum(dL_dz2, axis=0)

    # Hidden layer
    dL_dh1 = dL_dz2 @ W2
    dL_dz1 = dL_dh1 * chebyshev_T3_derivative(z1)
    dL_dW1 = dL_dz1.T @ X_train
    dL_db1 = np.sum(dL_dz1, axis=0)

    # Gradient clipping for stability
    grad_clip = 1.0
    dL_dW2 = np.clip(dL_dW2, -grad_clip, grad_clip)
    dL_db2 = np.clip(dL_db2, -grad_clip, grad_clip)
    dL_dW1 = np.clip(dL_dW1, -grad_clip, grad_clip)
    dL_db1 = np.clip(dL_db1, -grad_clip, grad_clip)

    # Update weights
    W2 -= learning_rate * dL_dW2
    b2 -= learning_rate * dL_db2
    W1 -= learning_rate * dL_dW1
    b1 -= learning_rate * dL_db1

# Use best weights
W1, b1, W2, b2 = best_W1, best_b1, best_W2, best_b2

# Test accuracy
z1_test, h1_test, z2_test, h2_test = forward(X_test, W1, b1, W2, b2)
y_pred_test = softmax(h2_test)
predictions_test = np.argmax(y_pred_test, axis=1)
test_acc = np.mean(predictions_test == y_test)

print("-" * 70)
print(f"\n✅ FINAL TEST ACCURACY: {test_acc:.4f}")
print(f"   Best training accuracy: {best_acc:.4f}\n")

# Show detailed predictions
class_names = ['Setosa', 'Versicolor', 'Virginica']
print("Sample predictions on test set:")
correct = 0
for i in range(len(X_test)):
    true_class = y_test[i]
    pred_class = predictions_test[i]
    if true_class == pred_class:
        correct += 1
    if i < 10:  # Show first 10
        status = '✓' if true_class == pred_class else '✗'
        print(f"  {status} Sample {i:2d}: True={class_names[true_class]:12s} Pred={class_names[pred_class]:12s}")

print(f"\nCorrect: {correct}/{len(X_test)}")

# Export to C++
print("\n" + "="*70)
print("C++ Code for HEIR (Copy into NeuralNetworkKernelsTest.cpp)")
print("="*70 + "\n")

print(f"""
// Trained weights for Iris classification with Chebyshev T_3(x) activation
// Test Accuracy: {test_acc:.4f}
// Architecture: 4 → 8 → 3 with T_3(x) = 4x³ - 3x
// Depth: 2 per layer (optimal x³ computation via PolynomialComposer)

""")

print("// Hidden layer weights (8x4)")
print("const std::vector<std::vector<double>> W1 = {")
for i in range(8):
    print("  {", end="")
    for j in range(4):
        print(f"{W1[i,j]:11.6f}", end="")
        if j < 3: print(",", end="")
    print("},")
print("};")
print()

print("// Hidden layer biases (8)")
print("const std::vector<double> b1 = {", end="")
for i in range(8):
    print(f"{b1[i]:11.6f}", end="")
    if i < 7: print(",", end="")
print("};")
print()

print("// Output layer weights (3x8)")
print("const std::vector<std::vector<double>> W2 = {")
for i in range(3):
    print("  {", end="")
    for j in range(8):
        print(f"{W2[i,j]:11.6f}", end="")
        if j < 7: print(",", end="")
    print("},")
print("};")
print()

print("// Output layer biases (3)")
print("const std::vector<double> b2 = {", end="")
for i in range(3):
    print(f"{b2[i]:11.6f}", end="")
    if i < 2: print(",", end="")
print("};")
print()

# Export test samples
print("// Test samples (one from each class)")
for class_idx in range(3):
    idx = np.where(y_test == class_idx)[0][0]
    class_name = class_names[class_idx].lower()

    print(f"// {class_names[class_idx]} sample (class {class_idx})")
    print(f"const std::vector<double> {class_name}_features = {{", end="")
    for i in range(4):
        print(f"{X_test[idx,i]:11.6f}", end="")
        if i < 3: print(",", end="")
    print("};")
    pred = predictions_test[idx]
    correct_mark = "✓ CORRECT" if pred == class_idx else "✗ WRONG"
    print(f"// Expected: class {class_idx}, Network predicts: class {pred} ({correct_mark})")
    print()

print("""
// Usage in HEIR test:
// 1. Build network DAG with implementMultiLayerNetwork()
// 2. Use chebyshevPolynomial(input, 3) for activation (automatically optimal depth)
// 3. Measure depth with MultiplicativeDepthVisitor
// 4. Expected depth: 2 layers × 2 (T_3 depth) = 4 in conservative model
//                    In real FHE: depth 2 (plaintext-ciphertext mul is free)
""")

print("\n" + "="*70)
print(f"✅ Training complete! Test accuracy: {test_acc:.4f}")
print("="*70)
