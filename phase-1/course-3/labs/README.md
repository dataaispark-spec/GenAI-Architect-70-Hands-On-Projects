# Course 3: Neural Networks Fundamentals - Hands-on Labs 🧠🛠️

## Lab Overview: Neuron Simulation and Neural Network Basics 🔬

This lab immerses you in building artificial neurons from scratch, understanding activation functions, and constructing simple neural networks. Perfect for grasping the mathematical core of deep learning through hands-on coding.

---

## Prerequisites
- Google account for Colab access
- Basic Python knowledge (data types, numpy arrays)
- Familiarity with tabular data processing
- No prior ML knowledge needed!

## 🏗️ **Lab Objectives**
- Simulate single neurons with code
- Experiment with different activation functions
- Build multi-layer neural networks from scratch
- Understand forward pass computations
- Visualize decision boundaries

## 📋 **Step-by-Step Instructions**

### Step 1: Setup Environment 🌟
Create a new Google Colab notebook: `course3_neural_networks.ipynb`

Import essential libraries:
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles
import warnings
warnings.filterwarnings('ignore')

# Enable inline plotting
%matplotlib inline
```

### Step 2: Single Neuron Simulation 🧩
Build your first artificial neuron in pure NumPy:

```python
def sigmoid(x):
    """Sigmoid activation: converts any value to 0-1 range"""
    return 1 / (1 + np.exp(-x))

def relu(x):
    """ReLU activation: max(0, x) - prevents vanishing gradients"""
    return np.maximum(0, x)

def tanh(x):
    """Tanh activation: -1 to 1 range, centered sigmoid"""
    return np.tanh(x)

def neuron_forward(inputs, weights, bias, activation):
    """Compute neuron output: weighted sum + activation"""
    weighted_sum = np.dot(inputs, weights) + bias
    return activation(weighted_sum), weighted_sum  # return both for analysis

# Example: Single neuron with 2 inputs
np.random.seed(42)
n_inputs = 2
weights = np.random.randn(n_inputs) * 0.1  # Small random weights
bias = 0.0  # No bias initially

# Test on simple inputs
test_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
print("Input → Weighted Sum → Output (Sigmoid)")
print("-" * 40)
for inputs in test_inputs:
    output, z = neuron_forward(inputs, weights, bias, sigmoid)
    print(f"{inputs} → {z:.3f} → {output:.3f}")

# Visualize different activations
x = np.linspace(-3, 3, 100)
plt.figure(figsize=(15, 4))

activations = [('Sigmoid', sigmoid), ('ReLU', relu), ('Tanh', tanh)]
for i, (name, func) in enumerate(activations, 1):
    plt.subplot(1, 3, i)
    plt.plot(x, func(x))
    plt.title(f'{name} Activation Function')
    plt.grid(True, alpha=0.3)
    plt.xlabel('Input (z)')
    plt.ylabel('Output (a)')

plt.tight_layout()
plt.show()
```

### Step 3: Activation Function Comparison ⚡
Experiment with different activations on the same data:

```python
# Generate sample data
X, y = make_moons(n_samples=1000, noise=0.1, random_state=42)

# Train simple neuron for binary classification
def simple_neuron_classifier(X, activations):
    """Compare different activations on same task"""
    results = {}
    for name, activation in activations.items():
        # Initialize random weights and biases
        weights = np.random.randn(X.shape[1])
        bias = 0.0
        
        # Simple gradient descent (we'll improve this later)
        learning_rate = 0.01
        n_epochs = 100
        
        losses = []
        for epoch in range(n_epochs):
            # Forward pass
            z = np.dot(X, weights) + bias
            predictions = activation(z)
            
            # Simple loss (for demonstration - not optimal)
            loss = np.mean((predictions - y) ** 2)
            losses.append(loss)
            
            # Naive gradient update (concept only)
            dz = predictions - y
            dw = np.dot(X.T, dz) / len(X)
            db = np.mean(dz)
            
            weights -= learning_rate * dw
            bias -= learning_rate * db
        
        results[name] = {
            'weights': weights,
            'bias': bias,
            'final_loss': losses[-1],
            'loss_history': losses
        }
    
    return results

# Compare sigmoid vs ReLU
activations = {'Sigmoid': sigmoid, 'ReLU': relu}
results = simple_neuron_classifier(X, activations)

# Plot comparison
plt.figure(figsize=(12, 4))

# Loss curves
plt.subplot(1, 2, 1)
for name, result in results.items():
    plt.plot(result['loss_history'], label=name)
plt.xlabel('Training Epochs') 
plt.ylabel('Loss')
plt.title('Learning Curves: Sigmoid vs ReLU')
plt.legend()
plt.grid(True, alpha=0.3)

# Decision boundaries
plt.subplot(1, 2, 2)
for name, result in results.items():
    weights = result['weights']
    bias = result['bias']
    
    # Create grid for boundary visualization
    xx, yy = np.meshgrid(np.linspace(X[:,0].min()-1, X[:,0].max()+1, 100),
                        np.linspace(X[:,1].min()-1, X[:,1].max()+1, 100))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    # Compute predictions on grid
    if name == 'Sigmoid':
        z_grid = np.dot(grid_points, weights) + bias
        preds_grid = sigmoid(z_grid)
    else:  # ReLU
        z_grid = np.dot(grid_points, weights) + bias
        preds_grid = relu(z_grid)
    
    # Visualize
    plt.contourf(xx, yy, preds_grid.reshape(xx.shape), alpha=0.3, cmap='coolwarm', levels=50)
    
plt.scatter(X[:,0], X[:,1], c=y, cmap='coolwarm', edgecolors='k', alpha=0.7)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2') 
plt.title('Decision Boundaries')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Step 4: Multi-Layer Network Construction 🏗️
Build a simple 2-layer neural network:

```python
class SimpleNeuralNetwork:
    """A simple neural network with 1 hidden layer"""
    
    def __init__(self, input_size, hidden_size, output_size):
        # Random weight initialization (He initialization for ReLU)
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2/input_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2/hidden_size)  
        self.b2 = np.zeros(output_size)
        
    def forward(self, X):
        """Forward pass through network"""
        # Hidden layer
        z1 = np.dot(X, self.W1) + self.b1
        a1 = relu(z1)  # ReLU activation
        
        # Output layer  
        z2 = np.dot(a1, self.W2) + self.b2
        a2 = sigmoid(z2)  # Sigmoid for binary classification
        
        return a2, {'z1': z1, 'a1': a1, 'z2': z2, 'a2': a2}
    
    def predict(self, X):
        """Make predictions"""
        predictions, _ = self.forward(X)
        return (predictions > 0.5).astype(int)

# Initialize network
network = SimpleNeuralNetwork(input_size=2, hidden_size=10, output_size=1)

# Test on sample data
sample_X = X[:10]
predictions, activations = network.forward(sample_X)

print(f"Sample predictions (probabilities): {predictions.ravel()[:5]}")
print(f"Sample activations: Hidden layer mean = {activations['a1'].mean():.3f}")
```

### Step 5: Network Architecture Visualization 📊
Understand how data flows through layers:

```python
def visualize_network_flow():
    """Visualize forward pass through a simple network"""
    # Create simple 2-layer network
    W1 = np.array([[0.6, 0.2], [0.1, 0.8]])  # 2 inputs -> 2 hidden
    b1 = np.array([0.1, 0.0])
    W2 = np.array([[0.7], [0.9]])  # 2 hidden -> 1 output
    b2 = np.array([0.0])
    
    # Sample input
    x = np.array([0.8, 0.6])
    
    # Forward pass step by step
    z1 = np.dot(x, W1) + b1
    a1 = relu(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)
    
    print("Network Flow Visualization:")
    print(f"Input: {x}")
    print(f"Hidden weighted sum (z1): {z1}")
    print(f"Hidden activation (a1): {a1}")
    print(f"Output weighted sum (z2): {z2}")  
    print(f"Final prediction (a2): {a2[0]:.4f}")
    
    # Visualize computation graph (text-based)
    print("\n" + "="*50)
    print("COMPUTATION GRAPH:")
    print("="*50)
    print(f"x1 = {x[0]}")  
    print("x1" + " " * 7 + f"x2 = {x[1]:.1f}")
    print("│" + " " * 7 + "│")
    print("├─w11=" + f"{W1[0][0]:.1f}" + "─┐ " + "├─w21=" + f"{W1[1][0]:.1f}" + "─┐")
    print("│" + " " * 14 + "│" + " " * 14 + "│")
    print(f"├─w12={W1[0][1]:.1f}" + "─┘ " + f"├─w22={W1[1][1]:.1f}" + "─┘")
    print("│" + " " * 14 + "│")
    print("v" + " " * 7 + "v")
    print(f"z1₁ = {z1[0]:.2f}" + f"      z1₂ = {z1[1]:.2f}")
    print("ReLU" + " " * 4 + "ReLU")
    print("│" + " " * 7 + "│")
    print("v" + " " * 7 + "v")
    print(f"a1₁ = {a1[0]:.2f}" + f"      a1₂ = {a1[1]:.2f}")
    print("│" + " " * 7 + "│")
    print("├─w31=" + f"{W2[0][0]:.1f}" + "─┐ " + "├─w41=" + f"{W2[1][0]:.1f}" + "─┐")
    print("└──────────────┘ " + "└──────────────┘")
    print("z2 = " + f"{z2[0]:.2f}")
    print("Sigmoid")
    print("│")
    print("v")
    print("Prediction = " + f"{a2[0]:.4f}")
    
visualize_network_flow()
```

### Step 6: Experiment with Complex Data 🔬
Test network performance on non-linear datasets:

```python
# Generate non-linear data
X_circles, y_circles = make_circles(n_samples=500, noise=0.05, random_state=42)
X_moons, y_moons = make_moons(n_samples=500, noise=0.05, random_state=42)

datasets = [('Circles', X_circles, y_circles), ('Moons', X_moons, y_moons)]

plt.figure(figsize=(16, 8))

for i, (name, X_data, y_data) in enumerate(datasets):
    # Train network (simplified training - concept demonstration)
    network = SimpleNeuralNetwork(2, 20, 1)  # 20 hidden neurons
    
    # Create decision boundary visualization
    xx, yy = np.meshgrid(np.linspace(X_data[:,0].min()-0.5, X_data[:,0].max()+0.5, 100),
                        np.linspace(X_data[:,1].min()-0.5, X_data[:,1].max()+0.5, 100))
    
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    predictions, _ = network.forward(grid_points)
    
    # Plot datasets and decision boundaries
    plt.subplot(2, 3, i*3+1)
    plt.scatter(X_data[:,0], X_data[:,1], c=y_data, cmap='coolwarm', alpha=0.8)
    plt.title(f'Raw {name} Data')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, i*3+2) 
    plt.contourf(xx, yy, predictions.reshape(xx.shape), alpha=0.8, cmap='coolwarm', levels=50)
    plt.scatter(X_data[:,0], X_data[:,1], c=y_data, cmap='coolwarm', edgecolors='k')
    plt.title(f'{name} Predictions (Untrained)')
    plt.grid(True, alpha=0.3)
    
    # Analyze layer activations
    plt.subplot(2, 3, i*3+3)
    sample_idx = np.random.randint(0, len(X_data), 50)
    sample_X = X_data[sample_idx]
    _, sample_activations = network.forward(sample_X)
    
    # Show distribution of hidden neurons
    plt.hist(sample_activations['a1'].flatten(), bins=20, alpha=0.7, color='blue')
    plt.title(f'{name} Hidden Layer Activations')
    plt.xlabel('Activation Value')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## 🔄 **Advanced Extensions** 🧪

### A. **Activation Function Composites**
```python
def swish(x):
    """Swish activation: x * sigmoid(x) - Google Brain discovery"""
    return x * sigmoid(x)

def gelu(x):
    """GELU activation - BERT and modern transformers"""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

# Compare modern activations
modern_activations = [('Swish', swish), ('GELU', gelu)]
```

### B. **Network Size Analysis**
```python
def analyze_network_depth():
    """Compare different network architectures"""
    architectures = [
        (1, 'Single Neuron'),
        (5, 'Small Hidden'), 
        (50, 'Large Hidden'),
        (5, 5, 'Two Layers')
    ]
    
    # Implementation for different depths...

analyze_network_depth()
```

---

## 🤔 **Lab Questions & Reflections**

### Conceptual Questions
1. **Neurons:** How does a neuron with ReLU activation behave differently from sigmoid?
2. **Layers:** Why do we need multiple layers instead of one powerful neuron?
3. **Activations:** What happens if we have a network without activation functions?

### Technical Questions  
1. **Computation:** Calculate manually for a 2-2-1 network (inputs: [1,0])
2. **Gradients:** Why is ReLU better than sigmoid for deep networks?
3. **Decision Boundaries:** Draw the boundary for a single neuron with fixed weights.

### Practical Questions
1. **Architecture Selection:** When would you prefer more hidden neurons vs more layers?
2. **Training:** What problems arise if weights start too large?
3. **Applications:** How could you extend this to image recognition?

---

## 📚 **Advanced Resources**
- **3Blue1Brown Neural Net Series:** https://www.3blue1brown.com/topics/neural-networks
- **DeepLearning.AI Neural Networks Specialization:** https://www.coursera.org/learn/neural-networks-deep-learning
- **Stanford CS229 Lecture Notes:** http://cs229.stanford.edu/notes/
- **Towards Data Science NN Tutorials:** https://towardsdatascience.com/neural-networks
- **Neural Computation Research:** Original McCullough & Pitts paper (1943)

---

## ✅ **Lab Completion Checklist**

- [ ] Set up Google Colab environment
- [ ] Implemented single neuron with different activations
- [ ] Compared activation functions on make_moons data  
- [ ] Built and visualized multi-layer network forward pass
- [ ] Created computation graph visualization
- [ ] Experimented on complex datasets (circles, moons)
- [ ] Completed questions and reflections
- [ ] Saved notebook with all visualizations

**Congratulations! 🧠** You've built functional artificial neurons and networks. This mathematical foundation will power all your deep learning adventures!

*Estimated completion time: 3-4 hours*
