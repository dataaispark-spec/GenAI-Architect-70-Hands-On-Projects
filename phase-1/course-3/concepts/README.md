# Course 3: Neural Networks Fundamentals 🧠

## Overview
This course introduces the building blocks of neural networks, the foundation of modern deep learning. We'll explore artificial neurons, network architecture, and activation functions that enable machines to learn complex patterns, inspired by the human brain.

## Key Learning Objectives
- Understand artificial neuron mathematics and computation
- Learn network architectures: input, hidden, output layers  
- Master activation functions and their roles
- Apply neural networks to basic problems
- Gain intuition for how deep learning models work

## Core Topics

### 1. **Artificial Neurons**
- **Biological Inspiration:** Neurons fire when input exceeds threshold (McCullough & Pitts, 1943 paper)
- **Artificial Neuron:** Inputs * weights + bias = weighted sum, then activation function
- **Math Formulation:** y = σ(∑(wᵢxᵢ + b)) where σ is activation

**Analogy from 3Blue1Brown:** Neurons as voting booths - inputs vote with weights, threshold decides output.

### 2. **Activation Functions** 
- **Sigmoid (σ):** S-shaped curve, 0-1 range (logistic curve from 1840s)
- **Tanh:** Centered sigmoid, -1 to 1, helps vanishing gradients
- **ReLU (Rectified Linear Unit):** max(0,x) - most popular, prevents vanishing gradients (Hahnloser et al., 2000)
- **Leaky ReLU:** Slight negative slope to avoid dead neurons
- **Softmax:** Converts to probabilities for multi-class (inspired by statistical mechanics)

**Why Needed:** Without activation, network collapses to linear regression (universal approximation theorem).

### 3. **Network Architecture**
- **Layer Types:**
  - Input Layer: Raw data (pixels, text tokens)
  - Hidden Layers: Feature extraction, learned representations
  - Output Layer: Final predictions (probability distributions)
- **Feedforward Process:** Data flows from input to output (no cycles)
- **Depth vs Width:** Deeper networks learn hierarchical features (Krizhevsky et al., 2012 AlexNet)

### 4. **Training Basics**
- **Forward Pass:** Compute predictions through network layers
- **Loss Functions:** Measures prediction error (MSE for regression, Cross-Entropy for classification)
- **Backward Pass:** Gradient descent updates weights (backpropagation, Rumelhart et al., 1986)
- **Optimization:** Adam optimizer combines momentum and adaptive learning rates (Kingma & Ba, 2015)

### 5. **Key Concepts**
- **Non-Linearity:** Transforms enable complex decision boundaries
- **Universality:** Single hidden layer can approximate any function (universal approximation theorem)
- **Gradient Vanishing/Exploding:** Common issues solved by careful weight initialization (Glorot & Bengio, 2010)

## Technical Implementation
```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

# Simple neuron simulation
def neuron(inputs, weights, bias, activation):
    z = np.dot(weights, inputs) + bias
    return activation(z)

# Example: 2-input neuron with ReLU
inputs = np.array([0.5, -0.2])
weights = np.array([0.1, 0.3])
bias = 0.1

output = neuron(inputs, weights, bias, relu)
print(f"Neuron output: {output:.3f}")  # ~0.020
```

## Real-World Applications
- **Image Recognition:** Convolutional layers process pixels (CNN breakthrough by LeCun et al., 1998)
- **Speech Recognition:** RNNs handle sequential audio data
- **Recommendation Systems:** Embedding layers learn user-item relationships

## Challenges & Solutions
- **Overfitting:** Dropout regularization (Srivastava et al., 2014) randomly deactivates neurons during training
- **Vanishing Gradients:** Modern architectures use skip connections (ResNet, He et al., 2016)
- **Computational Cost:** Batch normalization (Ioffe & Szegedy, 2015) stabilizes training

## Architect Perspective  
- **Model Selection:** Start simple, add complexity gradually (Occam's razor in ML)
- **Hyperparameter Tuning:** Grid search for learning rate, batch size (key for performance)
- **Hybrid Approximations:** Combine neural network insights with domain knowledge
- **Ethics:** Neural networks can amplify biases if trained on biased data

## Learning Outcomes
- ✅ Explain neuron computation and activation functions
- ✅ Design basic neural network architectures  
- ✅ Implement forward pass calculations
- ✅ Understand backpropagation fundamentals
- ✅ Apply neural networks to classification tasks

## Resources
- **3Blue1Brown YouTube:** Brilliant visual explanations of neural networks
- **DeepLearning.AI Neural Networks Course:** Free specialization on Coursera
- **Stanford CS229 Notes:** Mathematical derivations (ngrok.stanford.edu)
- **Towards Data Science Articles:** Practical implementation tutorials
- **arXiv Papers:** "Understanding Deep Learning Requires Rethinking Generalization" (Zhang et al.)

---

*Estimated Study Time: 8-10 hours | Prerequisites: Linear Algebra Basics*

## Next Up
**Course 4: Deep Learning Essentials** - Convolutional and recurrent networks for advanced AI.

*Connecting neurons into powerful computing systems! ⚡*
