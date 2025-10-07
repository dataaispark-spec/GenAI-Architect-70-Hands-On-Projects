# Phase 1: Foundations - Comprehensive Hands-on Lab Exercises 🛠️

## Lab Overview: Building AI/ML Foundation Through Practical Implementation

This comprehensive lab series guides you through 15 hands-on exercises covering the complete Phase 1 foundations. From Python installation to building your first generative AI models - experience the full journey! ⬇️

---

## 🚀 **Week 1 Labs: Introduction to AI & ML (Videos 1-5)**

### Lab 1: Python Installation & AI Exploration (15 minutes) 🎯

### Prerequisites
- Computer with internet access
- Administrator privileges (if on shared computer)

### Step-by-Step Instructions

#### For Windows Users 🪟
1. **Visit Python.org**: Go to [python.org/downloads](https://python.org/downloads)
2. **Download Python 3.11+**: Click "Download the latest version for Windows"
3. **Run Installer**: Double-click the downloaded .exe file
4. **During Installation**:
   - ✅ Check "Add Python to PATH"
   - ✅ "Install launcher for all users"
   - Click "Install Now"
5. **Verify Installation**:
   - Open Command Prompt: `Win + R` → type `cmd` → Enter
   - Type: `python --version` → Should show version like "Python 3.11.x"
   - Type: `pip --version` → Package manager verification

#### For macOS Users 🍎
1. **Install Homebrew** (if not already installed):
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```
2. **Install Python**:
   ```bash
   brew install python
   ```
3. **Verify Installation**:
   ```bash
   python3 --version
   pip3 --version
   ```

#### For Linux Users 🐧
1. **Update package manager**:
   ```bash
   sudo apt update  # Ubuntu/Debian
   # OR
   sudo yum update  # CentOS/RHEL
   ```
2. **Install Python**:
   ```bash
   sudo apt install python3 python3-pip  # Ubuntu
   # OR
   sudo yum install python3 python3-pip  # CentOS
   ```
3. **Verify Installation**:
   ```bash
   python3 --version
   pip3 --version
   ```

---

## ⚙️ **Lab 2: Jupyter Notebook Setup** (10 minutes)

### Why Jupyter?
Jupyter notebooks combine code, visualizations, and explanations - perfect for AI exploration!

### Installation Steps

1. **Open Terminal/Command Prompt**
2. **Install Jupyter**:
   ```bash
   pip install jupyter
   # OR
   pip3 install jupyter
   ```
3. **Launch Jupyter**:
   ```bash
   jupyter notebook
   ```
   - This opens in your browser at `http://localhost:8888`
   - You might see a security token - save it!

### Your First Notebook

1. **Create New Notebook**: Click "New" → "Python 3"
2. **Try Basic Code**:
   ```python
   print("Hello, AI World! 🤖")
   import sys
   print(f"Python version: {sys.version}")
   ```

---

## 🎯 **Lab 3: Exploring AI Demos** (15 minutes)

### Option 1: Simple AI Classification 🧠

Try this in your Jupyter notebook:
```python
# Simple AI simulation - Fruit Classifier
import random

def ai_classifier(fruit_features):
    """
    Simple rule-based 'AI' - pretends to be smart!
    """
    size = fruit_features['size']
    color = fruit_features['color']
    shape = fruit_features['shape']

    # Our 'AI' logic (very basic)
    if color == 'red' and shape == 'round':
        return 'Apple 🍎'
    elif color == 'yellow' and size > 5:
        return 'Banana 🍌'
    elif color == 'orange' and shape == 'round':
        return 'Orange 🍊'
    else:
        return 'Unknown fruit 🤔'

# Test our AI
fruits_to_test = [
    {'size': 3, 'color': 'red', 'shape': 'round'},
    {'size': 6, 'color': 'yellow', 'shape': 'long'},
    {'size': 4, 'color': 'orange', 'shape': 'round'}
]

for fruit in fruits_to_test:
    prediction = ai_classifier(fruit)
    print(f"Fruit features: {fruit}")
    print(f"AI predicts: {prediction}")
    print("---")
```

### Option 2: Text Generation Playground 🎤

```python
# Simple text manipulation as introduction to GenAI
import random

def simple_text_generator(prompt):
    """Basic text extension using patterns"""
    templates = [
        f"{prompt} is truly amazing! 🌟",
        f"Imagine a world where {prompt.lower()} changes everything.",
        f"The future of {prompt.lower()} is bright and full of possibilities!",
        f"{prompt} represents one of humanity's greatest achievements.",
    ]
    return random.choice(templates)

# Test the 'AI'
topics = ["Artificial Intelligence", "Generative AI", "Machine Learning", "Neural Networks"]

for topic in topics:
    generated_text = simple_text_generator(topic)
    print(f"Input: {topic}")
    print(f"Generated: {generated_text}")
    print("---")
```

### Option 3: Online AI Demos 🌐

Explore these free AI tools (no setup needed):
- **ChatGPT**: [chat.openai.com](https://chat.openai.com) - Text generation
- **DALL-E 2**: [openai.com/dall-e-2](https://openai.com/dall-e-2) - Image creation
- **Hugging Face Spaces**: [huggingface.co/spaces](https://huggingface.co/spaces) - Browse AI demos

---

## 🧪 **Lab 4: Reflection Exercise** (5 minutes)

### Questions to Consider:
1. **What impressed you most** about the AI tools you explored?
2. **How accurate** were the simple examples we built?
3. **What are 3 potential applications** of GenAI you can imagine for your field?
4. **What are potential limitations** you noticed?

### Log Your Thoughts:

Create a text file called `course1_reflections.txt` and write:
```
My first AI impressions:
- What I learned today:
- Questions I have:
- Ideas for GenAI applications:
```

### Lab 2: Machine Learning Fundamentals (20 minutes) 📊

Start building real ML models with Scikit-learn:

```python
# Lab 2: Linear Regression Implementation
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Create sample dataset
np.random.seed(42)
X = np.random.rand(100, 1) * 10
y = 2.5 * X.ravel() + 1.5 + np.random.randn(100) * 2

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")
print(f"Model coefficients: {model.coef_[0]:.2f}")
print(f"Model intercept: {model.intercept_:.2f}")

# Visualize
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.title('Linear Regression Prediction')
plt.legend()
plt.show()
```

**Lab Tasks:**
- Modify the data generation parameters
- Add cross-validation
- Implement polynomial features
- Compare with different test sizes

### Lab 3: Neural Network from Scratch (25 minutes) 🧠

Implement a single neuron in Python:

```python
# Lab 3: Single Neuron Implementation
import numpy as np

class SingleNeuron:
    def __init__(self, input_size):
        # Initialize weights and bias randomly
        self.weights = np.random.randn(input_size)
        self.bias = np.random.randn(1)
        self.learning_rate = 0.01

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def feedforward(self, inputs):
        # Weighted sum + bias
        z = np.dot(inputs, self.weights) + self.bias
        # Activation
        return self.sigmoid(z)

    def train(self, inputs, targets, epochs=100):
        for epoch in range(epochs):
            for x, y in zip(inputs, targets):
                # Forward pass
                output = self.feedforward(x)

                # Calculate error
                error = y - output

                # Backward pass (gradient descent)
                delta = error * output * (1 - output)

                # Update weights and bias
                self.weights += self.learning_rate * delta * x
                self.bias += self.learning_rate * delta

# Test the neuron
neuron = SingleNeuron(input_size=2)

# OR gate training data
training_data = [
    ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 1)
]

# Train
for inputs, target in training_data:
    neuron.train([inputs], [target], epochs=1000)

# Test
print("OR Gate Results:")
for inputs, target in training_data:
    output = neuron.feedforward(inputs)
    print(f"Input: {inputs} -> Output: {output[0]:.3f} (Target: {target})")
```

### Lab 4: Deep Learning with TensorFlow/Keras (30 minutes) 🏗️

Build a neural network for MNIST digit recognition:

```python
# Lab 4: MNIST Neural Network with Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# Preprocess data
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

# Build model
model = keras.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train
history = model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"\nTest accuracy: {test_acc:.4f}")

# Visualize training
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Model Training Progress')
plt.legend()
plt.show()

# Make predictions on sample
sample_images = X_test[:5]
predictions = model.predict(sample_images)

print("Sample Predictions:")
for i, prediction in enumerate(predictions):
    predicted_digit = np.argmax(prediction)
    confidence = np.max(prediction)
    print(f"Image {i+1}: Predicted digit {predicted_digit} with {confidence:.2f} confidence")
```

---

## 🎨 **Week 2 Labs: Generative Models (Videos 6-10)**

### Lab 5: Variational Autoencoders (VAEs) (45 minutes) 🎨

Build and train your first generative model:

```python
# Lab 5: VAE for Fashion-MNIST Generation
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# Define VAE architecture
class VAE(keras.Model):
    def __init__(self, latent_dim=2):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = keras.Sequential([
            layers.Flatten(input_shape=(28, 28)),
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
        ])

        # Latent space layers
        self.z_mean = layers.Dense(latent_dim)
        self.z_log_var = layers.Dense(latent_dim)

        # Decoder
        self.decoder = keras.Sequential([
            layers.Dense(64, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(28*28, activation='sigmoid'),
            layers.Reshape((28, 28))
        ])

    def encode(self, x):
        x_encoded = self.encoder(x)
        z_mean = self.z_mean(x_encoded)
        z_log_var = self.z_log_var(x_encoded)
        return z_mean, z_log_var

    def reparameterize(self, z_mu, z_log_var):
        epsilon = tf.random.normal(shape=tf.shape(z_mu))
        return z_mu + tf.exp(0.5 * z_log_var) * epsilon

    def decode(self, z):
        return self.decoder(z)

    def call(self, x):
        z_mean, z_log_var = self.encode(x)
        z = self.reparameterize(z_mean, z_log_var)
        x_reconstructed = self.decode(z)
        return x_reconstructed, z_mean, z_log_var

# Loss function for VAE
def vae_loss(x, x_reconstructed, z_mean, z_log_var):
    # Reconstruction loss
    reconstruction_loss = tf.reduce_sum(
        keras.losses.binary_crossentropy(x, x_reconstructed), axis=[1, 2]
    )
    reconstruction_loss = tf.reduce_mean(reconstruction_loss)

    # KL divergence
    kl_loss = -0.5 * tf.reduce_sum(
        1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1
    )
    kl_loss = tf.reduce_mean(kl_loss)

    return reconstruction_loss + kl_loss

# Load and preprocess data
(x_train, _), (x_test, _) = keras.datasets.fashion_mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# Train VAE
vae = VAE()
vae.compile(optimizer='adam')

vae.fit(x_train, x_train, epochs=10, batch_size=64, validation_data=(x_test, x_test))

# Generate new images
latent_dim = 2
random_latent_vectors = np.random.normal(size=(16, latent_dim))
generated_images = vae.decoder(random_latent_vectors)

# Visualize generated images
fig, axes = plt.subplots(4, 4, figsize=(8, 8))
axes = axes.flatten()

for i in range(16):
    axes[i].imshow(generated_images[i], cmap='gray')
    axes[i].axis('off')

plt.tight_layout()
plt.show()
```

**Advanced Tasks:**
- Experiment with different latent dimensions
- Add convolutional layers to encoder/decoder
- Generate interpolations between images
- Try conditional VAEs

### Lab 6: Generative Adversarial Networks (GANs) (45 minutes) ⚡

Implement and train a DCGAN:

```python
# Lab 6: DCGAN for MNIST Generation
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

def make_generator_model():
    model = keras.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))

    return model

def make_discriminator_model():
    model = keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

# Define loss functions
cross_entropy = keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# Optimizers
generator_optimizer = keras.optimizers.Adam(1e-4)
discriminator_optimizer = keras.optimizers.Adam(1e-4)

@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss

def train(dataset, epochs):
    for epoch in range(epochs):
        for image_batch in dataset:
            gen_loss, disc_loss = train_step(image_batch)

        if epoch % 10 == 0:
            generate_and_save_images(generator, epoch + 1, seed)

        print(f'Epoch {epoch + 1}, Generator Loss: {gen_loss:.4f}, Discriminator Loss: {disc_loss:.4f}')

# Generate and plot images
def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)
    fig = plt.figure(figsize=(4,4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')
    plt.suptitle(f'Epoch {epoch}')
    plt.show()

# Load and prepare data
(x_train, _), (_, _) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
x_train = (x_train - 127.5) / 127.5  # Normalize to [-1, 1]

BUFFER_SIZE = 60000
BATCH_SIZE = 256
train_dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# Create models
generator = make_generator_model()
discriminator = make_discriminator_model()

noise_dim = 100
num_examples_to_generate = 16
seed = tf.random.normal([num_examples_to_generate, noise_dim])

# Train the GAN
train(train_dataset, epochs=50)
```

---

## 📊 **Week 3 Labs: Professional Skills (Videos 11-15)**

### Lab 7: AI Ethics Analysis Toolkit (30 minutes) ⚖️

```python
# Lab 7: Bias Detection and Mitigation
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def analyze_dataset_bias(data_path, sensitive_attribute='gender'):
    """
    Analyze potential bias in training datasets
    """
    # Load sample biased dataset
    data = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 1000),
        'feature2': np.random.normal(0, 1, 1000),
        'gender': np.random.choice(['M', 'F'], 1000, p=[0.6, 0.4]),  # Biased gender ratio
        'target': np.random.randint(0, 2, 1000)
    })

    # Analyze distribution
    print("Dataset Bias Analysis")
    print("=" * 50)
    print(data['gender'].value_counts(normalize=True))
    print("\nFeature correlations with sensitive attribute:")

    for col in ['feature1', 'feature2', 'target']:
        male_mean = data[data['gender'] == 'M'][col].mean()
        female_mean = data[data['gender'] == 'F'][col].mean()
        print(".3f"

    # Test fairness
    X = data[['feature1', 'feature2']]
    y = data['target']

    # Split by gender
    X_male = data[data['gender'] == 'M'][['feature1', 'feature2']]
    y_male = data[data['gender'] == 'M']['target']
    X_female = data[data['gender'] == 'F'][['feature1', 'feature2']]
    y_female = data[data['gender'] == 'F']['target']

    # Train separate models
    model_all = LogisticRegression().fit(X, y)
    model_male = LogisticRegression().fit(X_male, y_male)
    model_female = LogisticRegression().fit(X_female, y_female)

    # Test on neutral data
    X_test = np.random.normal(0, 1, (100, 2))
    probabilities_all = model_all.predict_proba(X_test)[:, 1]
    probabilities_male = model_male.predict_proba(X_test)[:, 1]
    probabilities_female = model_female.predict_proba(X_test)[:, 1]

    print("
Fairness Analysis:")
    print(f"All group: Mean probability = {probabilities_all.mean():.3f}")
    print(f"Male subgroup: Mean probability = {probabilities_male.mean():.3f}")
    print(f"Female subgroup: Mean probability = {probabilities_female.mean():.3f}")

    bias_ratio = abs(probabilities_male.mean() - probabilities_female.mean())
    print(f"Bias ratio: {bias_ratio:.3f}")

    if bias_ratio > 0.1:
        print("⚠️ Significant bias detected! Consider mitigation techniques.")
    else:
        print("✅ Bias within acceptable range.")

    return bias_ratio

# Run analysis
bias_level = analyze_dataset_bias("sample_data.csv")
```

---

## ⏳ **Advanced Labs (Optional)**

### Lab 8: Custom Data Processing Pipeline 📊
### Lab 9: Performance Benchmarking Suite 📈
### Lab 10: Model Interpretability Tools 🔍

---

## 📚 **Master Lab Completion Checklist**

- [ ] **Week 1**: Python setup, ML basics, NN fundamentals
- [ ] **Week 2**: Deep learning, VAEs, GANs implementation
- [ ] **Week 3**: Ethics analysis, evaluation metrics, capstone
- [ ] **Documentation**: All experiments documented in Jupyter notebooks
- [ ] **GitHub Portfolio**: All code pushed with proper README files

**🎉 Phase 1 Complete!** You've mastered the foundations of AI/ML and built your first generative models!

*Next: Phase 2 - Core GenAI Concepts with LLMs and Transformers!*

*Total lab time: ~15 hours of hands-on coding and experimentation*
