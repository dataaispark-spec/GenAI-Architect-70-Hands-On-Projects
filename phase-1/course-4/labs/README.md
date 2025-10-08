# Course 4: Deep Learning Essentials - Hands-on Labs 🧠🛠️

## Lab Overview: CNN & RNN Training with TensorFlow/Keras 🔬

This comprehensive lab builds complete deep learning systems from scratch, training CNNs on MNIST for vision tasks and RNNs for sequence prediction. You'll master optimizers, regularization, and debugging techniques across multiple examples.

---

## Prerequisites
- Google account for Colab access
- Basic TensorFlow/Keras knowledge
- Familiarity with NumPy and matplotlib
- Compute resources (enable GPU in Colab)

## 🏗️ **Lab Objectives**
- Build and train CNNs for image classification
- Implement RNN/LSTM networks for sequential data
- Compare different optimizers performance
- Apply regularization and augmentation techniques
- Debug and optimize deep learning pipelines

## 📋 **Step-by-Step Instructions**

### Step 1: Setup Environment 🌟
Create `course4_deep_learning.ipynb` in Google Colab.

Import essential libraries and configure GPU:
```python
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Enable GPU (if available)
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {tf.config.list_physical_devices('GPU')}")
```

### Step 2: MNIST CNN Classification 🖼️
Build a complete CNN pipeline for digit recognition:

```python
# Load and preprocess MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize and reshape
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# One-hot encode labels
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

print(f"Training data shape: {x_train.shape}, Labels shape: {y_train.shape}")

# Visualize sample images
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(x_train[i].reshape(28, 28), cmap='gray')
    plt.title(f"Label: {np.argmax(y_train[i])}")
    plt.axis('off')
plt.show()
```

Build the CNN architecture:
```python
def create_cnn(optimizer='adam', dropout=0.2):
    model = keras.Sequential([
        # First convolutional block
        keras.layers.Conv2D(32, (3, 3), padding='same', input_shape=(28, 28, 1)),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        
        keras.layers.Conv2D(32, (3, 3)),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(dropout),
        
        # Second convolutional block
        keras.layers.Conv2D(64, (3, 3), padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        
        keras.layers.Conv2D(64, (3, 3)),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(dropout),
        
        # Fully connected layers
        keras.layers.Flatten(),
        keras.layers.Dense(128),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        keras.layers.Dropout(dropout),
        
        keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy', 'top_k_categorical_accuracy']
    )
    
    return model

# Create model
cnn_model = create_cnn()
cnn_model.summary()
```

Train and evaluate:
```python
# Training callbacks
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy', 
        patience=5, 
        restore_best_weights=True
    ),
    keras.callbacks.ModelCheckpoint(
        'best_cnn.keras', 
        monitor='val_accuracy', 
        save_best_only=True
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5, 
        patience=3, 
        min_lr=1e-6
    )
]

# Train model
history = cnn_model.fit(
    x_train, y_train,
    batch_size=128,
    epochs=50,
    validation_split=0.1,
    callbacks=callbacks,
    verbose=1
)

# Evaluate on test set
test_loss, test_acc, test_top5 = cnn_model.evaluate(x_test, y_test, verbose=0)
print(".4f" test_top5)

# Plot training history
plt.figure(figsize=(15, 5))

# Accuracy
plt.subplot(1, 3, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss
plt.subplot(1, 3, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Learning Rate
plt.subplot(1, 3, 3)
# Learning rate history (need to track manually or use custom callback)
plt.plot(history.epoch, [0.001] * len(history.epoch), label='Default LR')
plt.title('Learning Rate Schedule')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.legend()

plt.tight_layout()
plt.show()
```

### Step 3: Optimizer Comparison ⚡
Compare different optimizers performance:

```python
# Split data for fair comparison
x_sub = x_train[:10000]
y_sub = y_train[:10000]
x_sub_test = x_test[:2000]
y_sub_test = y_test[:2000]

optimizers = {
    'SGD': keras.optimizers.SGD(learning_rate=0.01),
    'Momentum': keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
    'RMSProp': keras.optimizers.RMSprop(learning_rate=0.001),
    'Adam': keras.optimizers.Adam(learning_rate=0.001),
    'AdamW': keras.optimizers.AdamW(learning_rate=0.001, weight_decay=1e-4)
}

optimizer_results = {}

for name, optimizer in optimizers.items():
    print(f"\\nTesting {name} optimizer:")
    
    model = create_cnn(optimizer=optimizer, dropout=0.3)
    
    history = model.fit(
        x_sub, y_sub,
        batch_size=64,
        epochs=10,
        validation_data=(x_sub_test, y_sub_test),
        verbose=0,
        callbacks=[keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)]
    )
    
    final_val_acc = max(history.history['val_accuracy'])
    optimizer_results[name] = {
        'model': model,
        'history': history,
        'final_val_acc': final_val_acc
    }
    
    print(".3f")

# Plot optimizer comparison
plt.figure(figsize=(10, 6))
for name, result in optimizer_results.items():
    plt.plot(result['history'].history['val_accuracy'], label=name)

plt.title('Optimizer Comparison on MNIST')
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### Step 4: RNN for Time Series 📈
Build LSTM networks for sequence prediction:

```python
# Generate synthetic time series data
def generate_time_series(batch_size=1000, n_steps=50):
    """Generate sine wave with trend and noise"""
    time = np.linspace(0, 10, n_steps)
    trend = 0.5 * time
    noise = np.random.normal(0, 0.3, (batch_size, n_steps))
    
    series = np.sin(time)[np.newaxis, :] * np.ones((batch_size, 1)) + trend[np.newaxis, :] + noise
    
    return series

# Create datasets
n_steps = 50
batch_size = 10000

# Training data
series = generate_time_series(batch_size, n_steps)
X_train_ts = series[:, :n_steps-1]  # Input: first 49 timesteps
y_train_ts = series[:, 1:]          # Target: next timestep (forecasting)

# Split into sequences
X_train_ts = X_train_ts.reshape(batch_size, n_steps-1, 1)
y_train_ts = y_train_ts.reshape(batch_size, n_steps-1, 1)

# Test data
test_series = generate_time_series(1000, n_steps)
X_test_ts = test_series[:, :n_steps-1].reshape(1000, n_steps-1, 1)
y_test_ts = test_series[:, 1:].reshape(1000, n_steps-1, 1)

# Visualize sample sequence
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(series[0], 'b-', label='Full sequence')
plt.plot(X_train_ts[0, :, 0], 'r--', label='Input sequence')
plt.plot(range(1, n_steps), y_train_ts[0, :, 0], 'g--', label='Target sequence')
plt.legend()
plt.title('Time Series Forecast Setup')
plt.grid(True, alpha=0.3)

plt.subplot(2, 1, 2)
plt.plot(series[0][:10], 'b.', markersize=8, label='Input points')
plt.plot(range(1, 11), y_train_ts[0][:10, 0], 'g.', markersize=8, label='Target points')
plt.legend()
plt.title('Zoomed Forecast Example')
plt.grid(True, alpha=0.3)
plt.show()

print(f"Training shapes - X: {X_train_ts.shape}, Y: {y_train_ts.shape}")
```

Build and train RNN model:
```python
def create_rnn_model(units=64, layers=1, optimizer='adam'):
    model = keras.Sequential()
    
    # First layer
    model.add(keras.layers.LSTM(units, input_shape=(None, 1), return_sequences=True))
    
    # Additional LSTM layers
    for _ in range(layers - 1):
        model.add(keras.layers.LSTM(units, return_sequences=True))
    
    # Output layer (predicts next value)
    model.add(keras.layers.TimeDistributed(keras.layers.Dense(1)))  # Output for each timestep
    
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model

# Create RNN models
simple_rnn = create_rnn_model(units=32, layers=1)    # Simple LSTM
deep_rnn = create_rnn_model(units=32, layers=3)     # Deeper LSTM
gru_rnn = keras.Sequential([        # GRU alternative
    keras.layers.GRU(64, input_shape=(None, 1), return_sequences=True),
    keras.layers.TimeDistributed(keras.layers.Dense(1))
])
gru_rnn.compile(optimizer='adam', loss='mse', metrics=['mae'])

models = {
    'Simple LSTM': simple_rnn,
    'Deep LSTM': deep_rnn,
    'GRU': gru_rnn
}

# Train and compare
rnn_results = {}

for name, model in models.items():
    print(f"\\nTraining {name}...")
    
    history = model.fit(
        X_train_ts[:2000], y_train_ts[:2000],  # Smaller sample for speed
        epochs=30,
        batch_size=128,
        validation_split=0.1,
        verbose=0,
        callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
    )
    
    test_loss, test_mae = model.evaluate(X_test_ts, y_test_ts, verbose=0)
    
    rnn_results[name] = {
        'model': model,
        'history': history,
        'test_loss': test_loss,
        'test_mae': test_mae
    }
    
    print(".3f" test_mae)

# Visualize predictions
plt.figure(figsize=(12, 8))

for i, (name, result) in enumerate(rnn_results.items(), 1):
    plt.subplot(2, 3, i)
    
    # Predict on test set
    predictions = result['model'].predict(X_test_ts[:5])
    
    for j in range(3):  # Show 3 examples
        plt.plot(y_test_ts[j, :, 0], 'b-', alpha=0.7, label='Actual' if j==0 else "")
        plt.plot(predictions[j, :, 0], 'r--', alpha=0.7, label='Predicted' if j==0 else "")
    
    plt.title(f'{name} Predictions')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 4)
for name, result in rnn_results.items():
    plt.plot(result['history'].history['val_loss'], label=name)
plt.title('Validation Loss Comparison')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 5)
for name, result in rnn_results.items():
    plt.plot(result['history'].history['val_mae'], label=name)
plt.title('Validation MAE Comparison')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 6)
names = list(rnn_results.keys())
maes = [result['test_mae'] for result in rnn_results.values()]
plt.bar(names, maes, color=['blue', 'green', 'orange'])
plt.title('Test MAE Comparison')
plt.ylabel('Mean Absolute Error')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Step 5: Data Augmentation & Regularization 🛠️
Enhance CNN with advanced techniques:

```python
# Data augmentation for MNIST
data_augmentation = keras.Sequential([
    keras.layers.experimental.preprocessing.RandomRotation(0.1),
    keras.layers.experimental.preprocessing.RandomZoom(0.1),
    keras.layers.experimental.preprocessing.RandomContrast(0.1),
])

# Create augmented model
def create_augmented_model():
    # Add augmentation as first layer
    augmented_model = keras.Sequential([
        data_augmentation,
        keras.layers.Normalization(input_shape=(28, 28, 1))
    ])
    
    # Add CNN layers (same as before)
    augmented_model.add(keras.layers.Conv2D(32, (3, 3), padding='same'))
    augmented_model.add(keras.layers.BatchNormalization())
    augmented_model.add(keras.layers.ReLU())
    
    augmented_model.add(keras.layers.Conv2D(32, (3, 3)))
    augmented_model.add(keras.layers.BatchNormalization())
    augmented_model.add(keras.layers.ReLU())
    augmented_model.add(keras.layers.MaxPooling2D((2, 2)))
    augmented_model.add(keras.layers.Dropout(0.3))
    
    augmented_model.add(keras.layers.Conv2D(64, (3, 3), padding='same'))
    augmented_model.add(keras.layers.BatchNormalization())
    augmented_model.add(keras.layers.ReLU())
    
    augmented_model.add(keras.layers.Conv2D(64, (3, 3)))
    augmented_model.add(keras.layers.BatchNormalization())
    augmented_model.add(keras.layers.ReLU())
    augmented_model.add(keras.layers.MaxPooling2D((2, 2)))
    augmented_model.add(keras.layers.Dropout(0.3))
    
    augmented_model.add(keras.layers.Flatten())
    augmented_model.add(keras.layers.Dense(128))
    augmented_model.add(keras.layers.BatchNormalization())
    augmented_model.add(keras.layers.ReLU())
    augmented_model.add(keras.layers.Dropout(0.3))
    augmented_model.add(keras.layers.Dense(10, activation='softmax'))
    
    return augmented_model

# Train with augmentation
augmented_model = create_augmented_model()
augmented_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001, weight_decay=1e-4),  # AdamW
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Fit normalization layer
augmented_model.layers[1].adapt(x_train)

# Train augmented model
augmented_history = augmented_model.fit(
    x_train, y_train,
    batch_size=128,
    epochs=30,
    validation_split=0.1,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint('best_augmented.keras', save_best_only=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
    ]
)

# Compare original vs augmented
original_model = create_cnn(dropout=0.3)
original_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
original_history = original_model.fit(
    x_train[:10000], y_train[:10000],  # Smaller sample to match
    epochs=30,
    validation_split=0.1,
    verbose=0,
    callbacks=[keras.callbacks.EarlyStopping(patience=5)]
)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(original_history.history['val_accuracy'], label='Original')
plt.plot(augmented_history.history['val_accuracy'], label='With Augmentation')
plt.title('Validation Accuracy: Original vs Augmented')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(original_history.history['val_loss'], label='Original')
plt.plot(augmented_history.history['val_loss'], label='With Augmentation')
plt.title('Validation Loss: Original vs Augmented')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Test on augmented data
aug_orig, acc_orig = original_model.evaluate(x_test, y_test, verbose=0)
aug_acc, acc_aug = augmented_model.evaluate(x_test, y_test, verbose=0)

print(".4f" acc_aug)
```

### Step 6: Going Beyond: Modern Techniques 🚀
Explore cutting-edge deep learning approaches:

```python
# Transfer learning with pre-trained CNN
base_model = keras.applications.MobileNetV2(
    input_shape=(96, 96, 3),  # Resize MNIST to fit
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # Freeze pre-trained weights

# Add classification head for MNIST
transfer_model = keras.Sequential([
    keras.layers.Input(shape=(28, 28, 1)),
    # Convert grayscale to RGB by replicating channels
    keras.layers.Conv2D(3, (3, 3), padding='same', activation='relu'),  # Grayscale to RGB
    keras.layers.UpSampling2D((96//28, 96//28)),  # Upsample to MobileNet input size
    base_model,
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation='softmax')
])

# Convert MNIST to RGB
x_train_rgb = np.repeat(x_train, 3, axis=-1)  # Gray → RGB by replication
x_test_rgb = np.repeat(x_test, 3, axis=-1)

# Upsample to 96x96 for MobileNet
x_train_up = tf.image.resize(x_train_rgb, (96, 96)).numpy()
x_test_up = tf.image.resize(x_test_rgb, (96, 96)).numpy()

transfer_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Fine-tune
transfer_history = transfer_model.fit(
    x_train_up, y_train,
    epochs=10,
    batch_size=128,
    validation_split=0.1,
    callbacks=[keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)]
)

# Unfreeze some layers for fine-tuning
base_model.trainable = True
for layer in base_model.layers[:-10]:  # Freeze first layers
    layer.trainable = False

transfer_model.compile(optimizer=keras.optimizers.Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
transfer_model.fit(x_train_up, y_train, epochs=5, validation_split=0.1, verbose=0)
```

---

## 🔄 **Advanced Extensions** 🧪

### A. **Mixed Precision Training**
```python
from tensorflow.keras.mixed_precision import experimental as mixed_precision

# Enable mixed precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)

# Build model with float16 automatically
mixed_model = create_cnn()
mixed_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

### B. **Custom Training Loop with tf.function**
```python
@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        preds = model(x)
        loss = tf.keras.losses.categorical_crossentropy(y, preds)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return loss

# Custom training loop
for epoch in range(10):
    for x_batch, y_batch in train_dataset:
        loss = train_step(x_batch, y_batch)
    print(f"Epoch {epoch}, Loss: {loss.numpy()}")
```

### C. **Model Interpretability/Explanability**
```python
# Install if needed: !pip install shap
import shap

# Explain model predictions
explainer = shap.Explainer(cnn_model.predict, x_test[:100])
shap_values = explainer(x_test[:10])

# Visualize feature importance
shap.image_plot(shap_values, x_test[:10])
```

---

## 🤔 **Lab Questions & Reflections**

### Technical Questions
1. **CNNs:** Why do CNNs outperform fully-connected networks on images?
2. **RNNs:** What happens if we unroll a very long RNN sequence?
3. **Optimizers:** Why does Adam often outperform SGD in practice?
4. **Regularization:** How does dropout prevent overfitting?
5. **BatchNorm:** Why does it improve training stability?

### Practical Questions
1. **Architecture Choices:** When would you choose more conv layers vs deeper FC layers?
2. **Transfer Learning:** What datasets work well for computer vision pre-training?
3. **Debugging:** How can you detect and fix vanishing gradients?
4. **Deployment:** What optimizations are needed for mobile CNNs?

### Research-Oriented Questions
1. **Scaling Laws:** How does model performance change with data/compute?
2. **Attention Mechanisms:** Why did transformers replace RNNs?
3. **Neural Architecture Search:** How can we automate architecture design?

---

## 📚 **Extended Resources**
- **TensorFlow 2 Advanced Tutorials**: https://www.tensorflow.org/tutorials
- **Keras Examples Repository**: https://github.com/keras-team/keras-io
- **Papers with Code Deep Learning**: https://paperswithcode.com/area/deep-learning
- **Stanford CS230 Course**: Deep Learning for Computer Vision
- **DeepLearning.AI TensorFlow Specialization**: Practical framework mastery

---

## ✅ **Lab Completion Checklist**

- [ ] Set up TensorFlow/Keras in Colab with GPU
- [ ] Built and trained CNN on MNIST with callbacks
- [ ] Compared different optimizers (SGD, Adam, RMSProp, etc.)
- [ ] Implemented RNN/LSTM/GRU for time series prediction
- [ ] Applied data augmentation and regularization techniques
- [ ] Explored transfer learning with pre-trained models
- [ ] Experimented with modern techniques (mixed precision, custom training)
- [ ] Answered technical and practical questions
- [ ] Saved all models and visualizations

**Master of Deep Learning! 🧠 You'll now handle real-world datasets and architectures with confidence.**

*Estimated completion time: 4-5 hours*
