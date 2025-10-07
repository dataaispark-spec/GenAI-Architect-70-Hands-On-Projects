# Course 1: Hands-on Lab Exercises 🛠️

## Lab Overview: Your First AI Setup & Exploration

This lab introduces you to Python setup and explores basic AI concepts through simple demonstrations. Perfect for beginners with no coding experience! ⬇️

---

## 🚀 **Lab 1: Python Installation & AI Exploration** (15 minutes)

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

---

## 📚 **Additional Resources**

- **Python Documentation**: [python.org](https://python.org)
- **Jupyter Tutorials**: [jupyter.org/try](https://jupyter.org/try)
- **AI Ethics Discussion**: Consider the implications of AI in society

---

## ✅ **Lab Completion Checklist**

- [ ] Python installed and verified
- [ ] Jupyter notebook running
- [ ] Ran the simple AI examples
- [ ] Created reflection document
- [ ] Explored at least 1 online AI demo

**Congratulations! 🎉** You've taken your first steps into the AI world. Ready for Course 2: Machine Learning Basics!

*Estimated completion time: 45 minutes*
