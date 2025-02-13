# 📚 Machine Learning & Deep Learning Math

This folder contains the **mathematical foundations of ML & AI**, with clear implementations in Python. Below, we visualize and explain essential formulas used in **optimization, activation functions, loss calculations, and more**.

---

## 🔢 1️⃣ Gradient Descent (Optimization Algorithm)

### **Formula:**
$$
\LARGE \theta_{t+1} = \theta_t - \eta \nabla J(\theta_t)
$$

- $`\theta_{t+1}`$ → Updated parameter.
- $`\theta_t`$ → Model parameters at step \( t \).  
- $`\eta`$ → Learning rate, controlling step size.  
- $`\nabla J(\theta_t)`$ → Gradient of the cost function $J(\theta)$ with respect to $\theta\$ at time step $\t$.  

📌 **Why It Matters?**
- Adjusts weights in ML/DL models to **minimize the loss function**.  
- The backbone of training algorithms in **neural networks, linear regression, etc.**  

🔗 **Python Implementation:** [`gradient_descent.py`](gradient_descent.py)

---

## 🧠 2️⃣ Softmax Function (Classification)

### **Formula:**
$$
\LARGE \text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{n} e^{z_j}}
$$

- $`z_i`$ → Raw model output (logits).  
- $`e^{z_i}`$ → Exponential transformation ensuring positive values.  
- $`\sum e^{z_j}`$ → Normalization factor ensuring probabilities sum to 1.  

📌 **Why It Matters?**
- Converts raw model outputs into **probabilities**.  
- Used in **multi-class classification (e.g., MNIST digit classification)**.  

🔗 **Python Implementation:** [`softmax.py`](softmax.py)

---

## 🎯 3️⃣ Cross-Entropy Loss (For Classification Models)

### **Formula:**
$$
\LARGE \mathcal{L} = -\sum_{i=1}^{n} y_i \log(\hat{y_i})
$$

- **\( y_i \)** → True class label (ground truth).  
- **\( \hat{y_i} \)** → Predicted probability for class \( i \).  

📌 **Why It Matters?**
- Measures how well predicted probabilities match true labels.  
- Common in **logistic regression, CNNs, NLP models**.  

🔗 **Python Implementation:** [`cross_entropy.py`](cross_entropy.py)

---

## 🔄 4️⃣ Adam Optimizer (Advanced Gradient Descent)

### **Formula:**
$$
\LARGE \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{v_t} + \epsilon} m_t
$$

- **\( \theta_t \)** → Model parameters at step \( t \).  
- **\( m_t \)** → First moment estimate (mean of gradients).  
- **\( v_t \)** → Second moment estimate (variance of gradients).  
- **\( \eta \)** → Learning rate.  
- **\( \epsilon \)** → Small constant to avoid division by zero.  

📌 **Why It Matters?**
- Adaptive learning rate method, used in **CNNs, RNNs, Transformers**.  
- Faster and more stable than standard **gradient descent**.  

🔗 **Python Implementation:** [`adam_optimizer.py`](adam_optimizer.py)

---

## 📈 5️⃣ Convolution Operation (Feature Extraction in CNNs)

### **Formula:**
$$
\LARGE O(i, j) = \sum_m \sum_n I(i+m, j+n) \cdot K(m, n)
$$

- **\( O(i, j) \)** → Output feature map at position \( i, j \).  
- **\( I(i+m, j+n) \)** → Input image pixel values affected by the filter.  
- **\( K(m, n) \)** → Kernel (filter) values.  

📌 **Why It Matters?**
- Core operation in **image processing & CNNs**.  
- Detects edges, patterns, and textures in **computer vision**.  

🔗 **Python Implementation:** [`convolution.py`](convolution.py)

---

## 🤖 6️⃣ Transformer Attention (NLP Models like BERT, GPT)

### **Formula:**
$$
\LARGE \text{Attention}(Q, K, V) = \text{softmax} \left( \frac{QK^T}{\sqrt{d_k}} \right) V
$$

- **\( Q, K, V \)** → Query, Key, and Value matrices.  
- **\( d_k \)** → Dimensionality of key vectors (scaling factor).  
- **\( QK^T \)** → Dot product of queries and keys to compute attention scores.  
- **\( \text{softmax} \)** → Normalization to ensure values sum to 1.  

📌 **Why It Matters?**
- Used in **transformers (GPT, BERT, T5)** for NLP.  
- Enables models to **focus on relevant words** in a sentence.  

🔗 **Python Implementation:** [`transformer_attention.py`](transformer_attention.py)

---

## 📌 Summary
This folder contains **core mathematical operations used in ML & AI**. Each `.py` file provides:  
✅ **A clean Python implementation**  
✅ **Well-commented, easy-to-understand code**  
✅ **Example usage for machine learning models**  
