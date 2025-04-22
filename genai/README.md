# 📘 Generative AI (GenAI)

---

## 🧠 What is Generative AI?
Generative AI refers to AI systems capable of generating text, images, video, audio, and synthetic data. Unlike discriminative models which classify data, generative models learn the **underlying distribution** of data and can **generate new samples** from it.

Use cases:
- Image synthesis (DALL·E, MidJourney)
- Music and audio generation (Jukebox)
- Text generation (GPT series)
- Video generation (Sora)

Mathematically, generative models try to model the **joint probability distribution** $P(x, y)$ or the **data distribution** $P(x)$.

---

## 🔍 CNN & DeepDream

### 🎯 Purpose
Understanding how neural networks perceive and represent visual features.

### 📐 Architecture: Convolutional Neural Networks
- **Layers**: Conv2D → ReLU → MaxPooling → Dense
- Used in early visual recognition tasks

### 🧮 Math
Convolution:
$$
Y(i, j) = \sum_m \sum_n X(i + m, j + n) \cdot K(m, n)
$$

### 🎨 DeepDream
- Visualizes patterns learned by a CNN
- Optimizes image to maximize activations of specific layers
- Uses **gradient ascent** on input image

---

## 🔁 GANs – Generative Adversarial Networks

### 🎯 Purpose
Learn to generate data by pitting two networks against each other.

### 📐 Architecture
- **Generator** $G(z)$: Generates fake samples from noise $z$
- **Discriminator** $D(x)$: Classifies real vs. fake

### 🧮 Math
Objective:
$$
\min_G \max_D \mathbb{E}_{x \sim p_{data}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]
$$

### 🔧 Variants
- DCGAN (Deep Convolutional GAN)
- CGAN (Conditional GAN)
- StyleGAN

---

## 🔐 VAEs – Variational Autoencoders

### 🎯 Purpose
Model probabilistic latent variables to generate smooth outputs.

### 📐 Architecture
- Encoder: $q_\phi(z|x)$
- Decoder: $p_\theta(x|z)$
- Bottleneck: Latent vector $z$

### 🧮 Math
Loss Function:
$$
\mathcal{L}(x) = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) \| p(z))
$$

---

## 🧠 Transformers

### 🎯 Purpose
Sequence modeling via attention instead of recurrence.

### 📐 Architecture
- Encoder: Self-attention + FFN
- Decoder: Masked self-attention + Encoder-decoder attention

### 🧮 Math
Self-Attention:
$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

### 📦 Applications
- GPT, BERT
- Vision Transformers (ViT)
- Language translation

---

## 🔗 CLIP – Contrastive Language-Image Pretraining

### 🎯 Purpose
Connect images and text into a joint embedding space.

### 📐 Architecture
- Image encoder (ResNet or ViT)
- Text encoder (Transformer)
- Contrastive loss aligns representations

### 🧮 Math
Contrastive loss:
$$
\mathcal{L} = -\log \frac{\exp(\text{sim}(x_i, y_i)/\tau)}{\sum_j \exp(\text{sim}(x_i, y_j)/\tau)}
$$

Where $\text{sim}$ is cosine similarity and $\tau$ is a temperature parameter.

---

## 💨 Diffusion Models

### 🎯 Purpose
Generate high-fidelity data via denoising process

### 📐 Architecture
- **Forward process**: Gradually adds noise to image $x_0 \to x_T$
- **Reverse process**: Learns to denoise step-by-step $x_T \to x_0$
- UNet is used as backbone

### 🧮 Math
Forward Process:
$$
q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t}x_{t-1}, \beta_t I)
$$

Reverse Process:
$$
\theta^* = \arg\min_\theta \mathbb{E}_{x_t, t} \left[ \|\epsilon - \epsilon_\theta(x_t, t)\|^2 \right]
$$

### 🧩 Variants
- DDPM (Denoising Diffusion Probabilistic Models)
- Latent Diffusion Models (used in Stable Diffusion)
- Classifier-Free Guidance

---

## 📚 References
- [Attention Is All You Need (2017)](https://arxiv.org/abs/1706.03762)
- [Auto-Encoding Variational Bayes (2013)](https://arxiv.org/abs/1312.6114)
- [Denoising Diffusion Probabilistic Models (2020)](https://arxiv.org/abs/2006.11239)
- [CLIP: Learning Transferable Visual Models (2021)](https://arxiv.org/abs/2103.00020)
- [Generative Adversarial Nets (2014)](https://arxiv.org/abs/1406.2661)

---