```
transformer_project/
│── models/
│   ├── __init__.py
│   ├── transformer_encoder.py   # Transformer Encoder
│   ├── transformer_decoder.py   # Transformer Decoder
│   ├── multi_head_attention.py  # Multi-Head Self-Attention
│   ├── feed_forward.py          # Feed Forward Network
│   ├── positional_encoding.py   # Positional Encoding
│   ├── transformer.py           # Full Transformer (Combines Encoder + Decoder)
│── train.py                     # Training Script
│── dataset_loader.py             # Data Preprocessing
│── inference.py                  # Inference & Testing
│── requirements.txt              # Dependencies
│── README.md                     # Documentation
```

- `multi_head_attention.py` → Implements Multi-Head Attention (used in both Encoder & Decoder).
- `feed_forward.py` → Implements the Feed-Forward Network.
- `positional_encoding.py` → Implements Positional Encoding for handling sequence order.
- `transformer_encoder.py` → Defines the Transformer Encoder block.
- `transformer_decoder.py` → Defines the Transformer Decoder block.
- `transformer.py` → Combines Encoder + Decoder into a full Transformer model.
- `train.py` → Loads dataset, initializes the Transformer model, and trains it.
- `inference.py` → Performs testing or generates predictions using a trained model.

---
#### Multi-Head Self-Attention    
    1. Computes Q (queries), K (keys), and V (values) using dense layers.
    2. Splits them into multiple heads for parallel processing.
    3. Uses Scaled Dot-Product Attention to compute attention weights.
    4. Concatenates outputs from all heads and passes through a final dense layer.

#### Feed-Forward Network
    1. This defines the Position-wise Feed-Forward Network used inside both encoder and decoder blocks.

