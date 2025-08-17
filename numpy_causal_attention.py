# numpy_causal_attention.py
import numpy as np

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=True)

class NumpyCausalSelfAttention:
    def __init__(self, d_model: int, n_head: int, seed: int = 0):
        assert d_model % n_head == 0
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_model // n_head
        rng = np.random.default_rng(seed)
        # Parameter (Kaiming/Glorot ist optional; hier simpel)
        self.W_qkv = (rng.standard_normal((d_model, 3*d_model)) / np.sqrt(d_model)).astype(np.float32)
        self.b_qkv = np.zeros((3*d_model,), dtype=np.float32)
        self.W_o   = (rng.standard_normal((d_model, d_model)) / np.sqrt(d_model)).astype(np.float32)
        self.b_o   = np.zeros((d_model,), dtype=np.float32)

    def _split_heads(self, x):
        # x: (B,T,C) -> (B,h,T,d)
        B,T,C = x.shape
        x = x.reshape(B, T, self.n_head, self.d_head)
        return np.transpose(x, (0,2,1,3))

    def _merge_heads(self, x):
        # x: (B,h,T,d) -> (B,T,C)
        B,h,T,d = x.shape
        x = np.transpose(x, (0,2,1,3)).reshape(B, T, h*d)
        return x

    def forward(self, x: np.ndarray):
        """
        x: (B,T,C) float32
        returns y: (B,T,C)
        """
        B,T,C = x.shape

        # 1) QKV
        qkv = x @ self.W_qkv + self.b_qkv         # (B,T,3C)
        q, k, v = np.split(qkv, 3, axis=-1)       # je (B,T,C)
        q = self._split_heads(q)                  # (B,h,T,d)
        k = self._split_heads(k)
        v = self._split_heads(v)

        # 2) Scores
        scale = 1.0 / np.sqrt(self.d_head)
        attn_logits = np.matmul(q, np.swapaxes(k, -2, -1)) * scale   # (B,h,T,T)

        # 3) Causal-Maske
        mask = np.tril(np.ones((T,T), dtype=bool))                    # (T,T)
        # set future positions to -inf
        neg_inf = np.finfo(np.float32).min
        attn_logits = np.where(mask[None,None,:,:], attn_logits, neg_inf)

        # 4) Softmax
        attn = softmax(attn_logits, axis=-1)                          # (B,h,T,T)

        # 5) Aggregation
        y = np.matmul(attn, v)                                        # (B,h,T,d)

        # 6) Merge Heads + Output-Proj
        y = self._merge_heads(y)                                      # (B,T,C)
        y = y @ self.W_o + self.b_o                                   # (B,T,C)
        return y

# --- Mini-Test ---
if __name__ == "__main__":
    B,T,C,h = 2, 6, 32, 4
    x = np.random.randn(B,T,C).astype(np.float32)
    attn = NumpyCausalSelfAttention(d_model=C, n_head=h, seed=42)
    y = attn.forward(x)
    print("out shape:", y.shape)  # (2,6,32)
