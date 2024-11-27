import torch
import torch.nn as nn
import torch.nn.functional as F

def get_freq_cis(n_embd, block_size, theta = 10000.0, device = 'cuda'):
    freqs = 1.0 / (theta ** (torch.arange(0, n_embd, 2)[:n_embd//2].float() / n_embd)).to(device)
    positions = torch.arange(block_size, device=device)
    freqs = torch.outer(positions, freqs).to(device)
    freq_cis = torch.polar(torch.ones_like(freqs), freqs)
    # freq_cis = freq_cis.to(torch.bfloat16)
    return freq_cis

def apply_rope(q, k, freq_cis):
    """
    Apply RoPE to the query and key tensors.
    
    Args:
        q: query tensor of shape (batch_size, block_size, n_embd)
        k: key tensor of shape (batch_size, block_size, n_embd)
        freq_cis: frequency tensor of shape (block_size, n_embd//2)
    Returns:
        q_out: query tensor of shape (batch_size, block_size, n_embd)
        k_out: key tensor of shape (batch_size, block_size, n_embd)
    """
    end = q.shape[1] # block_size
    q_ = torch.view_as_complex(q.float().reshape(q.shape[:-1] + (-1, 2))) # (batch_size, block_size, n_embd//2)
    k_ = torch.view_as_complex(k.float().reshape(k.shape[:-1] + (-1, 2))) # (batch_size, block_size, n_embd//2)
    # print (q.shape, k.shape, freq_cis.shape)
    q_out = q_ * freq_cis[:end, :] # (batch_size, block_size, n_embd//2) * (block_size, n_embd//2) -> (batch_size, block_size, n_embd//2)
    k_out = k_ * freq_cis[:end, :] # (batch_size, block_size, n_embd//2) * (block_size, n_embd//2) -> (batch_size, block_size, n_embd//2)
    q_out = torch.view_as_real(q_out).reshape(*q_out.shape[:-1], -1).to(q.dtype)
    k_out = torch.view_as_real(k_out).reshape(*k_out.shape[:-1], -1).to(k.dtype)
    return q_out, k_out


class AttentionHead(nn.Module):
    def __init__(self, n_embd, head_size, block_size, device):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.tril = torch.tril(torch.ones_like(torch.zeros(block_size, block_size))).to(device)
        self.freq_cis = get_freq_cis(head_size, block_size, theta = 10000.0, device = device)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        q, k = apply_rope(q, k, self.freq_cis)
        weight = q @ k.transpose(-2, -1) * (C**-0.5) # (B, T, C) @ (B, C, T) -> (B, T, T)
        # put upper triangular part of weight to -inf
        weight = weight.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weight = F.softmax(weight, dim=-1)
        v = weight @ v
        return v


class MHA(nn.Module):
    def __init__(self, n_embd, n_heads, block_size, device):
        super().__init__()
        head_size = n_embd // n_heads
        self.heads = nn.ModuleList([AttentionHead(n_embd, head_size, block_size, device) for _ in range(n_heads)])

    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim=-1)
    


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            # nn.ReLU(),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_heads, block_size, device):
        super().__init__()
        self.sa = MHA(n_embd, n_heads, block_size, device)
        self.ffwd = FeedForward(n_embd)
        # Keep LayerNorm in fp32
        self.ln1 = nn.LayerNorm(n_embd).to(torch.float32)
        self.ln2 = nn.LayerNorm(n_embd).to(torch.float32)

    def forward(self, x):
        # Convert to fp32 for LayerNorm, then back to bf16
        x = self.ln1(x.to(torch.float32)).to(torch.bfloat16)
        x = x + self.sa(x)
        x = self.ln2(x.to(torch.float32)).to(torch.bfloat16)
        x = x + self.ffwd(x)
        return x


class GPT(nn.Module):
    def __init__(self, n_embd, n_heads, n_blocks, block_size, vocab_size, device):
        super().__init__()
        self.n_embd = n_embd
        self.n_heads = n_heads
        self.n_blocks = n_blocks
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.device = device
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_heads, block_size, device) for _ in range(n_blocks)])
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T).to(self.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        logits = self.lm_head(x)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens, T = 1.0):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            if T > 0.01:
                logits = logits / T
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            else: # greedy sampling
                idx_next = logits.argmax(dim=-1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    # class method
    @classmethod
    def from_checkpoint(cls, filepath, n_embd, n_heads, n_blocks, block_size, vocab_size, device):
        """Load model from a checkpoint file"""
        model = cls(n_embd=n_embd, n_heads=n_heads, n_blocks=n_blocks, block_size=block_size, vocab_size=vocab_size, device=device)
        model.load_state_dict(torch.load(filepath))
        return model