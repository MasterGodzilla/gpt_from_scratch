import torch
import math
import time
from torch.amp import autocast
import matplotlib.pyplot as plt

# from mha import GPT
from mha_rope import GPT

torch.random.manual_seed(1337)
# hyperparameters

block_size = 128
n_blocks = 8
n_heads = 4
n_embd = 128

lr_max = 1e-3  # Peak learning rate
lr_min = 1e-4  # Minimum learning rate
total_iters = 3001
warmup_iters = 200  # Number of warmup iterations
lr_decay_iters = total_iters  # Total number of iterations for lr decay
anneal_iters = 200

batch_size = 128
eval_interval = 100
prefetch_size = 10
with open('input.txt', 'r') as file:
    text = file.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

char_to_idx = { ch:i for i, ch in enumerate(chars) }
idx_to_char = { i:ch for i, ch in enumerate(chars) }

def encode(s):
    return [char_to_idx[ch] for ch in s]

def decode(l):
    return ''.join([idx_to_char[i] for i in l])

def load_data():
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9*len(data))
    train_data = data[:n]
    val_data = data[n:]
    return train_data, val_data

train_data, val_data = load_data()
print(train_data[:100])

device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
# Add after device definition (around in[6])
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Current device: {device}")

# To enable BF16:
torch.set_default_dtype(torch.bfloat16)
print(f"Default dtype: {torch.get_default_dtype()}")

def load_batch(batch_size, block_size):
    random_indices = torch.randint(len(train_data) - block_size, (batch_size,))
    x = torch.stack([train_data[i:i+block_size] for i in random_indices])
    y = torch.stack([train_data[i+1:i+block_size+1] for i in random_indices])
    return x.to(device), y.to(device)

def prefetch_batches(num_batches, batch_size, block_size):
    batches = []
    xb, yb = load_batch(batch_size=batch_size*num_batches, block_size=block_size)
    # Reshape to split into num_batches
    xb = xb.view(num_batches, batch_size, -1) 
    yb = yb.view(num_batches, batch_size, -1)
    batches = list(zip(xb, yb))
    return batches


def estimate_loss(model, batch_size, block_size):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        X, Y = load_batch(batch_size, block_size)
        logits, loss = model(X, Y)
        out[split] = loss.item()
    model.train()
    return out


# try inference
model = GPT(n_embd=n_embd, n_heads=n_heads, n_blocks=n_blocks, block_size=block_size, vocab_size=vocab_size, device=device).to(device)
# model = GPT.from_checkpoint("model_19500.pth", n_embd, n_heads, n_blocks, block_size, vocab_size, device).to(device)
# model.compile()

message = "Hello, how are you? How are you doing?"
idx = torch.tensor(encode(message), dtype=torch.long).unsqueeze(0).to(device)
print(decode(model.generate(idx, max_new_tokens=300, T=0.3)[0].tolist()))

def get_lr(it):
    # Linear warmup for warmup_iters steps
    if it < warmup_iters:
        return lr_max * it / warmup_iters
    # Cosine learning rate decay
    if it > lr_decay_iters:
        return lr_min
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # Cosine decay
    return lr_min + coeff * (lr_max - lr_min)

def trapezoid_lr(it):
    if it < warmup_iters:
        return lr_max * it / warmup_iters
    if it > total_iters - anneal_iters:
        return lr_max - (it - (total_iters - anneal_iters)) * (lr_max - lr_min) / anneal_iters
    return lr_max





optimizer = torch.optim.AdamW(model.parameters(), lr=lr_max)

# Lists to store metrics for plotting
losses = []
learning_rates = []

start_time = time.time()
batches = prefetch_batches(prefetch_size, batch_size, block_size)
for i in range(total_iters):

    lr = trapezoid_lr(i)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    learning_rates.append(lr)

    batch_idx = i % prefetch_size
    if batch_idx == 0:
        batches = prefetch_batches(prefetch_size, batch_size, block_size)
    xb, yb = batches[batch_idx]
    with autocast('cuda', dtype=torch.bfloat16):
        logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    
    if i % eval_interval == 0:
        elapsed = time.time() - start_time
        print(f"iter {i}: loss {loss.item():.4f} (elapsed: {elapsed:.2f}s) lr {lr:.4f}")
        print("example output:", decode(model.generate(idx, max_new_tokens=100)[0].tolist()))
        out = estimate_loss(model, batch_size, block_size)
        print(f"train loss: {out['train']:.4f}, val loss: {out['val']:.4f}")
    
    # save a checkpoint before annealing
    if i == total_iters - anneal_iters:
        torch.save(model.state_dict(), f"model_{i}.pth")

torch.save(model.state_dict(), f"model_{i}.pth")

# save losses to logs folder
# time stamp
timestamp = time.strftime("%Y%m%d_%H%M%S")
with open(f'logs/losses_{timestamp}.txt', 'w') as f:
    for loss in losses:
        f.write(f"iter {i}: loss {loss:.4f}\n")

# Plot loss and learning rate
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

ax1.plot(losses)
ax1.set_title('Training Loss')
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Loss')

ax2.plot(learning_rates)
ax2.set_title('Learning Rate Schedule')
ax2.set_xlabel('Iteration') 
ax2.set_ylabel('Learning Rate')

plt.tight_layout()
plt.show()

# save plot as png
plt.savefig('plot_GELU.png')
