import os
import pickle
import numpy as np
import torch

from model import GPTConfig, GPT

train_data_path = "data/mnist/train.bin"
val_data_path = "data/mnist/val.bin"
out_dir     = "out_image"
vocab_size  = 2
img_h       = 28
img_w       = 28
seq_len     = img_h * img_w
block_size  = seq_len - 1
batch_size  = 32
max_iters   = 3000
eval_every  = 250
eval_iters  = 50
learning_rate = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"

n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.0

torch.manual_seed(1337)

train_data = np.memmap(train_data_path, dtype=np.uint16, mode="r")
val_data   = np.memmap(val_data_path, dtype=np.uint16, mode="r")

def get_batch(split):
    d = train_data if split == "train" else val_data
    imgs = d.reshape(-1, seq_len)
    ix = torch.randint(imgs.shape[0], (batch_size,))
    rows = torch.from_numpy(imgs[ix].astype(np.int64))
    x = rows[:, :-1].contiguous()
    y = rows[:, 1:].contiguous()
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

config = GPTConfig(block_size=block_size, vocab_size=vocab_size,
                   n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                   dropout=dropout, bias=False)
model = GPT(config).to(device)

optimizer = model.configure_optimizers(
    weight_decay=0.1, learning_rate=learning_rate,
    betas=(0.9, 0.99), device_type=device)

os.makedirs(out_dir, exist_ok=True)
best_val = float("inf")

for it in range(max_iters + 1):
    if it % eval_every == 0:
        losses = estimate_loss()
        print(f"iter {it:5d} | train {losses['train']:.4f} | val {losses['val']:.4f}")
        if losses["val"] < best_val:
            best_val = losses["val"]
            torch.save({"model": model.state_dict(), "config": config.__dict__},
                       os.path.join(out_dir, "ckpt.pt"))

    X, Y = get_batch("train")
    _, loss = model(X, Y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(f"done. best val loss {best_val:.4f}. checkpoint in {out_dir}/")
