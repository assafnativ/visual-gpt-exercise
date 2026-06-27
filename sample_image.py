import os
import sys
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
temperature = 0.8
learning_rate = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"

ckpt = torch.load(os.path.join(out_dir, "ckpt.pt"), map_location=device)
model = GPT(GPTConfig(**ckpt["config"])).to(device)
model.load_state_dict(ckpt["model"])
model.eval()

idx = torch.zeros((1, 1), dtype=torch.long, device=device)
out = model.generate(idx, max_new_tokens=block_size,
                     temperature=temperature, top_k=None)

img = np.array(out[0].tolist()).reshape(img_h, img_w)
print("Generated digit:")
for row in img:
    print("".join("##" if v else ".." for v in row))
