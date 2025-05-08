import os
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from Tokenizer.Tokenizer import MyTokenizer
from GPTModel.Model import *

max_iters = 10000
eval_interval = 100
learning_rate = 1e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(1337)

PRETOKENIZE_PATTERN = r"(?:'s|'t|'re|'ve|'m|'ll|'d)|\s?\p{L}+|\s?\p{N}+|\s?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
SPECIAL_TOKEN = ["[EOD]"]
dir_path = os.getcwd()

vocab = os.path.join(dir_path, "Tokenizer", "vocab.json")
merges = os.path.join(dir_path, "Tokenizer", "merges.json")

tokenizer = MyTokenizer(vocab, merges, PRETOKENIZE_PATTERN, SPECIAL_TOKEN)

'''  Very slow, just use the tensor instead
data_path = os.path.join(dir_path, "dataset", "joined_text.txt")
with open(data_path, 'r', encoding='utf-8') as f:
    text = f.read()

chunks = [text[i:i+5000] for i in range(0, len(text), 5000)]
encoded = [None] * len(chunks)

with ThreadPoolExecutor(max_workers=8) as executor:
    futures = {executor.submit(tokenizer.encode, chunk): idx for idx, chunk in enumerate(chunks)}
    for future in tqdm(as_completed(futures), total=len(futures), desc="Encoding text"):
        idx = futures[future]
        encoded[idx] = future.result()

flat_encoded = [id for sublist in encoded for id in sublist]
data = torch.tensor(flat_encoded, dtype=torch.long)
'''


data_path = os.path.join(dir_path, "dataset", "data.pt")

data = torch.load(data_path, map_location=device, weights_only=True)
print(f"Data shape: {data.shape}")

# Train and test splits
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


model = BigramLanguageModel()
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.tensor([[1147]], dtype=torch.long, device=device)
print(tokenizer.decode(m.generate(context, max_new_tokens=2000)[0].tolist()))

model_path = os.path.join(dir_path, "models", "gpt_model.pth")
torch.save(m.state_dict(), model_path)
print(f"Model saved to {model_path}")