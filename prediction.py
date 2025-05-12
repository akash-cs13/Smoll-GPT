import os
import time
import torch
from Tokenizer.Tokenizer import MyTokenizer
from GPTModel.Model import *

INPUT = "wine review : US : California : Bordeaux-style Red Blend : A blend of Cabernet"
TOTAL_NUMBER_OF_REVIEWS = 3
MAX_CONTEXT_LENGTH = 32

TYPEWRITER_MODE = True  
DESIRED_CHAR_PER_SEC = 40  
DESIRED_DELAY = 1.0 / DESIRED_CHAR_PER_SEC

device = 'cuda' if torch.cuda.is_available() else 'cpu'

PRETOKENIZE_PATTERN = r"(?:'s|'t|'re|'ve|'m|'ll|'d)|\s?\p{L}+|\s?\p{N}+|\s?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
SPECIAL_TOKEN = ["[EOD]"]
dir_path = os.getcwd()

vocab = os.path.join(dir_path, "Tokenizer", "vocab.json")
merges = os.path.join(dir_path, "Tokenizer", "merges.json")

tokenizer = MyTokenizer(vocab, merges, PRETOKENIZE_PATTERN, SPECIAL_TOKEN)
print("Tokenizer loaded...")

cwd = os.getcwd()
model_filename = "gpt_model.pth"
model_path = os.path.join(cwd, "models", model_filename)

model = BigramLanguageModel() 
state_dict = torch.load(model_path, map_location=device, weights_only=True)
model.load_state_dict(state_dict)
model.to(device)
model.eval()
print("Model loaded...\n")

print("Input: ")
print(INPUT + "\n")
print("Output: ")
print("------------------------- ")
print(INPUT, end='')

context = torch.tensor([tokenizer.encode(INPUT)], dtype=torch.long, device=device)
while TOTAL_NUMBER_OF_REVIEWS > 0:
    start_time = time.time()

    next_id = model.generate_simple(context)
    next_char = tokenizer.decode(next_id[0].tolist())

    if next_char == SPECIAL_TOKEN[0]:
        print("\n\n", end='', flush=True)
        TOTAL_NUMBER_OF_REVIEWS -= 1
    else:
        print(next_char, end='', flush=True)

    context = torch.cat((context, next_id), dim=1)
    context = context[:, -MAX_CONTEXT_LENGTH:]

    if TYPEWRITER_MODE:
        elapsed = time.time() - start_time
        if elapsed < DESIRED_DELAY:
            time.sleep(DESIRED_DELAY - elapsed)
