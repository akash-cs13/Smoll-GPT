import os
import torch
from Tokenizer.Tokenizer import MyTokenizer
from GPTModel.Model import *

NEW_TOKENS_TO_GENERATE = 1000
INPUT = "wine review : US : California : Bordeaux-style Red Blend : A blend of Cabernet"

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
print("Model loaded...")

print("Input: ")
print(INPUT)
context = torch.tensor([tokenizer.encode(INPUT)], dtype=torch.long, device=device)
output = tokenizer.decode(model.generate(context, max_new_tokens=NEW_TOKENS_TO_GENERATE)[0].tolist())

print("------------------------- ")
print("Output: ")
print_output = output.replace(SPECIAL_TOKEN[0], "\n\n")
print(print_output)
print("\n\n")