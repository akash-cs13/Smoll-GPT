import os
import requests
import base64
import json
from collections import defaultdict
import regex as re
from bs4 import BeautifulSoup

PRETOKENIZE_PATTERN = r"(?:'s|'t|'re|'ve|'m|'ll|'d)|\s?\p{L}+|\s?\p{N}+|\s?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
SPECIAL_TOKEN = ["[EOD]"]
VOCAB_SIZE = 5000

cwd = os.getcwd()
urls = [
    "https://en.wikipedia.org/wiki/Wine",
    "https://en.wikipedia.org/wiki/Alcoholic_beverage",
    "https://en.wikipedia.org/wiki/Old_World_wine"
]

def get_article_text(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    paragraphs = soup.find_all('p')
    return ' '.join([para.get_text() for para in paragraphs])

pre_tok = re.compile(PRETOKENIZE_PATTERN, re.VERBOSE)
def split_preserve_specials(text, specials):
    specials_pattern = '|'.join(re.escape(s) for s in specials)
    parts = re.split(f'({specials_pattern})', text)
    result = []
    for part in parts:
        if not part:
            continue
        if part in specials:
            result.append(part)
        else:
            result.append(part.rstrip())
    return result

def pre_tokenize(text: str, specials: list[str]) -> list[str]:
    merges = []
    split_at_special = split_preserve_specials(text, specials)
    for split in split_at_special:
        if split not in specials:
            merges.extend(re.findall(pre_tok, split))
        else:
            merges.append(split)
    return merges

class TrainTokenizer:
    def __init__(self, vocab_size: int = 256, special_tokens: list[str] = []):
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

        self.special_tokens = special_tokens
        self.special_token_ids = {token: len(self.vocab) + i for i, token in enumerate(self.special_tokens)}

        for token, token_id in self.special_token_ids.items():
            self.vocab[token_id] = token.encode('utf-8')

        self.vocab_size = vocab_size
        initial_size = len(self.vocab)
        self.num_merges = max(0, vocab_size - initial_size)

        self.merges: dict[tuple[int, int], int] = {}

    def get_stats(self, sequences: list[list[int]]) -> dict[tuple[int, int], int]:
        stats: dict[tuple[int, int], int] = defaultdict(int)
        for seq in sequences:
            for a, b in zip(seq, seq[1:]):
                if a in self.special_token_ids.values() or b in self.special_token_ids.values():
                    continue
                stats[(a, b)] += 1
        return stats

    def merge_sequence(self, seq: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        merged: list[int] = []
        i = 0
        while i < len(seq):
            if i < len(seq) - 1 and seq[i] == pair[0] and seq[i+1] == pair[1]:
                merged.append(new_id)
                i += 2
            else:
                merged.append(seq[i])
                i += 1
        return merged

    def bpe(self, sequences: list[list[int]], verbose: bool = False) -> list[list[int]]:
        seqs = sequences
        for merge_index in range(self.num_merges):
            stats = self.get_stats(seqs)
            if not stats:
                break
            best_pair = max(stats, key=stats.get)
            new_token_id = len(self.vocab) + merge_index
            if verbose:
                print(f"Merging pair {best_pair} -> {new_token_id}")
            seqs = [self.merge_sequence(seq, best_pair, new_token_id) for seq in seqs]
            self.merges[best_pair] = new_token_id
        for (a, b), new_id in self.merges.items():
            self.vocab[new_id] = self.vocab[a] + self.vocab[b]
        return seqs

if __name__ == "__main__":
    print("Fetching dataset...")
    all_text = ' '.join([get_article_text(url) for url in urls])

    data_path = os.path.join(cwd,"dataset", "tokenizer.txt")
    with open(data_path, 'r', encoding='utf-8') as f:
        joined_text = f.read()

    joined_text = joined_text + " " + all_text

    DATA = joined_text
    print("Preparing to train the tokenizer")

    # 1) Pre-tokenize including special token
    subtokens = pre_tokenize(DATA, SPECIAL_TOKEN)
    print("Pre-tokens:", subtokens)

    # 2) Convert each subtoken to a sequence of byte IDs
    tokenizer = TrainTokenizer(vocab_size=VOCAB_SIZE, special_tokens=SPECIAL_TOKEN)

    token_sequences = []
    for tok in subtokens:
        if tok in SPECIAL_TOKEN:
            token_sequences.append([tokenizer.special_token_ids[tok]])
        else:
            token_sequences.append(list(tok.encode('utf-8')))
    print("Token sequences:", token_sequences)

    # 3) Train BPE on these sequences
    merged_sequences = tokenizer.bpe(token_sequences, verbose=True)
    print("Merged sequences:", merged_sequences)

    # 4) Flatten for model input
    flat_ids = [token_id for seq in merged_sequences for token_id in seq]
    print("Flattened IDs:", flat_ids)

    # 5) Inspect results
    print("Final vocab size:", len(tokenizer.vocab))
    print("Merge rules count:", len(tokenizer.merges))
    print("Sample merges:", list(tokenizer.merges.items())[:10])

    vocab_serializable = {k: base64.b64encode(v).decode('utf-8') for k, v in tokenizer.vocab.items()}
    merges_serializable = {f"{k[0]},{k[1]}": v for k, v in tokenizer.merges.items()}

    with open(os.path.join(cwd, "vocab2.json"), "w") as f:
        json.dump(vocab_serializable, f, indent=2)

    with open(os.path.join(cwd, "merges2.json"), "w") as f:
        json.dump(merges_serializable, f, indent=2)

    print(f"Saved vocab and merges to: {cwd}")