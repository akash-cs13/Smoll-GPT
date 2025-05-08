import os
import regex as re
import json
import base64
from collections import Counter


class MyTokenizer:
    def __init__(self, vocab_path, merges_path, pretokenize_pattern, special_tokens):
        self._pretok_re = re.compile(pretokenize_pattern, re.VERBOSE)
        self.special_tokens = set(special_tokens)

        with open(vocab_path, "r") as f:
            data = json.load(f)
        self.vocab = {int(k): base64.b64decode(v) for k, v in data.items()}
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

        with open(merges_path, "r") as f:
            merges_data = json.load(f)
        self.merges = {tuple(map(int, k.split(','))): v for k, v in merges_data.items()}

    def split_preserve_specials(self, text):
        pat = "|".join(re.escape(s) for s in self.special_tokens)
        parts = re.split(f'({pat})', text)
        out = []
        for p in parts:
            if not p: 
                continue
            if p in self.special_tokens:
                out.append(p)
            else:
                out.extend(self._pretok_re.findall(p))
        return out

    def encode(self, text):
        substrs = self.split_preserve_specials(text)

        ids = []
        inv = self.inverse_vocab
        for s in substrs:
            if s in self.special_tokens:
                token_bytes = s.encode('utf-8')
                ids.append(inv[token_bytes])
            else:
                ids.extend(s.encode('utf-8'))

        merges = self.merges
        while True:

            cnt = Counter(zip(ids, ids[1:]))

            heap = [(merges.get(pair, float('inf')), pair) for pair in cnt]
            best_rank, best_pair = min(heap, key=lambda x: x[0], default=(None, None))

            if best_rank is None or best_rank == float('inf'):
                break
            new_id = merges[best_pair]

            out = []
            i = 0
            n = len(ids)
            a, b = best_pair
            while i < n:
                if i < n - 1 and ids[i] == a and ids[i + 1] == b:
                    out.append(new_id)
                    i += 2
                else:
                    out.append(ids[i])
                    i += 1
            ids = out
        return ids

    def decode(self, ids):
        chunks = [ self.vocab[i] for i in ids ]
        return b"".join(chunks).decode("utf-8", errors="replace")
    

if __name__ == "__main__":
    PRETOKENIZE_PATTERN = r"(?:'s|'t|'re|'ve|'m|'ll|'d)|\s?\p{L}+|\s?\p{N}+|\s?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
    SPECIAL_TOKEN = ["[EOD]"]
    dir_path = os.getcwd()

    vocab = os.path.join(dir_path, "vocab.json")
    merges = os.path.join(dir_path, "merges.json")

    tokenizer = MyTokenizer(vocab, merges, PRETOKENIZE_PATTERN, SPECIAL_TOKEN)

    test_string = "Hello mom!"

    print(test_string == tokenizer.decode(tokenizer.encode(test_string)))