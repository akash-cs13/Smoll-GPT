# Smoll GPT

A **compact language model** tailored for generating wine reviews. It leverages a curated dataset of wine tasting notes and a NanoGPT-inspired architecture to deliver coherent, flavorful descriptions with minimal resources.

---

## 📦 Project Overview

* **Model size**: 846,340 parameters
* **Weights file**: 3.5 MB (`gpt_model.pth`)
* **Dataset**: [zynicide/wine-reviews](https://www.kaggle.com/datasets/zynicide/wine-reviews)

This repository contains everything needed to reproduce the tokenizer, train the model, and generate wine reviews.

---

## 🧪 Try It Out

Spin up the demo instantly with Docker:

```bash
docker run --rm -p 3000:3000 akashcs13/wine-review:latest
```

Or try it online without setup at 👉 [Smoll-GPT](https://wine-review.vercel.app/)

---

## 🗂️ Directory Structure

```
├── dataset
│   ├── dataset.py             # Script to generate the text files
│   ├── joined_text.txt        # Full processed dataset with [EOD] markers
│   ├── sample_text.txt        # 200 random samples with [EOD]
│   ├── tokenizer.txt          # 250 samples joined for tokenizer training
│   └── data.pt                # Pre-encoded dataset tensor
│
├── Tokenizer
│   ├── vocab.json
│   ├── merges.json
│   ├── Tokenizer.py           # BPE tokenizer implementation
│   └── train_tokenizer.py     # Training script to generate vocab and merges
│  
├── GPTModel
│   └── Model.py               # NanoGPT-style model definition
│ 
├── models
│   └── gpt_model.pth          # Best trained model weights
│
├── main.py                    # Training script
├── prediction.py              # Text generation script
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation 
```

---

## 📝 Dataset Processing

All raw reviews were converted into the format:

```
wine review : {Country} : {Province} : {Variety} : {Description}
```

* **joined\_text.txt**: Complete dataset with `[EOD]` delimiters.
* **sample\_text.txt**: Subset of 200 random documents (for quick experimentation).
* **tokenizer.txt**: 250 samples concatenated with spaces to train the tokenizer.

*Processing note*: Converting `joined_text.txt` with the current (single-threaded) tokenizer takes \~2:20:53; parallelization can reduce this to \~0:00:33.

---

## 🔠 Tokenizer

Implement a Byte-Pair Encoding (BPE) tokenizer trained on:

1. `tokenizer.txt` (74,412 chars)
2. Three Wikipedia articles (\~74,635 chars total):

   * [Wine](https://en.wikipedia.org/wiki/Wine)
   * [Alcoholic beverage](https://en.wikipedia.org/wiki/Alcoholic_beverage)
   * [Old World wine](https://en.wikipedia.org/wiki/Old_World_wine)

* **Vocabulary size**: 5,000 tokens
* **Outputs**: `vocab.json`, `merges.json`

**Implementation details**:

```python
PRETOKENIZE_PATTERN = r"(?:'s|'t|'re|'ve|'m|'ll|'d)|\s?\p{L}+|\s?\p{N}+|\s?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
SPECIAL_TOKENS = ['[EOD]']
```

Provides `encode(text) -> List[int]` and `decode(tokens) -> str`.

---

## 🧠 Language Model

Based on [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT). Hyperparameters:

```yaml
batch_size: 32
block_size: 32
eval_iters: 200
n_embd: 64
n_head: 4
n_layer: 4
dropout: 0.0
vocab_size: 5000
```

* **Model code**: `models/GPTModel/Model.py`
* **Training loop**: `main.py`
* **Best checkpoint**: `models/gpt_model.pth`

---

## 🚀 Usage

### 1. Installation

```bash
git clone https://github.com/akash-cs13/Smoll-GPT.git
cd Smoll-GPT
python3 -m venv venv
source venv/bin/activate       # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

### 2. Generating Reviews

```bash
python prediction.py
```

*Example output:*

```
Input: 
wine review : US : California : Bordeaux-style Red Blend : A blend of Cabernet

Output: 
------------------------- 
wine review : US : California : Bordeaux-style Red Blend : A blend of Cabernet Sauvignon but well- structured wine extra draw. The variety comes from out and st yellow peach and a tangy, with its caramel riding highose chis chemical and cinnamon, grainy citrus creamy glass. Spices and smooth The wine, black plum flavors on the palate. There's moderate, but more rather only fresh fennelred by sour L leavess chunkyery peach note, yet countered have a dry, with an easygoing wine that's balanced by smoother and aged in neutral French oak and cocoa. Tightly packed with the freshnesslow to soften.

wine review : Italy : Tuscany : Red Blend : This blend of 87% Zinfandel, 26% Merlot, 20% Syrah and 9% Petit Verdot, this doesn't offering with a touch of Petit Verdot. Ripe juicy takes as elegant, lime, isn't drink firm, floral and toast leave an open feel lean palate Sauvignon Blancing T proportionate through the vintage structure. It gets a perfect mix of those exotic oak, raspberry, wood-tale Greek and wine. It's light and silky beginning to drink now, with a truffle and some ripe black fruit, which seems all add up to drink.

wine review : US : Virginia : Syrah : Leafy, leather and aromas draw you but refined tannins provide balance ofgain verbena high-drinking choppy acidity provides needed balance, this doesn't come that lend it concentrated, this red-bodied texture is a ripe berry aromas of fragrant won't sleek savory pear and an attractive in previous vintages, flush and personality.
```

### 3. Training the Model

```bash
python main.py
```

* Checkpoints will be saved in `models/` by default.

---

## ⚙️ Requirements

* Ubuntu 24.04.2 LTS
* Python 3.12.3

---

## 📝 To-Do

- [X] Optimize the model and create a Docker image.  
- [ ] Rebuild the tokenizer in C++ for speed.  
- [ ] Design a custom model instead of using nanoGPT.  
- [ ] Explore and build alternative models like BERT or reasoning-focused architectures.

---

## 📜 License

[MIT License](./LICENSE)

---


