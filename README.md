# Smoll GPT

A **compact language model** tailored for generating wine reviews. It leverages a curated dataset of wine tasting notes and a NanoGPT-inspired architecture to deliver coherent, flavorful descriptions with minimal resources.

---

## 📦 Project Overview

* **Model size**: 846,340 parameters
* **Weights file**: 3.5 MB (`gpt_model.pth`)
* **Dataset**: [zynicide/wine-reviews](https://www.kaggle.com/datasets/zynicide/wine-reviews)

This repository contains everything needed to reproduce the tokenizer, train the model, and generate wine reviews.

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
------------------------- 
Output: 
wine review : US : California : Bordeaux-style Red Blend : A blend of Cabernet, informal quality to this is finely sweet in the mouth. Bility elegant, with good apple skin, aromas of for drinking it high-tated Or. Fruity and baked blueberry, structured, withrell continues through with roast chicken flowers and sophisticated. And there's Carmenère.

wine review : US : Virginia : Cabernet France : Grigio : Foxy aromas and it's complex, features a toasty note from heady blackberry, bacon is ready to intoxication blend of who and new oak, appealing aromas of dried sandard pepper. Like most sush a lengthy and firm tannins, clean but it is up the intense big, bright and mouth. Upine-cherry Merlot, the lengthy finish is perfumed wine, and an easygoing cherry, firm acidity that lifts it offers touches of savory spice backed by a finalating of candied bland Pod tobacco and orange way does out across the nose, barely be accentu on the nose and palate, followed by bright acidity and upfront, nutm kick it cherry flavors of black plum and peaches. R but itlaive. Dusty liver terly with a sharp.relely, it's in cherry and tight and turn chocolaty-old vines provide the bouquet is tannins finish. Drink through 2020.

wine review : Italy : Tuscany : Red Blend : Light, enjoyment, spice, a soft, tar the wine is ready to citric lift to a concentrated texture.

wine review : France : Bordeaux : Bordeaux-style blends in wooded in the Entre-grained tannins and-bodied but in a uniquely und the end. Salty, light and ripe currant flavors have brings in feel, with a note.

wine review : Australia : South Australia : Rosé : Rosé : This Merlot from good, sophisticated aromas of lime, wet earth, dark the of of of red spice and black cherry on the nose. The palate offers great mouthfeel, while the finish is rich with acidity and textured, structured and purple fruit, with in acidity a dry, citrus aromas of spice. Like any apple, with a bright acidity and a mineral. Res of redarralood, blueberry, apple and spice flavors finish.

wine review : Australia : Sicily & Sardinia : Cabernet Sauvignon Blanc : This wine combines smoky, plump, toast and tropical fruit flavors estate vineyard since the spice creamy-textured Cab may not heavy but not make caramelized.

wine review : Italy :asabablanca Valley : Corvina citrus with ripe flavors of extract gives ahe effort.

wine review : US : California : Cabernet Sauvignon : Inspired-style Red Blend : Crushed violet, with rich wine but it feels being review anchino cherries. It is well as lively and sand se complexity. A good it is medium-balanced, packed but the bouquet With introduces flavors core is open feel due all, it apple and smooth. Acid prominently in acidity along with the very high acidity. Drink now.

wine review : Roberts : Thermenregion : St. Laurent : Despite its own earlier with the palate, but it's just best Sauvignon Blanc, this is is complex and youthful, currant, g yet ro possibly well for another decade of stash it extremely approachable. Drink now, but it ideallyesternably-yearlyow accompaniment off and melon. Blackes it'll show some fine tannins, this is just some off dry, its an edgy high-red by hints of La tart coffee and here with candy. The wine is now.

wine review : Spain : Northern Spain : Tempranillo : Fragon preserve and French oak flavors demand attention from the nose. Best from out w fig and round, this is rich in fruit and blackberry, moderately ripe fruit and as solid freshness to give way. Nut : Fresh palate is not labeled. The 2 through with acidity and goodicy raspberry flavors., gameing char pears ripe refined tannins provide the framework. Drink through 2019.

wine review : US : New York : Baco Noir : Despite its delicate body, there's a fleshy raspberry, forward andaffable that works in sweetness, buterscoring the wine that show drops. Fresh acidity is well.

wine review : US : California : Cabernet Sauvignon : G 100% Cabernet Sauvignon with its flashy touch of pineapples and Chardonnay fruit flavors. This packed with sush -scoring the palate is really too much.

wine review : Portugal : Tejo : Rosé : This blend of mango, blackberries, citrus brings a purelyav cherry fruit and cedar months aromas. It'sad style. Save a few bottles until 2020 toist soon.

wine review : US : California : Chardonnay : Racy acidity and fruity coat of a simple A good core of oak tones of oak, resin and balanced Pinot Noir displaysboa dishes. It's a Like any also with the  The wine, with a good price, veryenty and dry structure.

wine review :
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

## 📜 License

[MIT License](./LICENSE)

---


