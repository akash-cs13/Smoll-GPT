import os
import random
import kagglehub
import pandas as pd

path = kagglehub.dataset_download("zynicide/wine-reviews")

print("Path to dataset files:", path)

cwd = os.getcwd()
for file in os.listdir(path):
    print(file)

df = pd.read_csv(os.path.join(path, "winemag-data-130k-v2.csv"))
df.drop(columns=["Unnamed: 0"], inplace=True)
df.drop(columns=["designation", "points", "price", "region_1", "region_2", "taster_name", "taster_twitter_handle", "title", "winery"], inplace=True)
df = df.dropna()
df.loc[:, "processed"]  = (
    "wine review : " +
    df["country"].astype(str) + " : " +
    df["province"].astype(str) + " : " +
    df["variety"].astype(str) + " : " +
    df["description"].astype(str)
)
df.drop_duplicates(subset="processed", inplace=True)
sampled_text = df['processed'].sample(n=250, random_state=42).astype(str).tolist()

joined_text = " ".join(sampled_text)
with open(os.path.join(cwd, "dataset", "tokenizer.txt"), "w") as f:
  f.write(joined_text)
print("Generated tokenizer.txt")


sampled = df['processed'].astype(str).sample(n=200, random_state=42).tolist()
joined_text = '[EOD]'.join(sampled)
with open(os.path.join(cwd, "dataset", "sample_text.txt"), "w") as f:
  f.write(joined_text)
print("sample_text.txt")


joined_text = '[EOD]'.join(df['processed'].astype(str).tolist())
with open(os.path.join(cwd, "dataset", "joined_text.txt"), "w") as f:
  f.write(joined_text)
print("joined_text.txt")
