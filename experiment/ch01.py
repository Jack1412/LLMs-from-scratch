from importlib.metadata import version
import torch

print("torch version: ", version("torch"))
print("tiktoken version: ", version("tiktoken"))


## Tokenizing text
import os
import requests

# download the text file
print("\n================ download the text file ================")
if not os.path.exists("the-verdict.txt"):
    url = (
        "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"
    )
    file_path = "the-verdict.txt"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    with open(file_path, "wb") as f:
        f.write(response.content)

# read the text file
print("\n================ read the text file ================")
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
print("Total number of character:", len(raw_text))
print(raw_text[:99])

# a simple tokenizer test
print("\n================ a simple tokenizer test ================")
import re

text = "Hello, world. This, is a test."
result = re.split(r'(\s)', text) # \s - 匹配任何空白字符（空格、制表符、换行等）
print(result)

result = re.split(r'([,.]|\s)', text) # |	或：左右两边满足其一即可
print(result)

# Strip whitespace from each item and then filter out any empty strings.
result = [item for item in result if item.strip()]
print(result)

text = "Hello, world. Is this-- a test?"
result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
result = [item.strip() for item in result if item.strip()]
print(result)

print("\n================ tokenize text ================")
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(preprocessed[:30])
print(f"text token count: {len(preprocessed)}")

print("\n================ convert token into token IDS ================")
all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
print("Vocab size:", vocab_size)
vocab = {token:integer for integer,token in enumerate(all_words)}
for i,item in enumerate(vocab.items()):
    print(f"vocab: {item}")
    if i >= 10:
        break

class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join(self.int_to_str[i] for i in ids)
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text

tokenizer = SimpleTokenizerV1(vocab)
text = """"It's the last he painted, you know," 
           Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
print("Token IDs:", ids)
print("Decoded text:", tokenizer.decode(ids))
print("Decoded text:", tokenizer.decode(tokenizer.encode(text)))

print("\n================ adding special context tokens ================")
tokenizer = SimpleTokenizerV1(vocab)
text = "Hello, do you like tea. Is this-- a test?"
try:
    tokenizer.encode(text)
except KeyError as e:
    print("TEST: Error during encoding:", e)

all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token:integer for integer,token in enumerate(all_tokens)}
print("Afer add special token, new vocab size:", len(vocab.items()))
for i, item in enumerate(list(vocab.items())[-5:]):
    print(f"vocab: {item}")

class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}
    
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        preprocessed = [
            item if item in self.str_to_int
            else "<|unk|>" for item in preprocessed
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join(self.int_to_str[i] for i in ids)
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text
tokenizer = SimpleTokenizerV2(vocab)
text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))
print("Combined text:", text)
print("Combined text Encoded IDs:", tokenizer.encode(text))
print("Combined text from decode:", tokenizer.decode(tokenizer.encode(text)))


print("\n================ bytepair encoding ================")
import importlib
import tiktoken
print("tiktoken version:", importlib.metadata.version("tiktoken"))
tokenizer = tiktoken.get_encoding("gpt2")
text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
     "of someunknownPlace."
)
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print("Token IDs:", integers)
strings = tokenizer.decode(integers)
print(strings)

print("\n================ Data sampling with a sliding window ================")
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
enc_text = tokenizer.encode(raw_text)
print("Total number of tokens for <the-verdict.txt>:", len(enc_text))

enc_sample = enc_text[50:]
context_size = 4
x = enc_sample[:context_size]
y = enc_sample[1:context_size+1]
print(f"x: {x}")
print(f"y:      {y}")

for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(f"Context: {tokenizer.decode(context)} -> Desired: {tokenizer.decode([desired])}")

from torch.utils.data import Dataset, DataLoader

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        assert len(token_ids) > max_length, "Number of tokenized inputs must at least be equal to max_length+1"

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i+max_length]
            target_chunk = token_ids[i+1:i+max_length+1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):
    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create the dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create the dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return dataloader

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

dataloader = create_dataloader_v1(
    raw_text, batch_size=1, max_length=4, stride=1, shuffle=False
)
# Iterable = 可被遍历的对象（相当于容器）, 有 __iter__()
# Iterator = 实际遍历的工具（相当于指针）, 有 __iter__() + __next__()
# 将「Iterable」转换为「Iterator」，能够手动控制数据的读取过程。
data_iter = iter(dataloader)
first_batch = next(data_iter)
print("First batch input IDs:", first_batch)
second_batch = next(data_iter)
print("Second batch input IDs:", second_batch)

dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=4, stride=4, shuffle=False
)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Inputs:\n", inputs)
print("\nTargets:\n", targets)

print("\n================ Creating token embeddings ================")
vocab_size = 6
output_dim = 3
torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
print(f"vocab_size={vocab_size}, output_dim={output_dim}: ", embedding_layer.weight)
print("[3] embedding is: ", embedding_layer(torch.tensor([3])))
input_ids = torch.tensor([2, 3, 5, 1])
print("[2,3,5,1] embeddings are:\n", embedding_layer(input_ids))

print("\n================ Encoding word positions ================")
vocab_size = 50257
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
max_length = 4
dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=max_length,
    stride=max_length, shuffle=False
)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Token IDs:\n", inputs)
print("\nInputs shape:", inputs.shape)

token_embeddings = token_embedding_layer(inputs)
print("token_embeddings shape: ",token_embeddings.shape)  # should be (batch_size, max_length, output_dim)

context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(
    torch.arange(max_length)
)
print("pos_embeddings shape: ",pos_embeddings.shape)  # should be (max_length, output_dim)
input_embeddings = token_embeddings + pos_embeddings
print("token_embeddings + pos_embeddings shape: ", input_embeddings.shape)  # should be (batch_size, max_length, output_dim)
# pos_embedding_layer = torch.nn.Embedding(4, 8)
# # print(pos_embedding_layer.weight)
# print(pos_embedding_layer)
# pos_embeddings = pos_embedding_layer(torch.arange(4))
# # print(pos_embedding_layer.weight)
# print(pos_embeddings)