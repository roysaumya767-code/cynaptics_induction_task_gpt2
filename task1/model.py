import torch
import torch.nn as nn
from torch.nn import functional as F
import os





torch.manual_seed(1337)
"""
This script downloads the Tiny Shakespeare dataset from URL
and saves it locally as a text file named "shakespeare.txt".

You can change the save location by modifying the DATA_PATH variable.
"""
import requests
import os

url = "https://raw.githubusercontent.com/karpathy/char-rnn/refs/heads/master/data/tinyshakespeare/input.txt"
DATA_PATH = "shakespeare.txt"
def download_dataset()->None:
    if os.path.exists(DATA_PATH):
        print("File already exits. No changes made.")
        return

    text_file = requests.get(url).text
    with open(DATA_PATH,"w") as f:
        f.write(text_file)


def load_dataset(print_text = False)->str:
    """
    Loads the dataset from DATA_PATH into a string.

    Args:
        print_text (bool): If True, prints the first 500 characters
                           of the dataset for preview.

    Returns:
        str: The full dataset content as a single string.
    """
    with open(DATA_PATH, "r") as f:
        txt = f.read()

    print("Total Characters in text: ", len(txt))
    if print_text == True:
        print(txt[:500])

    return txt

if __name__ == "__main__":
    download_dataset()
    load_dataset(print_text=True)


"""from data import load_dataset """
text = load_dataset()

chars = sorted(list(set(text)))
vocab_size = len(chars)

#tokenizer

def get_stats(ids):
  counts={}
  for pair in zip(ids,ids[1:]):
    counts[pair] = counts.get(pair,0)+1

  return counts


def merge(ids,pair,idx):
  newids = []
  i = 0
  while i <len(ids):
    if i<len(ids)-1 and ids[i]==pair[0] and ids[i+1] == pair[1]:
      newids.append(idx)
      i += 2
    else:
      newids.append(ids[i])
      i+=1

  return newids

vocab_size = 276
num_merges = vocab_size - 256
tokens = list(text.encode("utf-8")) # Initialize tokens here
ids = list(tokens)
merges = {}
for i in range (num_merges):
  stats = get_stats(ids)
  pair = max(stats,key = stats.get)
  idx = 256+i

  ids=merge(ids, pair, idx)
  merges[pair] = idx


vocab = {idx: bytes([idx]) for idx in range(256)}
for (p0, p1), idx in merges.items():
  vocab[idx] = vocab[p0]+vocab[p1]

def decode(ids):
  tokens = b"".join(vocab[idx] for idx in ids)
  text = tokens.decode("utf-8",errors ="replace")
  return text


def encode(text):
  token = list(text.encode("utf-8"))
  while len(token)>=2:
    stats = get_stats(token)
    pair = min(stats, key=lambda p: merges.get(p, float("inf")))
    if pair not in merges:
      break
    idx = merges[pair]
    token = merge(token, pair, idx)

  return token

#data into training and validation

data =torch.tensor(encode(text),dtype = torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
def get_batch(chose):
  data = train_data if chose == 'train' else val_data                
  ix = torch.randint(len(data)-block_size,(batch_size,))
  x = torch.stack([data[i:i+block_size] for i in ix])   #(B,T)
  y = torch.stack([data[i+1:i+1+block_size] for i in ix])   #(B,T)
  x,y = x.to(device), y.to(device)

  return x,y

#making of loss function
@torch.no_grad()
def estimate_loss():
  out = {}
  model.eval()
  for split in ['train','val']:
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
      X,Y = get_batch(split)               
      logits, loss =model(X,Y)
      losses[k]=loss.item()

    out[split] = losses.mean()

  model.train()
  return out


#creating self attention head block

class Head(nn.Module):

  def __init__(self,head_size):
    super().__init__()
    self.key = nn.Linear(n_embd,head_size,bias = False) 
    self.query= nn.Linear(n_embd,head_size,bias = False)
    self.value = nn.Linear(n_embd,head_size,bias = False)
    self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))
    self.dropout = nn.Dropout(dropout)  


  def forward(self,x):
    B, T,C = x.shape
    k = self.key(x) #(B,T,head_size)
    q = self.query(x) #(B,T,head_size)
    v = self.value(x) #(B,T,head_size)
    wei = q@k.transpose(-2,-1) * (k.size(-1))**(-0.5)   #(B,T,head_size)*(B,head_size,T)=(B,T,T)
    wei = wei.masked_fill(self.tril[:T,:T]==0,float('-inf'))  
    wei = F.softmax(wei,dim=-1)
    wei = self.dropout(wei)

    out = wei@v #(B,T,T)*(B,T,head_size) = (B,T,head_size)
    return out



class MultiheadAttention(nn.Module):
  def __init__(self,num_head,head_size):
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size) for _ in range(num_head)])
    self.proj = nn.Linear(n_embd,n_embd)        
    self.dropout = nn.Dropout(dropout)


  def forward(self,x):
    out = torch.cat([h(x) for h in self.heads],dim=-1) #(B,T,head_size)+B,T,head_size)+......n times = (B,T,head_size*num_head)
    out = self.proj(out)      #(B,T,n_embd)
    out = self.dropout(out)

    return out



class FeedForward(nn.Module):
  def __init__(self):
    super().__init__()
    self.net = nn.Sequential (
        nn.Linear(n_embd,4*n_embd),
        nn.ReLU(),
        nn.Linear(4*n_embd,n_embd),
        nn.Dropout(dropout)

    )


  def forward(self, x):
    out = self.net(x)
    return out



class Block(nn.Module):
  def __init__(self, num_head, head_size):
    super().__init__()
    self.mh = MultiheadAttention(num_head,head_size)
    self.ffwd = FeedForward()
    self.ln1 = nn.LayerNorm(n_embd) #normalistion
    self.ln2 = nn.LayerNorm(n_embd)


  def forward(self, x):
    x = x + self.mh(self.ln1(x))
    x = x + self.ffwd(self.ln2(x)) 
    return x




class BigramLanguageModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.embedding_table = nn.Embedding(vocab_size, n_embd)  
    self.positioning_table = nn.Embedding(block_size, n_embd) 
    head_size = n_embd // n_head
    self.blocks = nn.Sequential(*[Block(n_head, head_size=head_size) for _ in range(n_layer)])
    self.ln_f = nn.LayerNorm(n_embd) 
    self.linear = nn.Linear(n_embd,vocab_size)


  def forward(self,idx,targets = None):
    B,T = idx.shape
    
    tok_emb = self.embedding_table(idx) #(B,T,C)
    pos_emb = self.positioning_table(torch.arange(T,device=device))#(T,C)
    x = tok_emb+pos_emb#(B,T,C)
    x = self.blocks(x) #(B,T,C) 
    x =  self.ln_f(x) #(B,T,C)
    logits =  self.linear(x)#(B,T,vocab_size)

    if targets is None:
      loss = None

    else :
      B,T,C = logits.shape
      logits = logits.view((B*T),C )
      targets = targets.view(-1) #(B*T)
      loss = F.cross_entropy(logits,targets)

    return logits,loss


  def generate(self,idx,max_new_tokens):
    for _ in range(max_new_tokens):
      idx_cond = idx[:,-block_size:] 
      logits,loss = self(idx_cond)
      logits = logits[:,-1,:] #(B,vocab_size) 
      probs = F.softmax(logits,dim=-1) 
      idx_next = torch.multinomial(probs,num_samples = 1) 
      idx = torch.cat((idx,idx_next),dim = 1)


    return idx


model = BigramLanguageModel()
m = model.to(device)


optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

for iter in range(max_iters):
  if iter % eval_interval == 0:
    losses = estimate_loss()
    print(f"step {iter}: train loss {losses['train']:.4f},val loss {losses['val']:.4f}")


  xb,yb =get_batch('train')
  logits,loss = m(xb,yb)
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()




intial = "once upon a time"
idx = torch.tensor([encode(intial)],dtype=torch.long, device=device)
print(decode(m.generate(idx =idx,max_new_tokens=1000)[0].tolist()))