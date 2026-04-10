from datasets import load_dataset


PROMPT_WITH_INPUT = (
    "Below is an instruction that describes a task, paired with an input that "
    "provides further context. Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n{output}"
)

PROMPT_WITHOUT_INPUT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n{output}"
)


def format_alpaca_prompt(example: dict) -> dict:
    """ format a single alpaca example into the prompt template."""
    if example.get("input") and example["input"].strip():
        text = PROMPT_WITH_INPUT.format(
            instruction=example["instruction"],
            input=example["input"],
            output=example["output"],
        )
    else:
        text = PROMPT_WITHOUT_INPUT.format(
            instruction=example["instruction"],
            output=example["output"],
        )
    return {"text": text}


def load_alpaca_dataset(split: str = "train", test_size: float = 0.1, valid_size: float = 0.05, seed: int = 42):
    """
    load the alpaca dataset from HF and apply the prompt template.
    """
    dataset = load_dataset("tatsu-lab/alpaca", split="train")


    dataset = dataset.map(format_alpaca_prompt)

    if split == "all":
        return dataset

    temp_size = test_size + valid_size
    split_dataset = dataset.train_test_split(test_size=temp_size, seed=seed)

    train_dataset = split_dataset["train"]
    temp_dataset = split_dataset["test"]

    valid_ratio = valid_size/temp_size
    temp_split = temp_dataset.train_test_split(test_size=valid_ratio, seed=seed)
    valid_dataset = temp_split["train"]
    test_dataset = temp_split["test"]

    if split == "train":
      return train_dataset

    elif split == "valid":
      return valid_dataset
    elif split == "test":
      return test_dataset









if __name__ == "__main__":
    # load and print a few samples
    train_data = load_alpaca_dataset(split="train")
    valid_data = load_alpaca_dataset(split="valid")
    test_data = load_alpaca_dataset(split="test")

    print(f"\ntrain size: {len(train_data)}")
    print(f"valid size:  {len(valid_data)}")
    print(f"test size:  {len(test_data)}")
    print(f"\n{'='*60}")
    print("formatted prompt:\n")
    print(train_data[0]["text"])
import torch
torch.cuda.empty_cache()
from torch.utils.data import Dataset

class InstructionDataset(Dataset):
  def __init__(self, data, tokenizer):
    self.data = data

    self.encoded_texts = []
    for entry in data:

      full_text = format_alpaca_prompt(entry)

      self.encoded_texts.append(
          tokenizer.encode(full_text["text"])
      )

  def __getitem__(self, index):
    return self.encoded_texts[index]

  def __len__(self):
    return len(self.data)


import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")



def custom_collate_draft_1(
    batch,
    pad_token_id=50256,
    device="cpu"
):
    batch_max_length = max(len(item)+1 for item in batch)
    inputs_lst = []
    for item in batch:
      new_item = item.copy()
      new_item += [pad_token_id]
      padded = (
          new_item + [pad_token_id] *
          (batch_max_length - len(new_item))
      )

      inputs = torch.tensor(padded[:-1])
      inputs_lst.append(inputs)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    return inputs_tensor
def custom_collate_draft(
    batch,
    pad_token_id=50256,
    ignore_index= -100,
    allowed_max_length=None,
    device="cpu"
):
    batch_max_length = max(len(item)+1 for item in batch)
    inputs_lst,targets_lst= [],[]
    for item in batch:
      new_item = item.copy()
      new_item += [pad_token_id]
      padded = (
          new_item + [pad_token_id] *
          (batch_max_length - len(new_item))
      )

      inputs = torch.tensor(padded[:-1])
      targets = torch.tensor(padded[1:])
      mask = targets == pad_token_id
      indices = torch.nonzero(mask).squeeze()
      if indices.numel()>1:
        targets[indices[1:]] = ignore_index

      if allowed_max_length is not None:
        inputs = inputs[:allowed_max_length]
        targets = targets[:allowed_max_length]



      inputs_lst.append(inputs)
      targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from functools import partial
customized_collate_fn = partial(custom_collate_draft, device= device , allowed_max_length = 128)
from torch.utils.data import DataLoader
num_workers = 0
batch_size = 4
torch.manual_seed(123)

train_dataset = InstructionDataset(train_data, tokenizer)
train_loader = DataLoader(
    train_dataset,
    batch_size = batch_size,
    collate_fn = customized_collate_fn,
    shuffle=True,
    drop_last = True,
    num_workers=num_workers

)
val_dataset = InstructionDataset(valid_data, tokenizer)
val_loader = DataLoader(
    val_dataset,
    batch_size = batch_size,
    collate_fn = customized_collate_fn,
    shuffle=True,
    drop_last = True,
    num_workers=num_workers

)
test_dataset = InstructionDataset(test_data, tokenizer)
test_loader = DataLoader(
    test_dataset,
    batch_size = batch_size,
    collate_fn = customized_collate_fn,
    shuffle=True,
    drop_last = True,
    num_workers=num_workers

)
from transformers import GPT2Tokenizer, GPT2LMHeadModel
tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
model = GPT2LMHeadModel.from_pretrained("distilgpt2")
#padding
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id
#device
model.to(device)
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt').to(device)
output = model(**encoded_input)
print(output)


def calc_loss_batch(input_batch , target_batch, model, device):
  input_batch, target_batch = input_batch.to(device), target_batch.to(device)
  outputs = model(input_batch)
  logits = outputs.logits
  loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)),target_batch.view(-1))
  return loss
def calc_loss_loader(data_loader, model, device, num_batches=None):
  total_loss = 0.
  if len(data_loader) == 0:
    return float("nan")

  elif num_batches is None:
    num_batches = len(data_loader)

  else:
    num_batches = min(num_batches, len(data_loader))

  model.eval()
  with torch.no_grad():
    for i,(input_batch, target_batch) in enumerate(data_loader):
       if i < num_batches:
         loss = calc_loss_batch(input_batch, target_batch, model, device)
         total_loss += loss.item()

       else:
         break

  return total_loss / num_batches

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    return train_loss, val_loss

def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    input_ids = tokenizer.encode(start_context, return_tensors='pt').to(device)
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=100,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.95
        )
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"\n{'='*60}")
    print("Generated sample:\n")
    print(generated_text)
    print(f"\n{'='*60}")

def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter, start_context, tokenizer):
  train_losses, val_losses, track_tokens_seen = [],[],[]
  tokens_seen, global_step = 0,-1

  for epoch in range(num_epochs):
    model.train()
    for input_batch, target_batch in train_loader:
      optimizer.zero_grad()
      loss = calc_loss_batch(input_batch, target_batch, model, device)
      loss.backward()
      optimizer.step()
      tokens_seen += input_batch.numel()
      global_step += 1


      if global_step % eval_freq == 0:
        train_loss, val_loss = evaluate_model (
            model, train_loader, val_loader, device, eval_iter)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        track_tokens_seen.append(tokens_seen)
        print(f"Ep {epoch+1} (Step {global_step:06d}):"
        f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")


  generate_and_print_sample (
      model, tokenizer, device, start_context
  )
  return train_losses, val_losses, track_tokens_seen

model.to(device)
torch.manual_seed(123)
with torch.no_grad():
  train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
  val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)


print("trainloss", train_loss)

print("valloss", val_loss)
import time
start_time = time.time()
torch.manual_seed(123)
optimizer = torch.optim.AdamW(model.parameters(), lr = 0.00005, weight_decay = 0.1)
num_epochs = 1

val_data = load_alpaca_dataset(split="valid")
train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=1000, eval_iter=5,
    start_context=format_alpaca_prompt(val_data[0])["text"], tokenizer = tokenizer
)
end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.")