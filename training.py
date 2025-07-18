import nltk
from nltk.corpus import brown
import re
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast
from torch.optim import AdamW
import torch.nn.functional as F
from typing import Optional

# constructing and cleansing dataset

def clean_text(text):
    text = re.sub(r'\s+([,.!?:;…])', r'\1', text)
    text = text.replace("``", '"').replace("''", '"')
    return text

nltk.download('brown')
paras = brown.paras()
sents = brown.sents()

texts = [" ".join([w for sent in para for w in sent]) for para in brown.paras()]
texts = texts + [" ".join(sent) for sent in brown.sents()]
texts = [clean_text(text) for text in texts]

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, block_size=128):
        self.examples = []
        for text in texts:
            enc = tokenizer(
                text,
                truncation=True,
                max_length=block_size,
                padding='max_length',
                return_tensors='pt'
            )
            input_ids = enc.input_ids.squeeze(0)
            attention_mask = enc.attention_mask.squeeze(0)
            labels = input_ids.clone()
            self.examples.append((input_ids, attention_mask, labels))
    def __len__(self):
        return len(self.examples)
    def __getitem__(self, idx):
        return self.examples[idx]

dataset = TextDataset(texts, tokenizer, block_size=128)
loader  = DataLoader(dataset, batch_size=4, shuffle=True)

config = GPT2Config(
    vocab_size=tokenizer.vocab_size,
    n_positions=128,
    n_ctx=128,
    n_embd=256,
    n_layer=2,
    n_head=4,
    pad_token_id=tokenizer.pad_token_id
)
model = GPT2LMHeadModel(config)
model.resize_token_embeddings(len(tokenizer))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.train()

optimizer = AdamW(model.parameters(), lr=5e-5)

print_every = 100
num_epochs = 1

for epoch in range(num_epochs):
    total_loss = 0.0
    local_loss = 0.0
    local_count = 0

    for i, (input_ids, attention_mask, labels) in enumerate(loader, start=1):
        input_ids      = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels         = labels.to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        batch_loss = loss.item()
        total_loss += batch_loss
        local_loss += batch_loss
        local_count += 1

        if i % print_every == 0:
            avg_local_loss = local_loss / local_count
            print(f"Epoch {epoch+1} | Batch {i}/{len(loader)} | "
                  f"Avg Loss (last {local_count} batches): {avg_local_loss:.4f}")
            local_loss = 0.0
            local_count = 0

    if local_count > 0:
        avg_local_loss = local_loss / local_count
        print(f"Epoch {epoch+1} | End of Epoch | "
              f"Avg Loss (last {local_count} batches): {avg_local_loss:.4f}")

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1} — Average Loss over all batches: {avg_loss:.4f}")

def evaluate_cross_entropy(
    sentences: list[str],
    model: torch.nn.Module,
    tokenizer,
    device: Optional[torch.device] = None
) -> torch.Tensor:

    device = device or next(model.parameters()).device
    model.eval()

    enc = tokenizer(
        sentences,
        return_tensors="pt",
        padding=True,
        truncation=True
    )
    input_ids = enc.input_ids.to(device)             # (B, T)
    mask      = enc.attention_mask.to(device).float()  # (B, T)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=mask,
            labels=input_ids
        )
        logits = outputs.logits                       # (B, T, V)

    B, T, V = logits.size()
    logits_flat = logits.view(-1, V)                 # (B*T, V)
    labels_flat = input_ids.view(-1)                 # (B*T)
    loss_flat = F.cross_entropy(
        logits_flat,
        labels_flat,
        reduction="none",
        ignore_index=tokenizer.pad_token_id
    ).view(B, T)                                     # (B, T)

    ce_per_sentence = (loss_flat * mask).sum(dim=1) / mask.sum(dim=1)  # (B,)
    return ce_per_sentence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def compute_dual_cross_entropy(
    sentence: str,
    model: torch.nn.Module,
    tokenizer,
    device: torch.device | None = None
) -> tuple[float, float]:

    device = device or next(model.parameters()).device
    model.to(device).eval()

    ce_custom = evaluate_cross_entropy([sentence], model, tokenizer, device)[0].item()

    gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    gpt2_model     = AutoModelForCausalLM.from_pretrained("gpt2").to(device).eval()

    if gpt2_tokenizer.pad_token_id is None:
        gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token

    ce_gpt2 = evaluate_cross_entropy([sentence], gpt2_model, gpt2_tokenizer, device)[0].item()

    return ce_custom, ce_gpt2

# Grammatical Sentences
sentence = "James went to Indonesia during his vacation."
ce_custom, ce_gpt2 = compute_dual_cross_entropy(sentence, model, tokenizer)
print(f"Custom model CE: {ce_custom:.4f} nats/tok")
print(f"GPT-2 CE:        {ce_gpt2:.4f} nats/tok")
sentence = "When smelted into an alloy, copper and tin produce incredible tensile strength."
ce_custom, ce_gpt2 = compute_dual_cross_entropy(sentence, model, tokenizer)
print(f"Custom model CE: {ce_custom:.4f} nats/tok")
print(f"GPT-2 CE:        {ce_gpt2:.4f} nats/tok")
sentence = "I went shopping yesterday. Sadly, the store was out of apples."
ce_custom, ce_gpt2 = compute_dual_cross_entropy(sentence, model, tokenizer)
print(f"Custom model CE: {ce_custom:.4f} nats/tok")
print(f"GPT-2 CE:        {ce_gpt2:.4f} nats/tok")

# Nonsense Sentences
sentence = "and the and the and the and the"
ce_custom, ce_gpt2 = compute_dual_cross_entropy(sentence, model, tokenizer)
print(f"Custom model CE: {ce_custom:.4f} nats/tok")
print(f"GPT-2 CE:        {ce_gpt2:.4f} nats/tok")
sentence = "teij49dgowigen"
ce_custom, ce_gpt2 = compute_dual_cross_entropy(sentence, model, tokenizer)
print(f"Custom model CE: {ce_custom:.4f} nats/tok")
print(f"GPT-2 CE:        {ce_gpt2:.4f} nats/tok")
sentence = "moonbeam icecream"
ce_custom, ce_gpt2 = compute_dual_cross_entropy(sentence, model, tokenizer)
print(f"Custom model CE: {ce_custom:.4f} nats/tok")
print(f"GPT-2 CE:        {ce_gpt2:.4f} nats/tok")
