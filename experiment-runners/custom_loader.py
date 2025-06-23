import torch
from torch.utils.data import Dataset, DataLoader
import json, os
from transformers import AutoTokenizer
import math



class DisfluentSet(Dataset):
  def __init__(self, phase="train", text_key="plain_text"):  # train test
    self.audio_feats_path = "/ephemeral/features/"
    self.manifest_file = "../" + phase + "_manifest.txt"
    self.jsonl_path = "../all_" + phase + ".jsonl"
    self.textual_data = []
    self.text_key = text_key

    with open(self.jsonl_path, "r") as injs:
      for line in injs:
        chunk = json.loads(line)
        self.textual_data.append(chunk)

  def __len__(self):
    return len(self.textual_data)

  def __getitem__(self, index):

    chunk = self.textual_data[index]
    fname = chunk["filename"]
    textjs = chunk["text"]
    textual_input = textjs[self.text_key]
    ground_truth = textjs["ground_truth"]
    features_path = self.audio_feats_path + fname + "_features.pt"
    audio_features = torch.load(features_path)


    item = {
        "audio_features": audio_features,
        "textual_input": textual_input,
        "transcript": ground_truth
    }
    return item

model = "meta-llama/Llama-3.2-1B-Instruct"
llama_tokenizer = AutoTokenizer.from_pretrained(model)
llama_tokenizer.pad_token = llama_tokenizer.eos_token


def variable_batcher(batch):
    # Textual inputs
    text_prompts = [item["textual_input"] for item in batch]
    transcripts = [item["transcript"] for item in batch]

    full_texts = [prompt + " TRANSCRIPT: " + transcript for prompt, transcript in zip(text_prompts, transcripts)]
    tokenized = llama_tokenizer(full_texts, return_tensors="pt", padding=True, truncation=True)
    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]

    loss_mask = torch.ones_like(input_ids, dtype=torch.float32)
    for i, prompt in enumerate(text_prompts):
      prompt_len = len(llama_tokenizer(prompt, add_special_tokens=False)["input_ids"])
      loss_mask[i, :prompt_len+1] = 0  # Exclude prompt tokens from the loss, skip beginning of the text

    audio_feats = [item["audio_features"] for item in batch]
    audio_features = torch.stack(audio_feats, dim=0)
    return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,  # Attention mask
            "loss_mask":loss_mask,  # Loss mask
            "audio_features": audio_features,
        }


def dis_loader(phase, text_key, batch_size, shuffle=False):
  dataset = DisfluentSet(phase="train", text_key=text_key)
  return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=variable_batcher)


