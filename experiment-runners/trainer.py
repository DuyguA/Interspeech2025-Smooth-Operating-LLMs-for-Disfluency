import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
import numpy as np

import os, copy, json, csv
from tqdm import tqdm

from models import create_disfluency_model
from custom_loader import dis_loader
from loss_functions import compute_masked_loss
from torch.cuda.amp import autocast, GradScaler


class Trainer:
  def __init__(self, args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    self.train_loader = dis_loader("train", text_key=args.text_key, batch_size=args.train_batch_size, shuffle=True)
    self.valid_loader = dis_loader("test", text_key=args.text_key, batch_size=args.train_batch_size, shuffle=True)
    print('Dataset prep done!\n')

    if torch.cuda.device_count() > 0:
      print(f"{torch.cuda.device_count()} GPUs found")

    print('Initializing model....')
    input_dim = 1  
    conformer_hidden_dim = 1024  # Hidden dimension of the Conformer
    final_dim = 1024 
    model = create_disfluency_model(1, conformer_dim, final_dim)

    params = model.parameters()
    self.optimizer = AdamW(params, lr=args.lr, weight_decay=0.01)
    self.model = model
    self.scaler = GradScaler()


    self.args = args
    self.epoch_accuracies = []
    self.all_losses = []

  def train(self):
    best_epoch = 0
    print("First epoch will start soon")
    for epoch in range(self.args.epochs):
      print(f"{'*' * 20}Epoch: {epoch + 1}{'*' * 20}")
      loss = self.train_epoch()
      eval_loss  = self.eval()
      print(f'Eval loss: {eval_loss:.3f}')

  def train_epoch(self):
    device = self.device
    self.model.train()
    epoch_loss = 0
    loader = self.train_loader
    for i, batch in tqdm(enumerate(loader), total=len(loader)):
        audio_features = batch["audio_features"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        loss_mask = batch["loss_mask"]

        with autocast(dtype=torch.bfloat16):
          logits = self.model(audio_features, input_ids, attention_mask)

        effective_mask = loss_mask *  attention_mask 
        loss = compute_masked_loss(logits, input_ids, effective_mask)

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        optimizer.zero_grad()


        interval = max(len(loader) // 20, 1)
        if i % interval == 0 or i == len(loader) - 1:
            lloss = round(loss.item(), 3)
            print(f'Batch: {i + 1}/{len(loader)}\ttotal loss: {lloss:.3f}')
            self.all_losses.append(lloss) # append epoch losses
        epoch_loss += loss.item()
    return epoch_loss / len(loader)


  def eval(self):
    self.model.eval()
    device = self.device
    label_pred = []
    label_true = []

    loader = self.valid_loader 
    total_loss=0.0
    total_tokens=0

    with torch.no_grad():
      for i, batch in enumerate(loader):
        audio_features = batch["audio_features"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        loss_mask = batch["loss_mask"]

        logits = self.model(audio_features, input_ids, attention_mask)
        effective_mask = loss_mask *  attention_mask 
        loss = compute_masked_loss(logits, input_ids, effective_mask)
        total_loss += loss.item()
    return total_loss / len(loader)

      


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_batch_size', type=int, default=48, help='batch size of training')
    parser.add_argument('--val_batch_size', type=int, default=4, help='batch size of testing')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--text_key', type=str, default="plain_text")
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default="exp", help='Directory for saving the model')
    args = parser.parse_args()

    print(args)
    engine = Trainer(args)
    engine.train()

