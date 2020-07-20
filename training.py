
import torch
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup

import numpy as np
import pandas as pd
from tqdm import tqdm_notebook as tqdm

from data_processing import get_paraphrase_data, AttrDict
from gpt2 import load_model

def ParaphraseDataset(tokenizer):
    
    train_df, test_df = get_paraphrase_data()
    df = pd.concat([train_df, test_df], ignore_index=True)
    df = df[ df["Quality"] == 1 ]

    tokenized_dataset = []
    for index, row in df.iterrows():
        try:
            input_str = "Paraphrase:\n" + row["#1 String"] + " => " + row["#2 String"]
            tokenized_dataset.append( tokenizer.encode(input_str, return_tensors="pt")[0] )
        except:
            # skip bad example
            pass

    samples_num = len(tokenized_dataset)
    max_tokens_num = max(map(len, tokenized_dataset))

    input_ids = np.full((samples_num, max_tokens_num), tokenizer.eos_token_id, dtype=np.int64)
    for i, tokens in enumerate(tokenized_dataset):
        input_ids[i, :len(tokens)] = tokens

    return torch.from_numpy(input_ids)

def pytorch_train(model, tokenizer, device, training_loader, opts):
    optimizer = AdamW(model.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)
    
    train_loss = []
    for e in tqdm(range(opts.epochs), total=opts.epochs):
        for input_ids in tqdm(training_loader, total=len(training_loader)):
            model.train()

            input_ids = input_ids.to(device)
            loss = model(input_ids, labels=input_ids)[0]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        train_loss.append(loss.item())
                
    return model, train_loss