# -*- coding: utf-8 -*-
"""task1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1a4IW-L6qvwkc3HwngGfmdTKA8zctqaPa

# HW4: Deep Learning on NER

### Setup environment
"""

""" Google colab specific code for env
!pip install -q datasets accelerate

!wget http://nlp.stanford.edu/data/glove.6B.zip

!unzip glove.6B.zip

!wget https://raw.githubusercontent.com/sighsmile/conlleval/master/conlleval.py

!ls # Sanity check
"""

"""## Task 0: Prepare Data

### Load dataset
"""

import datasets

dataset = datasets.load_dataset("conll2003")
dataset

"""### Create vocabulary"""

import itertools
from collections import Counter

word_freq = Counter(itertools.chain(*dataset["train"]["tokens"]))  # type: ignore

word_freq = {word: freq for word, freq in word_freq.items() if freq >= 3}

word2idx = {word: idx for idx, word in enumerate(word_freq.keys(), start=2)}

word2idx["[PAD]"] = 0
word2idx["[UNK]"] = 1

"""### Tokenize to ids"""

dataset = dataset.map(
    lambda x: {
        "input_ids": [word2idx.get(word, word2idx["[UNK]"]) for word in x["tokens"]]
    }
)

dataset["train"]["input_ids"][:3]

dataset = dataset.rename_column("ner_tags", "labels")
dataset = dataset.remove_columns(["pos_tags", "chunk_tags"])
dataset

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""## Task 1: Bidirectional LSTM Model

### Define class
"""

import torch.nn as nn

embedding_dim = 100
num_lstm_layers = 1
lstm_hidden_dim = 256
lstm_dropout = 0.33
linear_output_dim = 128


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, num_classes):
        super(BiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.bi_lstm = nn.LSTM(
            embedding_dim,
            lstm_hidden_dim,
            num_lstm_layers,
            bidirectional=True,
            batch_first=True,
        )
        self.linear = nn.Linear(lstm_hidden_dim * 2, linear_output_dim)
        self.elu = nn.ELU()
        self.classifier = nn.Linear(linear_output_dim, num_classes)

    def forward(self, x):
        embedding = self.embedding(x)
        output, _ = self.bi_lstm(embedding)
        output = self.linear(output)
        output = self.elu(output)
        output = self.classifier(output)

        return output


import os

vocab_size, num_classes = len(word2idx), 9
model = BiLSTM(vocab_size, num_classes)
model.to(device)

using_loaded_weights = False

model_path = "./weights/task1.pt"
if os.path.exists(model_path):
    using_loaded_weights = True
    model.load_state_dict(torch.load(model_path))
    print(f"Model loaded from {model_path}")

model

"""### Build train set"""

from torch.nn.utils.rnn import pad_sequence


def collate_fn(batch):
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]

    batch_first = True
    input_ids = pad_sequence(
        [torch.tensor(seq) for seq in input_ids], batch_first, padding_value=0
    )
    labels = pad_sequence(
        [torch.tensor(seq) for seq in labels], batch_first, padding_value=9
    )

    return {"input_ids": input_ids, "labels": labels}


from torch.utils.data import DataLoader

batch_size = 32
shuffle = True

train_loader = DataLoader(dataset["train"], batch_size, shuffle, collate_fn=collate_fn)
dev_loader = DataLoader(dataset["validation"], batch_size, collate_fn=collate_fn)
test_loader = DataLoader(dataset["test"], batch_size, collate_fn=collate_fn)


# Helper function to print green text
def print_green(text):
    print(f"\033[92m{text}\033[0m")


"""### Train model"""

import torch.optim as optim
from conlleval import evaluate


def train_model(model):
    print("Begin training BiLSTM")

    lr = 1e-3
    loss_fn = nn.CrossEntropyLoss(ignore_index=9)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    tag_to_index = {
        "O": 0,
        "B-PER": 1,
        "I-PER": 2,
        "B-ORG": 3,
        "I-ORG": 4,
        "B-LOC": 5,
        "I-LOC": 6,
        "B-MISC": 7,
        "I-MISC": 8,
    }
    index_to_tag = {index: tag for tag, index in tag_to_index.items()}

    num_epochs = 20
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss_total = 0
        for batch in train_loader:
            inputs = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs.permute(0, 2, 1), labels.long())
            loss.backward()
            optimizer.step()

            train_loss_total += loss.item()

        train_loss_ave = train_loss_total / len(train_loader)
        print(f"\nEpoch {epoch+1}/{num_epochs}, train loss: {train_loss_ave:.4f}")

        # Evaluation phase
        model.eval()
        dev_loss_total = 0
        pred_tags = []
        true_tags = []
        with torch.no_grad():
            for batch in dev_loader:
                inputs = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(inputs)
                loss = loss_fn(outputs.permute(0, 2, 1), labels.long())
                dev_loss_total += loss.item()

                preds = torch.argmax(outputs, dim=2)
                for i in range(labels.size(0)):
                    pred_seq = preds[i].cpu().numpy()
                    true_seq = labels[i].cpu().numpy()

                    indices_valid = true_seq != 9
                    valid_pred_tags = [
                        index_to_tag[idx] for idx in pred_seq[indices_valid]
                    ]
                    valid_true_tags = [
                        index_to_tag[idx] for idx in true_seq[indices_valid]
                    ]

                    pred_tags.append(valid_pred_tags)
                    true_tags.append(valid_true_tags)

        dev_loss_ave = dev_loss_total / len(dev_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, dev loss: {dev_loss_ave:.4f}")

        # Calculate metrics
        pred_tags_flattened = []
        for valid_pred_tag in pred_tags:
            for tag in valid_pred_tag:
                pred_tags_flattened.append(tag)

        true_tags_flattened = []
        for valid_true_tag in true_tags:
            for tag in valid_true_tag:
                true_tags_flattened.append(tag)

        precision, recall, f1 = evaluate(true_tags_flattened, pred_tags_flattened)
        print(
            f"Epoch {epoch+1}/{num_epochs}, Precision: {precision}, Recall: {recall}, F1: {f1}"
        )

        early_stopping_epoch, min_f1 = 10, 77
        if epoch >= early_stopping_epoch and f1 >= min_f1:
            print_green("Expected F1 reached! 🚀" f"Epoch: {epoch+1}, F1: {f1}")
            break


if not using_loaded_weights:
    print("Training model...")
    train_model(model)
    torch.save(model.state_dict(), model_path)
else:
    print("Using loaded model wieghts")

"""### Evaluate model"""


def test_model(model, loader, desc):
    tag_to_index = {
        "O": 0,
        "B-PER": 1,
        "I-PER": 2,
        "B-ORG": 3,
        "I-ORG": 4,
        "B-LOC": 5,
        "I-LOC": 6,
        "B-MISC": 7,
        "I-MISC": 8,
    }
    index_to_tag = {index: tag for tag, index in tag_to_index.items()}

    # Testing phase
    model.eval()
    pred_tags = []
    true_tags = []
    with torch.no_grad():
        for batch in loader:
            inputs = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=2)
            for i in range(labels.size(0)):
                pred_seq = preds[i].cpu().numpy()
                true_seq = labels[i].cpu().numpy()

                indices_valid = true_seq != 9
                valid_pred_tags = [index_to_tag[idx] for idx in pred_seq[indices_valid]]
                valid_true_tags = [index_to_tag[idx] for idx in true_seq[indices_valid]]

                pred_tags.append(valid_pred_tags)
                true_tags.append(valid_true_tags)

    # Calculate metrics
    pred_tags_flattened = []
    for valid_pred_tag in pred_tags:
        for tag in valid_pred_tag:
            pred_tags_flattened.append(tag)

    true_tags_flattened = []
    for valid_true_tag in true_tags:
        for tag in valid_true_tag:
            true_tags_flattened.append(tag)

    precision, recall, f1 = evaluate(true_tags_flattened, pred_tags_flattened)
    print_green(f"{desc} Data:\n" f"Precision: {precision}, Recall: {recall}, F1: {f1}")


test_model(model, train_loader, "Train")
test_model(model, dev_loader, "Validation")
test_model(model, test_loader, "Test")
