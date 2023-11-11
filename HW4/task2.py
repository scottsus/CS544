# -*- coding: utf-8 -*-
"""task2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/14M0CKl5k8d1XbIL3qpp8K0jF2F6niDjX

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

"""## Data Preparation

### Load dataset
"""

import datasets

dataset = datasets.load_dataset("conll2003")
dataset

"""### Use GloVe embeddings"""

import numpy as np

vocab, embeddings = [], []
with open("glove.6B.100d.txt", "rt") as glove_file:
    full_content = glove_file.read().strip().split("\n")

for i in range(len(full_content)):
    word = full_content[i].split(" ")[0]
    embedding = [float(val) for val in full_content[i].split(" ")[1:]]
    vocab.append(word)
    embeddings.append(embedding)

vocab_npa = np.array(vocab)
embeddings_npa = np.array(embeddings)

vocab_npa = np.insert(vocab_npa, 0, "[PAD]")
vocab_npa = np.insert(vocab_npa, 1, "[UNK]")

pad_embeddings_npa = np.zeros((1, embeddings_npa.shape[1]))
unk_embeddings_npa = np.mean(embeddings_npa, axis=0, keepdims=True)

embeddings_npa = np.vstack((pad_embeddings_npa, unk_embeddings_npa, embeddings_npa))
embeddings_npa

word2idx = {word.lower(): idx for idx, word in enumerate(vocab, start=2)}

word2idx["[PAD]"] = 0
word2idx["[UNK]"] = 1

dataset = dataset.map(
    lambda x: {
        "input_ids": [
            word2idx.get(word.lower(), word2idx["[UNK]"]) for word in x["tokens"]
        ]
    }
)

dataset["train"]["input_ids"][:3]

dataset = dataset.rename_column("ner_tags", "labels")
dataset = dataset.remove_columns(["pos_tags", "chunk_tags"])
dataset

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""## Task 2: Bidirectional LSTM Model

### Define class
"""

import torch.nn as nn

embedding_dim = 100
num_lstm_layers = 1
lstm_hidden_dim = 256
lstm_dropout = 0.33
linear_output_dim = 128


class BiLSTMGlove(nn.Module):
    def __init__(self, vocab_size, num_classes):
        super(BiLSTMGlove, self).__init__()
        self.embedding = torch.nn.Embedding.from_pretrained(
            torch.from_numpy(embeddings_npa).float()
        )
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
model = BiLSTMGlove(vocab_size, num_classes)
model.to(device)

using_loaded_weights = False

model_path = "./weights/task2.pt"
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
dev_loader = DataLoader(
    dataset["validation"], batch_size, shuffle, collate_fn=collate_fn
)
test_loader = DataLoader(dataset["test"], batch_size, shuffle, collate_fn=collate_fn)


# Helper function to print green text
def print_green(text):
    print(f"\033[92m{text}\033[0m")


"""### Train model"""

import torch.optim as optim
from conlleval import evaluate


def train_model(model):
    print("Begin training BiLSTM with GloVe embeddings")

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

        early_stopping_epoch, min_f1 = 10, 88
        if epoch >= early_stopping_epoch and f1 >= min_f1:
            print_green("Expected F1 reached! 🚀" f"Epoch: {epoch+1}, F1: {f1}")
            break


"""### Train model and save weights"""

if not using_loaded_weights:
    print("Training model...")
    train_model(model)
    torch.save(model.state_dict(), model_path)
else:
    print("Using loaded model weights")

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