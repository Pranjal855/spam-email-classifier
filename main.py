# ============================================================
# SMS SPAM DETECTION USING DEEP LEARNING
# PyTorch TextCNN
# Dataset: mail_data.csv
# ============================================================

import os
import re
import json
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 1. SETTINGS
# ============================================================

DATA_FILE = "mail_data.csv"

MODEL_FILE = "spam_model.pth"
VOCAB_FILE = "vocabulary.json"

MAX_VOCAB_SIZE = 10000
MAX_LENGTH = 100

EMBEDDING_DIM = 128
NUM_FILTERS = 64

BATCH_SIZE = 64
EPOCHS = 8
LEARNING_RATE = 0.001

SEED = 42


# ============================================================
# 2. RANDOM SEED
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# 3. DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("SMS SPAM DETECTION - DEEP LEARNING")
print("=" * 60)

print("Device:", device)


# ============================================================
# 4. LOAD DATASET
# ============================================================

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(
        f"ERROR: {DATA_FILE} not found.\n"
        "Put mail_data.csv in the same folder as this Python file."
    )


try:
    df = pd.read_csv(
        DATA_FILE,
        encoding="utf-8"
    )

except UnicodeDecodeError:
    df = pd.read_csv(
        DATA_FILE,
        encoding="latin-1"
    )


print("\nDataset shape:")
print(df.shape)

print("\nColumns:")
print(df.columns)


# ============================================================
# 5. CHECK REQUIRED COLUMNS
# ============================================================

if "Category" not in df.columns:
    raise ValueError("Column 'Category' not found.")

if "Message" not in df.columns:
    raise ValueError("Column 'Message' not found.")


# ============================================================
# 6. KEEP REQUIRED COLUMNS
# ============================================================

df = df[
    ["Category", "Message"]
].copy()


# ============================================================
# 7. CLEAN DATA
# ============================================================

df["Category"] = (
    df["Category"]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.lower()
)

df["Message"] = (
    df["Message"]
    .fillna("")
    .astype(str)
    .str.strip()
)


# Remove empty messages
df = df[
    df["Message"] != ""
].copy()


# Keep only HAM and SPAM
df = df[
    df["Category"].isin(
        ["ham", "spam"]
    )
].copy()


# ============================================================
# 8. CREATE LABEL
# ============================================================

# HAM  = 0
# SPAM = 1

df["label"] = (
    df["Category"] == "spam"
).astype(np.int64)


print("\nClass distribution:")
print(
    df["Category"].value_counts()
)


# ============================================================
# 9. TOKENIZER
# ============================================================

def tokenize(text):

    text = str(text).lower()

    return re.findall(
        r"[a-z0-9']+",
        text
    )


# ============================================================
# 10. BUILD VOCABULARY
# ============================================================

from collections import Counter

counter = Counter()

for message in df["Message"].tolist():

    counter.update(
        tokenize(message)
    )


vocabulary = {
    "<PAD>": 0,
    "<UNK>": 1
}


for word, count in counter.most_common(
    MAX_VOCAB_SIZE - 2
):

    vocabulary[word] = len(
        vocabulary
    )


print(
    "\nVocabulary size:",
    len(vocabulary)
)


# Save vocabulary
with open(
    VOCAB_FILE,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        vocabulary,
        file,
        ensure_ascii=False
    )


# ============================================================
# 11. ENCODE TEXT
# ============================================================

def encode_text(text):

    tokens = tokenize(text)

    sequence = []

    for token in tokens:

        sequence.append(
            vocabulary.get(
                token,
                vocabulary["<UNK>"]
            )
        )


    # Limit length
    sequence = sequence[
        :MAX_LENGTH
    ]


    # Padding
    if len(sequence) < MAX_LENGTH:

        sequence.extend(
            [vocabulary["<PAD>"]]
            * (
                MAX_LENGTH
                - len(sequence)
            )
        )


    return sequence


# ============================================================
# 12. TRAIN / VALIDATION / TEST SPLIT
# ============================================================

texts = df["Message"].to_numpy()

labels = df["label"].to_numpy(
    dtype=np.int64
)


train_texts, temp_texts, train_labels, temp_labels = train_test_split(

    texts,

    labels,

    test_size=0.20,

    random_state=SEED,

    stratify=labels
)


validation_texts, test_texts, validation_labels, test_labels = train_test_split(

    temp_texts,

    temp_labels,

    test_size=0.50,

    random_state=SEED,

    stratify=temp_labels
)


print("\nData split:")

print(
    "Training:",
    len(train_texts)
)

print(
    "Validation:",
    len(validation_texts)
)

print(
    "Testing:",
    len(test_texts)
)


# ============================================================
# 13. PYTORCH DATASET
# ============================================================

class SMSDataset(Dataset):

    def __init__(
        self,
        texts,
        labels,
        vocabulary
    ):

        # Convert texts to normal Python strings
        self.texts = [
            str(text)
            for text in texts
        ]

        # IMPORTANT FIX:
        # Convert labels explicitly to NumPy float32
        # before creating the PyTorch tensor.

        self.labels = np.asarray(
            labels,
            dtype=np.float32
        )


        self.vocabulary = vocabulary


    def __len__(self):

        return len(self.labels)


    def __getitem__(self, index):

        text = self.texts[index]

        label = self.labels[index]


        encoded_text = encode_text(
            text
        )


        # Convert encoded text to tensor
        x = torch.tensor(
            encoded_text,
            dtype=torch.long
        )


        # Convert SINGLE label to Python float
        # This avoids the pandas/object dtype problem.
        y = torch.tensor(
            float(label),
            dtype=torch.float32
        )


        return x, y


# ============================================================
# 14. CREATE DATASETS
# ============================================================

train_dataset = SMSDataset(
    train_texts,
    train_labels,
    vocabulary
)

validation_dataset = SMSDataset(
    validation_texts,
    validation_labels,
    vocabulary
)

test_dataset = SMSDataset(
    test_texts,
    test_labels,
    vocabulary
)


print("\nDatasets created successfully.")


# ============================================================
# 15. DATA LOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# 16. TEXT CNN MODEL
# ============================================================

class SpamCNN(nn.Module):

    def __init__(
        self,
        vocabulary_size
    ):

        super().__init__()


        # ----------------------------------------------------
        # Embedding layer
        # ----------------------------------------------------

        self.embedding = nn.Embedding(
            num_embeddings=vocabulary_size,
            embedding_dim=EMBEDDING_DIM,
            padding_idx=0
        )


        # ----------------------------------------------------
        # CNN layers
        # ----------------------------------------------------

        self.conv3 = nn.Conv1d(
            EMBEDDING_DIM,
            NUM_FILTERS,
            kernel_size=3
        )

        self.conv4 = nn.Conv1d(
            EMBEDDING_DIM,
            NUM_FILTERS,
            kernel_size=4
        )

        self.conv5 = nn.Conv1d(
            EMBEDDING_DIM,
            NUM_FILTERS,
            kernel_size=5
        )


        # ----------------------------------------------------
        # Dropout
        # ----------------------------------------------------

        self.dropout = nn.Dropout(
            0.4
        )


        # ----------------------------------------------------
        # Fully connected layer
        # ----------------------------------------------------

        self.fc = nn.Linear(
            NUM_FILTERS * 3,
            1
        )


    def forward(self, x):

        # Embedding
        #
        # [batch, sequence]
        #
        # becomes
        #
        # [batch, sequence, embedding]

        x = self.embedding(x)


        # CNN needs:
        #
        # [batch, embedding, sequence]

        x = x.permute(
            0,
            2,
            1
        )


        # ====================================================
        # CNN 3
        # ====================================================

        x3 = self.conv3(x)

        x3 = torch.relu(x3)

        x3 = torch.max(
            x3,
            dim=2
        ).values


        # ====================================================
        # CNN 4
        # ====================================================

        x4 = self.conv4(x)

        x4 = torch.relu(x4)

        x4 = torch.max(
            x4,
            dim=2
        ).values


        # ====================================================
        # CNN 5
        # ====================================================

        x5 = self.conv5(x)

        x5 = torch.relu(x5)

        x5 = torch.max(
            x5,
            dim=2
        ).values


        # ====================================================
        # Combine
        # ====================================================

        x = torch.cat(
            [
                x3,
                x4,
                x5
            ],
            dim=1
        )


        # Dropout
        x = self.dropout(x)


        # Output
        x = self.fc(x)


        return x.squeeze(1)


# ============================================================
# 17. CREATE MODEL
# ============================================================

model = SpamCNN(
    vocabulary_size=len(vocabulary)
)

model = model.to(device)


print("\nModel:")
print(model)


# ============================================================
# 18. LOSS FUNCTION
# ============================================================

spam_count = np.sum(
    train_labels == 1
)

ham_count = np.sum(
    train_labels == 0
)


pos_weight = torch.tensor(
    [
        ham_count /
        max(spam_count, 1)
    ],
    dtype=torch.float32,
    device=device
)


criterion = nn.BCEWithLogitsLoss(
    pos_weight=pos_weight
)


# ============================================================
# 19. OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# 20. EVALUATION
# ============================================================

def evaluate(data_loader):

    model.eval()

    all_labels = []

    all_predictions = []

    all_probabilities = []


    with torch.no_grad():

        for inputs, labels in data_loader:

            inputs = inputs.to(device)


            outputs = model(
                inputs
            )


            probabilities = torch.sigmoid(
                outputs
            )


            predictions = (
                probabilities >= 0.5
            ).long()


            all_labels.extend(
                labels.numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_probabilities.extend(
                probabilities.cpu().numpy()
            )


    all_labels = np.asarray(
        all_labels,
        dtype=np.int64
    )

    all_predictions = np.asarray(
        all_predictions,
        dtype=np.int64
    )

    all_probabilities = np.asarray(
        all_probabilities,
        dtype=np.float32
    )


    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    precision = precision_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    recall = recall_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        zero_division=0
    )


    return (
        accuracy,
        precision,
        recall,
        f1,
        all_labels,
        all_predictions,
        all_probabilities
    )


# ============================================================
# 21. TRAINING
# ============================================================

print("\n")
print("=" * 60)
print("STARTING TRAINING")
print("=" * 60)


best_f1 = -1

best_state = None


for epoch in range(
    EPOCHS
):

    model.train()

    total_loss = 0.0


    for inputs, labels in train_loader:

        inputs = inputs.to(device)

        labels = labels.to(device)


        # Clear gradients
        optimizer.zero_grad()


        # Forward
        outputs = model(
            inputs
        )


        # Loss
        loss = criterion(
            outputs,
            labels
        )


        # Backpropagation
        loss.backward()


        # Update
        optimizer.step()


        total_loss += (
            loss.item()
            * inputs.size(0)
        )


    average_loss = (
        total_loss
        / len(train_loader.dataset)
    )


    # Validation
    (
        val_accuracy,
        val_precision,
        val_recall,
        val_f1,
        _,
        _,
        _
    ) = evaluate(
        validation_loader
    )


    print(
        f"\nEpoch {epoch + 1}/{EPOCHS}"
    )

    print(
        f"Loss      : {average_loss:.4f}"
    )

    print(
        f"Val Acc   : {val_accuracy:.4f}"
    )

    print(
        f"Val Prec  : {val_precision:.4f}"
    )

    print(
        f"Val Recall: {val_recall:.4f}"
    )

    print(
        f"Val F1    : {val_f1:.4f}"
    )


    # Save best model
    if val_f1 > best_f1:

        best_f1 = val_f1

        best_state = {
            key: value.detach().cpu().clone()
            for key, value
            in model.state_dict().items()
        }


# ============================================================
# 22. LOAD BEST MODEL
# ============================================================

if best_state is not None:

    model.load_state_dict(
        best_state
    )


# ============================================================
# 23. SAVE MODEL
# ============================================================

torch.save(
    model.state_dict(),
    MODEL_FILE
)


print(
    "\nModel saved successfully:"
)

print(
    MODEL_FILE
)


# ============================================================
# 24. TEST MODEL
# ============================================================

(
    test_accuracy,
    test_precision,
    test_recall,
    test_f1,
    true_labels,
    predictions,
    probabilities
) = evaluate(
    test_loader
)


print("\n")
print("=" * 60)
print("FINAL TEST RESULTS")
print("=" * 60)


print(
    f"Accuracy  : {test_accuracy:.4f}"
)

print(
    f"Precision : {test_precision:.4f}"
)

print(
    f"Recall    : {test_recall:.4f}"
)

print(
    f"F1 Score  : {test_f1:.4f}"
)


# ============================================================
# 25. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        true_labels,
        predictions,
        target_names=[
            "HAM",
            "SPAM"
        ],
        digits=4,
        zero_division=0
    )
)


# ============================================================
# 26. CONFUSION MATRIX
# ============================================================

print("\nConfusion Matrix:")

cm = confusion_matrix(
    true_labels,
    predictions
)

print(cm)


# ============================================================
# 27. PREDICT NEW SMS
# ============================================================

def predict_message(message):

    model.eval()


    encoded = encode_text(
        message
    )


    x = torch.tensor(
        [encoded],
        dtype=torch.long
    ).to(device)


    with torch.no_grad():

        output = model(x)

        probability = torch.sigmoid(
            output
        ).item()


    if probability >= 0.5:

        prediction = "SPAM"

    else:

        prediction = "HAM"


    return prediction, probability


# ============================================================
# 28. INTERACTIVE PREDICTION
# ============================================================

print("\n")
print("=" * 60)
print("SMS PREDICTION")
print("=" * 60)

print(
    "Enter an SMS to classify it."
)

print(
    "Type 'exit' to stop."
)


while True:

    message = input(
        "\nEnter SMS: "
    )


    if message.strip().lower() == "exit":

        print(
            "Program stopped."
        )

        break


    if not message.strip():

        print(
            "Please enter a message."
        )

        continue


    prediction, probability = predict_message(
        message
    )


    print(
        "\nPrediction:",
        prediction
    )

    print(
        "Spam probability:",
        f"{probability:.2%}"
    )