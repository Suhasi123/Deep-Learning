# %% [markdown]
# # PS11: Music genre classification with an LSTM (GTZAN)
# Dataset layout (Kaggle GTZAN): Data/genres_original/<genre>/*.wav  (10 genres x 100 clips of 30 s)
# Each clip is cut into 10 segments of 3 s; each segment becomes a sequence of MFCC frames (130 x 13).

# %%
import os

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

DATA_DIR = "../datasets/gtzan/Data/genres_original"
EPOCHS = 40
SR = 22050
SEGMENTS = 10
N_MFCC = 13
SEG_LEN = SR * 30 // SEGMENTS

# %%
genres = sorted(os.listdir(DATA_DIR))
files, labels = [], []
for label, genre in enumerate(genres):
    for f in sorted(os.listdir(os.path.join(DATA_DIR, genre))):
        files.append(os.path.join(DATA_DIR, genre, f))
        labels.append(label)
print(genres, "files:", len(files))

signal, _ = librosa.load(files[0], sr=SR, duration=30)
mfcc = librosa.feature.mfcc(y=signal, sr=SR, n_mfcc=N_MFCC)
fig, ax = plt.subplots(1, 2, figsize=(12, 3))
ax[0].plot(signal)
ax[0].set_title(f"Waveform ({genres[labels[0]]})")
librosa.display.specshow(mfcc, sr=SR, x_axis="time", ax=ax[1])
ax[1].set_title("MFCC")
plt.show()


# %%
def extract(paths, labels):
    X, y = [], []
    for path, label in zip(paths, labels):
        try:
            signal, _ = librosa.load(path, sr=SR, duration=30)
        except Exception:
            print("Skipping unreadable file:", path)
            continue
        for s in range(SEGMENTS):
            seg = signal[s * SEG_LEN:(s + 1) * SEG_LEN]
            if len(seg) == SEG_LEN:
                X.append(librosa.feature.mfcc(y=seg, sr=SR, n_mfcc=N_MFCC).T)
                y.append(label)
    return np.array(X), np.array(y)


# Split by song so segments of one song never appear in both train and test.
f_train, f_test, l_train, l_test = train_test_split(files, labels, test_size=0.2, stratify=labels, random_state=42)
X_train, y_train = extract(f_train, l_train)
X_test, y_test = extract(f_test, l_test)

mean, std = X_train.mean(axis=(0, 1)), X_train.std(axis=(0, 1))
X_train, X_test = (X_train - mean) / std, (X_test - mean) / std
print("Train:", X_train.shape, " Test:", X_test.shape)

# %%
model = keras.Sequential([
    layers.Input(X_train.shape[1:]),
    layers.LSTM(128, return_sequences=True),
    layers.LSTM(64),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(len(genres), activation="softmax"),
])
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

# %%
history = model.fit(
    X_train, y_train, validation_split=0.1, epochs=EPOCHS, batch_size=32,
    callbacks=[keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)], verbose=2)

# %%
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {acc:.4f}   Test loss: {loss:.4f}")

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for i, metric in enumerate(["loss", "accuracy"]):
    ax[i].plot(history.history[metric], label="train")
    ax[i].plot(history.history["val_" + metric], label="validation")
    ax[i].set_title(metric)
    ax[i].set_xlabel("epoch")
    ax[i].legend()
plt.show()

# %%
y_pred = model.predict(X_test, verbose=0).argmax(axis=1)
print(classification_report(y_test, y_pred, target_names=genres, digits=4, zero_division=0))

plt.figure(figsize=(9, 7))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues",
            xticklabels=genres, yticklabels=genres)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion matrix (3 s segments)")
plt.show()


# %%
def predict_song(path):
    X, _ = extract([path], [0])
    if len(X) == 0:
        return "unreadable", 0.0
    probs = model.predict((X - mean) / std, verbose=0).mean(axis=0)
    return genres[probs.argmax()], probs.max()


for path, label in list(zip(f_test, l_test))[:5]:
    genre, conf = predict_song(path)
    print(f"{os.path.basename(path):20s} actual={genres[label]:10s} predicted={genre:10s} ({conf:.2f})")
