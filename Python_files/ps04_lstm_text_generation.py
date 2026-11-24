# %% [markdown]
# # PS4: LSTM for character-level text generation
# Corpus: Shakespeare text, downloaded automatically.

# %%
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

EPOCHS = 15
SEQ_LEN = 40

# %%
path = keras.utils.get_file(
    "shakespeare.txt", "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt")
text = open(path, encoding="utf-8").read().lower()[:200000]
chars = sorted(set(text))
char_to_id = {c: i for i, c in enumerate(chars)}
print("Characters:", len(text), " Vocabulary:", len(chars))
print(text[:300])

# %%
encoded = np.array([char_to_id[c] for c in text])
starts = range(0, len(encoded) - SEQ_LEN, 3)
X = np.array([encoded[i:i + SEQ_LEN] for i in starts])
y = np.array([encoded[i + SEQ_LEN] for i in starts])
print("X:", X.shape, " y:", y.shape)

# %%
model = keras.Sequential([
    layers.Input((None,)),
    layers.Embedding(len(chars), 64),
    layers.LSTM(256),
    layers.Dense(len(chars), activation="softmax"),
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

# %%
history = model.fit(X, y, epochs=EPOCHS, batch_size=128, validation_split=0.1, verbose=2)

# %%
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for i, metric in enumerate(["loss", "accuracy"]):
    ax[i].plot(history.history[metric], label="train")
    ax[i].plot(history.history["val_" + metric], label="validation")
    ax[i].set_title(metric)
    ax[i].set_xlabel("epoch")
    ax[i].legend()
plt.show()
print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
print(f"Validation perplexity: {np.exp(history.history['val_loss'][-1]):.2f}")


# %%
def generate(seed, length=300, temperature=0.5):
    out = seed.lower()
    for _ in range(length):
        x = np.array([[char_to_id[c] for c in out[-SEQ_LEN:]]])
        probs = model(x, training=False).numpy()[0].astype("float64")
        probs = np.exp(np.log(probs + 1e-9) / temperature)
        probs /= probs.sum()
        out += chars[np.random.choice(len(chars), p=probs)]
    return out


seed = text[1000:1000 + SEQ_LEN]
for t in [0.2, 0.5, 1.0]:
    print(f"--- temperature {t} ---")
    print(generate(seed, temperature=t))
    print()
