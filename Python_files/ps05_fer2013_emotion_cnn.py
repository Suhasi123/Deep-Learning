# %% [markdown]
# # PS5: Facial emotion recognition with a CNN (FER-2013)
# Dataset layout: FER-2013/train/<emotion>/*.jpg and FER-2013/test/<emotion>/*.jpg (48x48 grayscale).

# %%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras
from tensorflow.keras import layers

DATA_DIR = "../datasets/FER-2013"
EPOCHS = 25
IMG = (48, 48)

# %%
train_ds = keras.utils.image_dataset_from_directory(
    f"{DATA_DIR}/train", color_mode="grayscale", image_size=IMG, batch_size=64,
    validation_split=0.1, subset="training", seed=42)
val_ds = keras.utils.image_dataset_from_directory(
    f"{DATA_DIR}/train", color_mode="grayscale", image_size=IMG, batch_size=64,
    validation_split=0.1, subset="validation", seed=42)
test_ds = keras.utils.image_dataset_from_directory(
    f"{DATA_DIR}/test", color_mode="grayscale", image_size=IMG, batch_size=64, shuffle=False)
class_names = train_ds.class_names
print(class_names)

# %%
counts = [len(tf.io.gfile.listdir(f"{DATA_DIR}/train/{c}")) for c in class_names]
plt.bar(class_names, counts)
plt.title("Training images per emotion")
plt.show()

plt.figure(figsize=(10, 3))
for images, labels in train_ds.take(1):
    for i in range(7):
        plt.subplot(1, 7, i + 1)
        plt.imshow(images[i].numpy().squeeze(), cmap="gray")
        plt.title(class_names[labels[i]], fontsize=9)
        plt.axis("off")
plt.show()

train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)

# %%
model = keras.Sequential([
    layers.Input(IMG + (1,)),
    layers.Rescaling(1 / 255.0),
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.Conv2D(32, 3, padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.Dropout(0.3),
    layers.Flatten(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(len(class_names), activation="softmax"),
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

# %%
history = model.fit(
    train_ds, validation_data=val_ds, epochs=EPOCHS,
    callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)], verbose=2)

# %%
loss, acc = model.evaluate(test_ds, verbose=0)
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
y_true = np.concatenate([labels.numpy() for _, labels in test_ds])
y_pred = model.predict(test_ds, verbose=0).argmax(axis=1)
print(classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0))

plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion matrix")
plt.show()

# %%
images, labels = next(iter(test_ds.shuffle(100)))
preds = model.predict(images, verbose=0).argmax(axis=1)
plt.figure(figsize=(12, 3))
for i in range(8):
    plt.subplot(1, 8, i + 1)
    plt.imshow(images[i].numpy().squeeze(), cmap="gray")
    plt.title(f"P:{class_names[preds[i]]}\nA:{class_names[labels[i]]}", fontsize=8)
    plt.axis("off")
plt.show()
