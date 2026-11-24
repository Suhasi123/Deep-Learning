# %% [markdown]
# # PS2: Pre-trained model (MobileNetV2) with fine-tuning for image classification
# Dataset: TensorFlow flower photos (5 classes), downloaded automatically.

# %%
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras
from tensorflow.keras import layers

EPOCHS = 5
FINE_EPOCHS = 5
IMG = (160, 160)

# %%
path = keras.utils.get_file(
    "flower_photos.tgz",
    "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz",
    extract=True,
)
data_dir = next(pathlib.Path(path).parent.rglob("daisy")).parent
print("Data:", data_dir)

train_ds = keras.utils.image_dataset_from_directory(
    data_dir, validation_split=0.2, subset="training", seed=42, image_size=IMG, batch_size=32)
val_ds = keras.utils.image_dataset_from_directory(
    data_dir, validation_split=0.2, subset="validation", seed=42, image_size=IMG, batch_size=32)
class_names = train_ds.class_names
print(class_names)

plt.figure(figsize=(10, 4))
for images, labels in train_ds.take(1):
    for i in range(8):
        plt.subplot(2, 4, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
plt.show()

# %%
base = keras.applications.MobileNetV2(input_shape=IMG + (3,), include_top=False, weights="imagenet")
base.trainable = False

inputs = keras.Input(IMG + (3,))
x = layers.RandomFlip("horizontal")(inputs)
x = layers.RandomRotation(0.1)(x)
x = keras.applications.mobilenet_v2.preprocess_input(x)
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(len(class_names), activation="softmax")(x)
model = keras.Model(inputs, outputs)

model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
h1 = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=2)

# %%
# Fine-tuning: unfreeze the top layers of the base model and train with a small learning rate.
base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False

model.compile(optimizer=keras.optimizers.Adam(1e-5), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
h2 = model.fit(train_ds, validation_data=val_ds, epochs=FINE_EPOCHS, verbose=2)

# %%
loss, acc = model.evaluate(val_ds, verbose=0)
print(f"Validation accuracy: {acc:.4f}   Validation loss: {loss:.4f}")

hist = {k: h1.history[k] + h2.history[k] for k in h1.history}
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for i, metric in enumerate(["loss", "accuracy"]):
    ax[i].plot(hist[metric], label="train")
    ax[i].plot(hist["val_" + metric], label="validation")
    ax[i].axvline(EPOCHS - 1, color="gray", linestyle="--", label="fine-tune start")
    ax[i].set_title(metric)
    ax[i].set_xlabel("epoch")
    ax[i].legend()
plt.show()

# %%
y_true, y_pred = [], []
for images, labels in val_ds:
    y_true.extend(labels.numpy())
    y_pred.extend(model.predict(images, verbose=0).argmax(axis=1))
print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion matrix")
plt.show()

# %%
images, labels = next(iter(val_ds))
preds = model.predict(images, verbose=0).argmax(axis=1)
plt.figure(figsize=(10, 4))
for i in range(8):
    plt.subplot(2, 4, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(f"P:{class_names[preds[i]]}\nA:{class_names[labels[i]]}", fontsize=9)
    plt.axis("off")
plt.show()
