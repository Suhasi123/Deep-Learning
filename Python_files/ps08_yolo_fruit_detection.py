# %% [markdown]
# # PS8: Object detection with YOLOv8 on fruit images
# Dataset: https://www.kaggle.com/datasets/afsananadia/fruits-images-dataset-object-detection
# Layout: "Train File/Train File" and "Test File/Test File", each with images and YOLO .txt box labels.
# Every label file uses class 0, so the real class is taken from the file name (e.g. 05_Banana.jpg -> banana).
#
# Install once: `pip install ultralytics`

# %%
import random
import shutil
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from ultralytics import YOLO

SRC = Path("../datasets/fruits-images-dataset")
OUT = Path("fruits_yolo")
EPOCHS = 30
ALIASES = {"hogpulm": "hogplum", "lichi": "litchi"}


# %%
def key(path):
    return path.stem.lower().replace(" ", "")


def fruit(path):
    name = key(path).split("_", 1)[1]
    return ALIASES.get(name, name)


pairs = {}
for split, folder in [("train", "Train File"), ("val", "Test File")]:
    files = list((SRC / folder).rglob("*"))
    labels = {key(f): f for f in files if f.suffix == ".txt"}
    pairs[split] = [(img, labels[key(img)]) for img in files
                    if img.suffix.lower() in (".jpg", ".jpeg", ".png") and key(img) in labels]
classes = sorted({fruit(img) for items in pairs.values() for img, _ in items})
print("Classes:", classes)
for split, items in pairs.items():
    print(split, len(items), dict(Counter(fruit(img) for img, _ in items)))

# %%
for split, items in pairs.items():
    for d in ("images", "labels"):
        (OUT / d / split).mkdir(parents=True, exist_ok=True)
    for i, (img, label) in enumerate(items):
        name = f"{i:04d}_{key(img)}"
        cls = classes.index(fruit(img))
        lines = [f"{cls} " + " ".join(l.split()[1:]) for l in label.read_text().splitlines() if l.strip()]
        shutil.copy(img, OUT / "images" / split / f"{name}{img.suffix.lower()}")
        (OUT / "labels" / split / f"{name}.txt").write_text("\n".join(lines))

yaml_path = OUT / "data.yaml"
yaml_path.write_text(
    f"path: {OUT.resolve().as_posix()}\ntrain: images/train\nval: images/val\nnames:\n"
    + "".join(f"  {i}: {c}\n" for i, c in enumerate(classes)))
print(yaml_path.read_text())

# %%
model = YOLO("yolov8n.pt")
model.train(data=str(yaml_path), epochs=EPOCHS, imgsz=640, seed=42)
save_dir = Path(model.trainer.save_dir)

# %%
metrics = model.val(data=str(yaml_path))
print(f"Precision: {metrics.box.mp:.4f}  Recall: {metrics.box.mr:.4f}")
print(f"mAP@0.5: {metrics.box.map50:.4f}  mAP@0.5:0.95: {metrics.box.map:.4f}")

for name in ["results.png", "confusion_matrix.png"]:
    plt.figure(figsize=(12, 7))
    plt.imshow(plt.imread(save_dir / name))
    plt.title(name)
    plt.axis("off")
    plt.show()

# %%
val_images = sorted((OUT / "images" / "val").glob("*"))
plt.figure(figsize=(12, 8))
for i, img in enumerate(random.sample(val_images, min(6, len(val_images)))):
    r = model.predict(str(img), conf=0.25, verbose=False)[0]
    plt.subplot(2, 3, i + 1)
    plt.imshow(r.plot()[..., ::-1])
    plt.title(img.name, fontsize=8)
    plt.axis("off")
plt.show()
