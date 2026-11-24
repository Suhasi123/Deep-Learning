# %% [markdown]
# # PS9: YOLO on the Fruits and Vegetables dataset
# Dataset: https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition
# Layout: train/, validation/, test/ each with 36 class folders. There are no bounding boxes,
# so YOLOv8 is used in classification mode (yolov8n-cls).
#
# Install once: `pip install ultralytics`

# %%
import random
from pathlib import Path

import matplotlib.pyplot as plt
from ultralytics import YOLO

DATA = Path("../datasets/fruit-and-vegetable-image-recognition")
EPOCHS = 10

# %%
# YOLO looks for a folder named "val".
if (DATA / "validation").exists() and not (DATA / "val").exists():
    (DATA / "validation").rename(DATA / "val")

for split in ["train", "val", "test"]:
    classes = sorted(p.name for p in (DATA / split).iterdir() if p.is_dir())
    n = sum(1 for _ in (DATA / split).rglob("*.*"))
    print(f"{split:5s} classes={len(classes)} images={n}")
print(classes)

# %%
model = YOLO("yolov8n-cls.pt")
model.train(data=str(DATA.resolve()), epochs=EPOCHS, imgsz=224, seed=42)
save_dir = Path(model.trainer.save_dir)

# %%
metrics = model.val(data=str(DATA.resolve()), split="test")
print(f"Test top-1 accuracy: {metrics.top1:.4f}   Test top-5 accuracy: {metrics.top5:.4f}")

for name in ["results.png", "confusion_matrix.png"]:
    plt.figure(figsize=(14, 9))
    plt.imshow(plt.imread(save_dir / name))
    plt.title(name)
    plt.axis("off")
    plt.show()

# %%
test_images = sorted((DATA / "test").rglob("*.*"))
plt.figure(figsize=(12, 6))
for i, img in enumerate(random.sample(test_images, 8)):
    r = model.predict(str(img), verbose=False)[0]
    plt.subplot(2, 4, i + 1)
    plt.imshow(r.orig_img[..., ::-1])
    plt.title(f"P:{r.names[r.probs.top1]} ({r.probs.top1conf:.2f})\nA:{img.parent.name}", fontsize=9)
    plt.axis("off")
plt.show()
