# %% [markdown]
# # PS7: YOLO on the Flowers dataset
# Dataset: https://www.kaggle.com/datasets/imsparsh/flowers-dataset (train/<daisy, dandelion, rose, sunflower, tulip>).
# The dataset has class folders but no bounding boxes, so YOLOv8 is used in classification mode (yolov8n-cls).
#
# Install once: `pip install ultralytics`

# %%
import random
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
from ultralytics import YOLO

SRC = Path("../datasets/flowers-dataset/train")
OUT = Path("flowers_yolo")
EPOCHS = 10

# %%
# YOLO classification needs train/<class> and val/<class>, so 20% of each class becomes validation.
random.seed(42)
for cls_dir in sorted(p for p in SRC.iterdir() if p.is_dir()):
    files = sorted(cls_dir.glob("*"))
    random.shuffle(files)
    n_val = int(0.2 * len(files))
    for split, part in [("val", files[:n_val]), ("train", files[n_val:])]:
        dst = OUT / split / cls_dir.name
        dst.mkdir(parents=True, exist_ok=True)
        for f in part:
            shutil.copy(f, dst / f.name)
    print(f"{cls_dir.name:12s} train={len(files) - n_val}  val={n_val}")

# %%
model = YOLO("yolov8n-cls.pt")
model.train(data=str(OUT.resolve()), epochs=EPOCHS, imgsz=224, seed=42)
save_dir = Path(model.trainer.save_dir)

# %%
metrics = model.val(data=str(OUT.resolve()))
print(f"Top-1 accuracy: {metrics.top1:.4f}   Top-5 accuracy: {metrics.top5:.4f}")

for name in ["results.png", "confusion_matrix.png"]:
    plt.figure(figsize=(12, 7))
    plt.imshow(plt.imread(save_dir / name))
    plt.title(name)
    plt.axis("off")
    plt.show()

# %%
val_images = sorted((OUT / "val").rglob("*.*"))
samples = random.sample(val_images, 8)
plt.figure(figsize=(12, 6))
for i, img in enumerate(samples):
    r = model.predict(str(img), verbose=False)[0]
    plt.subplot(2, 4, i + 1)
    plt.imshow(r.orig_img[..., ::-1])
    plt.title(f"P:{r.names[r.probs.top1]} ({r.probs.top1conf:.2f})\nA:{img.parent.name}", fontsize=9)
    plt.axis("off")
plt.show()
