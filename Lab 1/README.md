# Deep Learning Applications — Laboratory 1
## Traffic Sign Classification and Detection on GTSRB

This lab is a study on how to exploit and adapt pre-trained convolutional networks to solve new problems, using the [German Traffic Sign Recognition Benchmark (GTSRB)](https://benchmark.ini.rub.de/) as the target domain. The work covers three main areas: a thorough exploratory data analysis paired with a feature extraction baseline, a full end-to-end fine-tuning pipeline (exercises 1.3 and 2 were developed together as a single consolidated effort), and a traffic sign detector built on top of the classification backbone using Faster R-CNN.

---

## Exercise 1 — Exploratory Data Analysis and Baseline

### Dataset Overview

The GTSRB training set contains 26,640 images across 43 classes, representing real-world traffic sign photos captured under varying conditions. Before doing anything else, the dataset was studied carefully to understand its structure and identify potential problems that would need to be addressed during training.

![Class distribution bar chart](doc/class_distribution.png)
*Class distribution across the 43 GTSRB categories, sorted by frequency.*

**Key findings from the EDA:**

- **Variable image sizes.** Image height and width both have a mean of ~50px with a standard deviation of ~23px, and some images go as small as 25px on a side. A fixed resize to 64×64 is necessary before any model can process them in batches. Aspect ratio is very close to 1.0 across all 43 classes, so a square resize introduces no meaningful distortion.

- **Brightness heterogeneity.** Mean pixel brightness (computed on greyscale) ranges from roughly 0.1 to 0.6 depending on the class, with high within-class variance too. This is the dominant visual challenge in the dataset and directly motivated the use of `ColorJitter` in augmentation.

- **Contrast is more uniform.** RMS contrast (pixel std in greyscale) sits around 0.53 on average and is fairly consistent across classes, so it is less of a concern.

- **Class imbalance.** The distribution is noticeably skewed: the most frequent classes reach ~1500 samples while the rarest have only ~150. This is not extreme but is enough to bias a naively trained model toward the majority classes. It is addressed with a `WeightedRandomSampler` at the DataLoader level.

- **No missing values.** Checked explicitly across all features grouped by class — the dataset is clean.

![Brightness and contrast boxplots per class](doc/brightness_contrast.png)
*Brightness (left) and contrast (right) distributions per class, shown as boxplots.*

### Data Augmentation

The augmentation pipeline was designed around the EDA findings. `ColorJitter` targets the brightness/contrast variability observed in the data. Rotations are kept small (±15°) to avoid misrepresenting directional signs. `RandomAffine` with shear adds a mild perspective effect. ImageNet normalization stats are used since the backbone was pretrained on ImageNet.

```python
T.Compose([
    T.Resize((64, 64), antialias=True),
    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
    T.RandomRotation(degrees=15),
    T.RandomAffine(degrees=0, shear=10),
    T.ToDtype(torch.float32, scale=True),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

The test set uses only resize + normalize, no augmentation.

### Baseline — ResNet50 as Feature Extractor + LinearSVC

To establish a reference point before any fine-tuning, ResNet50 pretrained on ImageNet was used as a frozen feature extractor. The final fully connected layer was replaced with `nn.Identity()`, so the network outputs a 2048-dimensional vector for each image instead of a class prediction. These features were extracted for the entire training set (26,640 × 2048) and test set (12,630 × 2048) in a single pass with `torch.no_grad()`, then used to train a `LinearSVC` from scikit-learn (`dual=False`, `max_iter=10000`).

The idea is simple but powerful: ImageNet pretraining already gives the backbone strong general visual representations, and a linear classifier on top of those features can go a long way without touching the network weights at all. This serves as the lower bound to beat with fine-tuning.

> ⚠️ The final SVM classification report output was not captured in the notebook. The pipeline ran correctly (features extracted, SVM fitted) but the cell output is missing.

---

## Exercises 1.3 + 2 — Fine-tuning Pipeline

Exercises 1.3 (fine-tuning baseline) and 2 (pipeline consolidation) were developed together as a single effort. Rather than doing a quick one-off fine-tuning in 1.3 and then refactoring it in 2, the reproducible pipeline was built directly from the start. The result is a `GTSRB_Trainer` class that handles everything: model instantiation from a config, optimizer selection, training loop with per-epoch evaluation, early stopping, WandB logging, and checkpoint saving.

Configuration is managed via `OmegaConf` (YAML-based), making it easy to run and compare experiments by changing a single file. Metrics and model artifacts are tracked with Weights & Biases, allowing full reproducibility of any run.

### Model

ResNet50 pretrained on ImageNet, with the final `fc` layer replaced by a small MLP head:

```python
model.fc = nn.Sequential(
    nn.Linear(in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, num_classes)   # 43 classes
)
```

All parameters are updated during training (full fine-tuning, not linear probing). The pretrained backbone provides a strong initialization that drastically reduces the number of epochs needed to converge.

### Training Configuration

| Hyperparameter | Value |
|:---|:---:|
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Batch size | 128 |
| Epochs | 5 |
| Early stopping patience | 3 |
| Input resolution | 64 × 64 |
| Loss | CrossEntropyLoss |
| Class balancing | WeightedRandomSampler |

### Results

| Epoch | Train Loss | Test Accuracy |
|:---:|:---:|:---:|
| 1 | 0.367 | 97.66% ✓ |
| 2 | 0.095 | 95.78% |
| 3 | 0.096 | 95.25% |
| 4 | 0.063 | **97.91%** ← best checkpoint |
| 5 | 0.064 | 96.74% |

**Final test accuracy: 96.74%** (best checkpoint saved at epoch 4: **97.91%**).

The model converges remarkably fast — epoch 1 already hits 97.66%, which is a direct consequence of the strong ImageNet initialization. After that, accuracy oscillates slightly while loss continues to drop, a sign that the learning rate is slightly too high for the later stages of fine-tuning. A cosine annealing or step decay scheduler would likely stabilize the final epochs and squeeze out another point or two.

Early stopping with `patience=3` correctly identified epoch 4 as the best checkpoint and would have stopped training at epoch 7 if more epochs had been run. The `WeightedRandomSampler` played a measurable role in keeping the model honest on underrepresented classes by oversampling them during training.

![Training curve](doc/training_curve.png)
*Training loss and test accuracy across 5 epochs. Best checkpoint marked at epoch 4.*

---

## Exercise 3.3 — Traffic Sign Detection with Faster R-CNN

The final exercise extends the classification work into full object detection: given a real-world road scene, locate and classify all traffic signs in the image. A new dataset was used for this: [keremberke/german-traffic-sign-detection](https://huggingface.co/datasets/keremberke/german-traffic-sign-detection) on HuggingFace, which contains full-frame images (1360×800px, consistent size) with bounding box annotations. The split is 386 train / 111 validation / 57 test images.

### Architecture and Transfer

The approach is to start from `fasterrcnn_resnet50_fpn` pretrained on COCO, and replace its backbone with the ResNet50 that was already fine-tuned on GTSRB in the previous exercise. The intuition is that the backbone already "knows" what traffic signs look like at a feature level, so the Region Proposal Network should be able to learn to propose good candidate boxes faster and more reliably than starting from a generic COCO backbone.

The transfer is done by extracting all state dict keys that do not belong to the `fc` layer (i.e., everything up to and including `layer4`) and loading them into `frcnn_model.backbone.body` with `strict=False`. Zero missing or unexpected keys were reported, meaning the architectures aligned perfectly.

The box predictor head is replaced with a new `FastRCNNPredictor` adapted for 43 classes:

```python
num_classes = 43  # 42 real classes + background (class 0)
in_features = frcnn_model.roi_heads.box_predictor.cls_score.in_features
frcnn_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
```

Training ran for 3 epochs with batch size 4 (required by Faster R-CNN's variable-size input handling via `collate_fn`).

### Class ID Mismatch

An important subtlety emerged during evaluation. The detection dataset uses class IDs **1–42**, while the GTSRB classification dataset uses **0–42**. Since Faster R-CNN internally reserves class 0 for background, this means the detection labels are already correctly 1-indexed for the FRCNN convention. However, the fine-tuned backbone was trained to associate feature patterns with GTSRB class IDs 0–42, so there is a systematic one-off shift between what the backbone "expects" and what the detection dataset labels say. This is a primary contributor to the poor classification accuracy observed at evaluation.

### Results

Evaluation was performed on the 57-image test set at IoU ≥ 0.5:

| Metric | Score |
|:---|:---:|
| Localization only (IoU ≥ 0.5) | **87.36%** |
| Localization + correct class | **11.49%** |

The localization result is encouraging — the backbone transfer works and the RPN learns to propose reasonable bounding boxes around signs in just 3 epochs. The classification collapse, however, is explained by two compounding issues: the class ID offset described above, and an extremely sparse training set for detection (some classes appear only once in the 386 training images), which makes it nearly impossible to learn a reliable 43-class classifier from scratch in 3 epochs regardless of backbone quality.

![Detection example](doc/detection_example.png)
*Example detections on a test image. Green: ground truth boxes. Red: predicted boxes with score > 0.5.*

---

## Notes

- `torchvision.transforms.v2` was used throughout for better efficiency and compatibility with bounding box transforms.
- WandB and OmegaConf integrate cleanly: configs are logged automatically at run start and model checkpoints are saved as artifacts.
- `WeightedRandomSampler` requires `replacement=True` to work correctly with large imbalance ratios — sampling without replacement would exhaust minority classes before the epoch ends.
- Faster R-CNN performs its own internal normalization via `GeneralizedRCNNTransform`. Images passed to the model must be raw `[0, 1]` float tensors — applying ImageNet normalization manually in the dataset on top of this causes incorrect inputs and `nan` losses.
- The `nan` loss observed in an early detection training run (cells 84) was caused exactly by this double-normalization bug, fixed by removing the manual `Normalize` from `GTSRBDetectionDataset` and letting FRCNN handle it internally.
