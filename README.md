# 🚗 ParkVision — AI-Powered Smart Parking Management System

> **A Full-Stack Smart Parking Management System integrating YOLO11, EasyOCR, Computer Vision, A* Pathfinding and Parking Management.**

ParkVision is an intelligent parking management platform designed to automate vehicle identification, license plate recognition, parking slot allocation, route optimization, and parking monitoring.

The project combines:

**Computer Vision + Deep Learning + Algorithms + Backend Engineering + Full-Stack Development**

---

# 🧠 AI & Computer Vision System

ParkVision uses a two-stage Automatic License Plate Recognition pipeline:

```text
Input Image / Camera
        │
        ▼
   Image Capture
     OpenCV
        │
        ▼
 ┌───────────────┐
 │    YOLO11     │
 │ Plate Detector│
 └───────┬───────┘
         │
         ▼
 Bounding Box
         │
         ▼
 License Plate Crop
         │
         ▼
 Image Preprocessing
         │
         ▼
 ┌───────────────┐
 │    EasyOCR    │
 │ Text Detector │
 │       +       │
 │ Text Recognizer
 └───────┬───────┘
         │
         ▼
 Raw Plate Text
         │
         ▼
 Text Normalization
         │
         ▼
 Vietnamese License Plate
         │
         ▼
 Parking Management System
```

The two AI components solve different problems:

```text
YOLO11
→ Where is the license plate?

EasyOCR
→ What characters are written on the plate?
```

This separation makes the pipeline easier to optimize because object localization and character recognition can be evaluated independently.

---

# 🔍 License Plate Detection Algorithm

ParkVision uses **YOLO11n** as the primary object detector.

YOLO stands for:

> **You Only Look Once**

Unlike traditional object detection approaches that may process an image multiple times or generate a large number of candidate regions, YOLO performs object detection through a single neural-network forward pass.

For an input image:

```text
I ∈ R^(H × W × 3)
```

the network learns a function:

```text
f(I; θ)
```

where:

```text
I = input image
θ = neural network parameters
```

The output contains predictions such as:

```text
[x, y, w, h, confidence, class]
```

For ParkVision, the target class is primarily:

```text
License Plate
```

The detector therefore learns both:

1. **Whether a license plate exists**
2. **Where the license plate is located**

---

# 🏗️ YOLO11 Core Architecture

The detection architecture can conceptually be divided into three major components:

```text
              Input Image
                   │
                   ▼
        ┌────────────────────┐
        │      Backbone      │
        │                    │
        │ Conv               │
        │ C3k2               │
        │ SPPF               │
        │ C2PSA              │
        └──────────┬─────────┘
                   │
                   ▼
        ┌────────────────────┐
        │        Neck        │
        │                    │
        │ Upsample           │
        │ Concat             │
        │ C3k2               │
        │ Feature Fusion     │
        └──────────┬─────────┘
                   │
                   ▼
        ┌────────────────────┐
        │  Detection Head    │
        │                    │
        │ P3                 │
        │ P4                 │
        │ P5                 │
        │                    │
        │ Box Regression     │
        │ Classification     │
        └──────────┬─────────┘
                   │
                   ▼
           License Plate Box
```

---

# 1. Backbone

The **Backbone** is responsible for extracting visual information from an image.

An image contains millions of raw pixel values.

The backbone transforms these pixels into progressively more meaningful feature representations.

Conceptually:

```text
Raw Pixels
    ↓
Edges
    ↓
Corners
    ↓
Textures
    ↓
Shapes
    ↓
Object Components
    ↓
License Plate Features
```

Early neural-network layers usually learn low-level visual information such as:

- edges
- color transitions
- corners
- simple textures

Deeper layers learn more abstract representations such as:

- rectangular plate shapes
- character-like structures
- vehicle components
- relative spatial patterns

---

# 🔹 Convolution Layer

Convolution is one of the fundamental operations of the network.

A convolution kernel slides across an input feature map:

```text
Input Feature Map
       │
       ▼
  Convolution Kernel
       │
       ▼
 Output Feature Map
```

Mathematically, a simplified convolution operation can be represented as:

```text
Y = W * X + b
```

where:

```text
X = input feature map
W = learnable convolution weights
b = bias
Y = resulting feature map
```

The weights `W` are learned automatically during training.

Instead of manually programming rules such as:

```text
"License plates are rectangular"
```

the neural network learns useful filters directly from training images.

---

# 🔹 C3k2 Layer

One important building block in YOLO11 is **C3k2**.

Its primary purpose is to efficiently extract deeper features while reducing unnecessary computational cost.

Conceptually:

```text
Input
  │
  ├─────────────► Shortcut / Feature Path
  │
  ▼
Convolution
  │
  ▼
Feature Blocks
  │
  ▼
Feature Fusion
  │
  ▼
Output
```

C3k2 helps balance:

```text
Feature Extraction
        +
Computational Efficiency
        +
Gradient Flow
```

For license plate detection, this is important because the model needs to recognize small rectangular objects without making the network unnecessarily heavy.

A lightweight architecture is particularly useful for future deployment on:

- parking cameras
- edge devices
- low-power computers
- real-time parking systems

---

# 🔹 SPPF — Spatial Pyramid Pooling Fast

Another important YOLO component is:

## SPPF

**Spatial Pyramid Pooling Fast**

SPPF increases the receptive field of the network.

In simple terms, it allows the model to inspect visual information at multiple spatial scales.

```text
Feature Map
     │
     ├── Small Region
     │
     ├── Medium Region
     │
     └── Large Region
            │
            ▼
       Feature Fusion
```

This matters because a license plate can appear differently depending on:

- camera distance
- vehicle position
- perspective
- image resolution
- plate size

For example:

```text
Near Camera
→ Large license plate

Far Camera
→ Small license plate
```

SPPF helps the network preserve useful contextual information in both situations.

---

# 🔹 C2PSA — Attention Mechanism

YOLO11 also introduces attention-related processing through components such as **C2PSA**.

The purpose of attention is to allow the network to emphasize more relevant visual features.

Instead of treating every region equally:

```text
Image Features
      │
      ▼
Attention
      │
      ├── Important features → stronger response
      │
      └── Less useful features → lower response
```

For ParkVision, useful features may include:

- strong plate boundaries
- rectangular structures
- areas containing dense characters
- vehicle regions surrounding a plate

This can help the detector focus on semantically meaningful regions.

---

# 2. Neck — Multi-Scale Feature Fusion

The **Neck** combines feature maps produced at different depths of the network.

This is critical in object detection.

Different layers contain different types of information:

```text
Shallow Layers
→ High spatial detail

Deep Layers
→ Strong semantic information
```

The neck combines both.

Typical operations include:

```text
Upsample
+
Concatenate
+
C3k2
```

Conceptually:

```text
Deep Features
     │
     ▼
  Upsample
     │
     ├───────────────┐
     │               │
     ▼               ▼
Deep Features   Shallow Features
     │               │
     └──────┬────────┘
            ▼
          Concat
            │
            ▼
        C3k2 Block
            │
            ▼
     Combined Features
```

This helps ParkVision detect plates at different sizes.

---

# 📐 Multi-Scale Detection

YOLO detects objects from multiple feature-map levels.

Conceptually:

```text
P3
→ Small Objects

P4
→ Medium Objects

P5
→ Large Objects
```

For a parking camera, this is particularly useful.

Example:

```text
Vehicle near entrance
        ↓
Large plate
        ↓
P4 / P5 features

Vehicle far from camera
        ↓
Small plate
        ↓
P3 features
```

Without multi-scale detection, small license plates could be much harder to identify.

---

# 3. Detection Head

The Detection Head converts extracted features into actual object predictions.

For every candidate object, the detector estimates information including:

```text
Bounding Box
+
Class Score
+
Confidence
```

A bounding box can be represented as:

```text
B = (x, y, w, h)
```

where:

```text
x = horizontal position
y = vertical position
w = bounding-box width
h = bounding-box height
```

The final output is approximately:

```text
License Plate
Confidence: 0.xx

Bounding Box:
x1, y1
x2, y2
```

---

# 📦 Bounding Box Regression

The detector must learn to place the bounding box as closely as possible around the real license plate.

Suppose:

```text
Ground Truth Box
        │
        ▼
 ┌───────────────┐
 │ LICENSE PLATE │
 └───────────────┘
```

and YOLO predicts:

```text
Predicted Box
      │
      ▼
 ┌─────────────────┐
 │  LICENSE PLATE  │
 └─────────────────┘
```

The network measures how well the predicted region overlaps with the real annotation.

One fundamental measurement is:

## Intersection over Union — IoU

```text
IoU =
Area of Intersection
────────────────────
 Area of Union
```

or:

```text
IoU = |A ∩ B| / |A ∪ B|
```

where:

```text
A = Ground Truth Bounding Box
B = Predicted Bounding Box
```

An IoU closer to `1.0` indicates better localization.

---

# 🧮 Training Objective

During training, YOLO attempts to minimize several types of error.

Conceptually:

```text
Total Loss
    =
Bounding Box Loss
    +
Classification Loss
    +
Distribution-Based Localization Loss
```

The exact internal implementation is handled by the Ultralytics training framework.

---

# 📦 Bounding Box Loss

Bounding-box loss teaches the network:

> How should the predicted bounding box move or resize to better overlap the actual license plate?

Example:

```text
Prediction:
 ┌──────────────────┐
 │    PLATE         │
 └──────────────────┘

Ground Truth:
      ┌─────────────┐
      │    PLATE    │
      └─────────────┘
```

Training attempts to minimize this difference.

---

# 📊 Distribution Focal Loss

Modern YOLO detection heads can represent bounding-box localization using distributions rather than only predicting direct coordinates.

Distribution Focal Loss helps the network learn more precise object boundaries.

This is especially useful for:

```text
Small objects
+
Precise localization
```

which are both relevant to license plate detection.

---

# 🏷️ Classification Loss

Classification loss determines whether the detected object belongs to the target class.

In this project, the task is simpler than general datasets containing dozens of object classes.

Conceptually:

```text
Detected Object
       │
       ▼
License Plate?
       │
    ┌──┴──┐
   Yes    No
```

The network learns to distinguish license plates from:

- vehicle lights
- bumpers
- signs
- text on vehicles
- background objects

---

# 🔄 Model Training Process

ParkVision uses **Transfer Learning**.

Instead of initializing every parameter randomly:

```text
Random Model
    ↓
Learn everything from zero
```

the model begins with:

```text
yolo11n.pt
```

which already contains general visual representations.

Then the model is fine-tuned on the Vietnamese license plate dataset.

```text
Pretrained YOLO11
       │
       ▼
General Visual Knowledge
       │
       ▼
Vietnamese Plate Dataset
       │
       ▼
Fine-Tuning
       │
       ▼
Specialized ANPR Detector
```

---

# ⚙️ Training Configuration

| Parameter | Configuration |
|---|---:|
| Architecture | YOLO11n |
| Base Weights | `yolo11n.pt` |
| Epochs | `100` |
| Image Size | `640 × 640` |
| Batch Size | `16` |
| Optimizer | `AdamW` |
| Initial Learning Rate | `0.001` |
| Early Stopping | `20 epochs` |
| Data Augmentation | Enabled |
| Framework | Ultralytics / PyTorch |
| Training Device | CPU |

---

# 🖼️ Input Resolution

Training images are processed at:

```text
640 × 640
```

This provides a practical balance between:

```text
Detection Accuracy
        ↕
Inference Speed
        ↕
Memory Usage
```

Using larger images may preserve more detail on distant license plates but increases computational cost.

---

# 🔢 Batch Size

The project currently uses:

```text
batch = 16
```

Meaning that approximately 16 training samples are processed before an optimizer update is performed.

Conceptually:

```text
16 Images
   │
   ▼
Forward Pass
   │
   ▼
Loss
   │
   ▼
Backward Pass
   │
   ▼
Average Gradient
   │
   ▼
Update Model
```

---

# ⚙️ AdamW Optimizer

ParkVision uses:

```text
AdamW
```

AdamW combines adaptive learning-rate behavior with decoupled weight decay.

A simplified update can be represented as:

```text
θ(t+1)
=
θ(t)
-
Learning Update
-
Weight Decay
```

where:

```text
θ = model parameters
```

Compared with basic gradient descent, AdamW maintains information about previous gradients and their magnitudes.

Conceptually:

```text
Gradient
   │
   ├── First Moment
   │
   ├── Second Moment
   │
   ▼
Adaptive Parameter Update
```

This generally makes optimization more stable for deep neural networks.

---

# 🧠 Neural Network Weights

The model contains a large number of trainable parameters.

Every convolution layer contains learned tensors such as:

```text
Weights
+
Biases
```

A simplified neural computation is:

```text
y = Wx + b
```

Training attempts to find values of `W` and `b` that minimize prediction error.

---

# 🔄 Backpropagation

During each training iteration:

```text
Training Image
      │
      ▼
Forward Propagation
      │
      ▼
Prediction
      │
      ▼
Compare with Label
      │
      ▼
Calculate Loss
      │
      ▼
Backpropagation
      │
      ▼
Calculate Gradients
      │
      ▼
AdamW
      │
      ▼
Update Weights
```

This process repeats across many batches and epochs.

As training progresses:

```text
Random / General Features
           ↓
Vehicle Features
           ↓
License Plate Features
           ↓
Specialized Plate Detector
```

---

# 💾 Model Checkpoints

Three weight states are particularly important.

## `yolo11n.pt`

Pretrained YOLO11n model.

```text
General visual knowledge
```

Used as the starting point for transfer learning.

---

## `last.pt`

Represents the most recent training state.

Main purpose:

```text
Resume interrupted training
```

The training system can restore this checkpoint instead of starting from the beginning.

---

## `best.pt`

Represents the checkpoint selected from training based on validation performance.

This is the main model used for:

```text
Inference
+
Validation
+
Deployment
```

---

# 📊 Detection Performance

The trained detector achieved the following reported validation metrics:

| Metric | Result |
|---|---:|
| **Precision** | **99.64%** |
| **Recall** | **98.79%** |
| **mAP@0.5** | **99.48%** |
| **mAP@0.5:0.95** | **70.86%** |

These metrics specifically describe the **YOLO license plate detector**.

They do **not** mean that the complete ANPR system reads 99.48% of all license plates correctly.

YOLO detection accuracy and OCR recognition accuracy are different measurements.

---

# 🎯 Precision — 99.64%

Precision answers:

> When the model says that an object is a license plate, how often is that prediction correct?

Formula:

```text
Precision =
TP
───────
TP + FP
```

where:

```text
TP = True Positive
FP = False Positive
```

Result:

```text
Precision = 99.64%
```

This suggests that the detector produces very few false positive detections on the evaluated validation data.

Example:

```text
Model detects:
100 objects as license plates

Very high precision
        ↓
Most of those detections
are actually license plates
```

---

# 🔎 Recall — 98.79%

Recall answers:

> Of all real license plates in the dataset, how many were successfully detected?

Formula:

```text
Recall =
TP
───────
TP + FN
```

where:

```text
FN = False Negative
```

Result:

```text
Recall = 98.79%
```

A high recall indicates that the model misses relatively few visible plates within the validation set.

---

# 📈 mAP@0.5 — 99.48%

mAP stands for:

> **mean Average Precision**

For:

```text
mAP@0.5
```

a prediction is evaluated using an IoU threshold around:

```text
IoU ≥ 0.5
```

Reported result:

```text
99.48%
```

This suggests that the detector performs extremely well when a moderate bounding-box overlap criterion is used.

---

# 📐 mAP@0.5:0.95 — 70.86%

This is a significantly stricter evaluation metric.

Instead of evaluating at only:

```text
IoU = 0.50
```

performance is averaged across increasingly strict overlap thresholds:

```text
0.50
0.55
0.60
...
0.90
0.95
```

Reported result:

```text
70.86%
```

---

# ⚠️ Why mAP@0.5 Is 99.48% but mAP@0.5:0.95 Is 70.86%

This difference is important.

The model appears highly effective at answering:

```text
"Is there a license plate here?"
```

but under stricter criteria it is less consistently perfect at answering:

```text
"Can I draw the bounding box extremely close
to the exact plate boundary?"
```

For example:

```text
Ground Truth
        ┌───────────────┐
        │  51A-123.45   │
        └───────────────┘

Prediction
    ┌──────────────────────┐
    │     51A-123.45       │
    └──────────────────────┘
```

The detector clearly finds the license plate.

At:

```text
IoU = 0.50
```

the prediction may easily count as correct.

But at:

```text
IoU = 0.90
```

the bounding box must align much more precisely with the annotation.

Therefore:

```text
mAP50 = 99.48%
```

and:

```text
mAP50-95 = 70.86%
```

are not contradictory.

They measure different levels of localization strictness.

---

# ⚠️ Interpreting Very High Accuracy

The reported detection metrics are strong, but model quality should not be evaluated from one metric alone.

Important factors include:

```text
Dataset Size
Dataset Diversity
Train / Validation Separation
Camera Angles
Lighting Conditions
Plate Distance
Blur
Night Conditions
Occlusion
Plate Types
Real-World Testing
```

For example, a model can achieve:

```text
mAP50 > 99%
```

on a relatively consistent validation dataset but perform worse when deployed under:

```text
Rain
Night
Motion Blur
Extreme Camera Angles
Dirty Plates
Very Small Plates
Different Cameras
```

Therefore, the current values should be interpreted as:

> **Validation performance on the project's evaluation dataset**

rather than a guaranteed real-world accuracy of 99% under every operating condition.

---

# 🔤 OCR Stage

Once YOLO finds a plate:

```text
Bounding Box
     │
     ▼
Crop Image
     │
     ▼
EasyOCR
```

OCR then converts pixels into text.

For example:

```text
Image:
┌──────────────┐
│  51A 123.45  │
└──────────────┘

          ↓

OCR

          ↓

"51A12345"
```

---

# 🧠 Why YOLO + OCR?

Running OCR directly on the entire image is inefficient.

A complete parking frame may contain:

- vehicles
- signs
- advertisements
- road markings
- buildings
- random text

Therefore ParkVision first performs:

```text
YOLO
↓
Region of Interest
↓
OCR
```

instead of:

```text
Entire Image
↓
OCR Everything
```

This reduces unnecessary OCR search space and usually improves recognition reliability.

---

# 📊 Complete ANPR Accuracy

The complete ANPR system contains two major probabilities:

```text
Plate Detection
        ×
Character Recognition
```

Conceptually:

```text
P(complete success)
≈
P(correct detection)
×
P(correct OCR | correct detection)
```

Therefore:

```text
YOLO mAP ≠ Final ANPR Accuracy
```

A detector can successfully identify a plate while OCR may still read:

```text
5 ↔ S

0 ↔ O

1 ↔ I

8 ↔ B
```

incorrectly.

A future evaluation should therefore include:

### Detection metrics

- Precision
- Recall
- mAP@0.5
- mAP@0.5:0.95

### OCR metrics

- Character Accuracy
- Character Error Rate
- Full Plate Accuracy

### End-to-End metrics

- Correct Detection + Correct Plate Recognition
- End-to-End ANPR Accuracy

---

# 📏 Recommended OCR Metrics

## Character Accuracy

Example:

```text
Ground Truth:
51A12345

Prediction:
51A12845
```

Only one character is incorrect.

Character-level accuracy therefore measures OCR quality more precisely than simply considering the entire plate correct or incorrect.

---

# 🎯 Full Plate Accuracy

For a parking system, an important metric is:

```text
Exact Plate Match
```

Example:

```text
Ground Truth:
51A-123.45

Prediction:
51A-123.45

→ Correct
```

but:

```text
Ground Truth:
51A-123.45

Prediction:
51A-123.46

→ Incorrect
```

Even a single incorrect character can cause the system to associate a vehicle with the wrong record.

Therefore exact plate accuracy is particularly important for production use.

---

# 🚀 Model Strengths

Current architecture provides several advantages.

### Lightweight detector

YOLO11n is designed as a relatively compact member of the YOLO11 family.

This makes it suitable for future real-time inference.

### Transfer learning

The model does not learn all visual features from scratch.

### Multi-scale detection

P3/P4/P5 representations help detect different plate sizes.

### Attention-enhanced feature processing

C2PSA helps the network emphasize useful visual information.

### Specialized dataset

Fine-tuning teaches the model features specific to Vietnamese plates.

### Strong reported precision

```text
99.64%
```

suggests very low false-positive behavior on validation data.

### Strong reported recall

```text
98.79%
```

suggests relatively few missed plates within that validation set.

---

# ⚠️ Current Model Limitations

Potential failure conditions include:

```text
Night-time images
Motion blur
Overexposure
Underexposure
Extreme viewing angle
Partial plate occlusion
Very distant vehicles
Damaged plates
Dirty plates
Non-standard plates
Camera vibration
Heavy rain
```

OCR introduces additional failure modes:

```text
Similar characters
Small characters
Plate reflections
Low contrast
Blurred digits
Unusual fonts
```

---

# 🔬 Future AI Improvements

Planned improvements include:

- [ ] Independent test dataset
- [ ] End-to-end ANPR evaluation
- [ ] Character Error Rate measurement
- [ ] Exact plate recognition accuracy
- [ ] Night-time dataset
- [ ] Motion-blur augmentation
- [ ] Perspective augmentation
- [ ] More diverse camera angles
- [ ] Plate tracking across video frames
- [ ] OCR confidence filtering
- [ ] Multi-frame OCR voting
- [ ] ONNX export
- [ ] TensorRT optimization
- [ ] Edge deployment
- [ ] Real-time inference benchmark
- [ ] FPS measurement
- [ ] Latency measurement
- [ ] Model-size benchmark

---

# 🌐 Full System Architecture

```text
Camera
  │
  ▼
OpenCV
  │
  ▼
YOLO11
  │
  ├── Backbone
  │     ├── Conv
  │     ├── C3k2
  │     ├── SPPF
  │     └── C2PSA
  │
  ├── Neck
  │     ├── Upsampling
  │     ├── Feature Concatenation
  │     └── Multi-Scale Fusion
  │
  └── Detection Head
        ├── P3
        ├── P4
        └── P5
             │
             ▼
       Plate Bounding Box
             │
             ▼
          EasyOCR
             │
             ▼
      License Plate Text
             │
             ▼
       Backend / API
             │
      ┌──────┴──────┐
      ▼             ▼
Parking Manager   Database
      │
      ▼
A* Pathfinding
      │
      ▼
Web Dashboard
```

---

# 👨‍💻 Author

## Thang Nguyen

**2nd-year Information Technology Student**

Interested in:

- Artificial Intelligence
- Computer Vision
- Machine Learning
- Full-Stack Development
- Backend Engineering
- Algorithms
- Cloud Computing
- Software Architecture

This project was developed as part of my journey in learning how artificial intelligence, computer vision, algorithms, backend systems, and full-stack technologies can be integrated into a practical real-world application.

GitHub:

```text
https://github.com/xuanthang300507-ctrl
```

---

# 🎯 Project Vision

ParkVision is not intended to be only a YOLO demonstration.

The long-term goal is to build a complete:

> **AI-Powered Full-Stack Smart Parking Management Platform**

combining:

```text
Computer Vision
        +
Deep Learning
        +
Data Structures & Algorithms
        +
Backend Engineering
        +
Database Systems
        +
Frontend Development
        +
Cloud Infrastructure
```

into a single end-to-end software system.
