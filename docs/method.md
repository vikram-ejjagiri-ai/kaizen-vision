### Question

A new inspection station needs a vision system. Which approach should it use: classical image processing, a
pretrained neural network, a network trained on the task, or anomaly detection that learns only from good parts?
And what must the camera and lighting deliver so that the chosen method keeps working?

### Test series

| | Steel surface (NEU) | Cast impeller (casting) | Part markings |
|---|---|---|---|
| Task | which of 6 defect types | OK or defective | read QR, Data Matrix, serial number |
| Images | 1,440 train / 360 test, 200 px | 910 train / 390 test, 256 px | 60 per type, generated on real steel texture |
| Methods | classical, CNN features, fine-tuned CNN | + anomaly detection | OpenCV, ZBar, ZXing-C++, RapidOCR |
| Measured | accuracy | accuracy, AUROC, missed defects, false rejects | read rate |

1. **Training on clean images** (as after commissioning).
2. **Test on clean images and under 11 changed conditions at 4 severity levels each** (44 conditions):
   under- and overexposure, low contrast, uneven light, defocus, motion blur, sensor noise, lower resolution,
   part rotation, camera tilt, JPEG compression.
3. **Training data:** the same methods trained on 2 % to 100 % of the training images (three random subsets each).
4. **Speed:** milliseconds per image on a laptop CPU, training time, model size.
5. **Counter-measure:** the same methods trained again with random lighting, blur and noise variation.

**Tolerance** of a method to a condition = highest severity level at which its accuracy stays within 5 percentage
points of its clean accuracy.

### Methods

- **Classical features + SVM:** about 80 hand-crafted numbers per image (grey-level statistics, local binary
  patterns, co-occurrence texture, edges, dark/bright spots) and a support vector machine. No neural network.
- **Pretrained CNN features + linear classifier:** an ImageNet-pretrained network (YOLO11s-cls) is used as a
  fixed feature extractor; only a logistic regression is trained.
- **Fine-tuned CNN:** the same network fully retrained on the task (12 epochs, flips and 90° turns as
  augmentation).
- **Anomaly detection (PatchCore method):** a memory bank of mid-level patch features from good parts only; the
  score is the largest distance of any patch to the nearest normal patch. The decision threshold comes from good
  parts that were not in the memory bank (97th percentile).
