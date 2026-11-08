# Test report: comparing machine vision methods for quality inspection

Kaizen Vision · Vikram Ejjagiri · test run 2026-11-07 16:23 (90.1 min, x86_64, 2 threads)

## 1. Objective

Select a vision method for a new inspection station. Compared: classical image processing, pretrained and fine-tuned neural networks, and anomaly detection, for accuracy, robustness to changed imaging conditions, training data needed and computing time. Readers for QR codes, Data Matrix codes and plain text are compared as well.

## 2. Test setup

- **Steel surface (NEU database):** 6 defect types, 1,440 training and 360 test images, 200 x 200 px.
- **Castings (pump impellers):** OK / defective, 910 training and 390 test images, 256 x 256 px.
- **Markings:** 60 generated QR, Data Matrix and text markings each, on real steel texture.
- **Hardware:** laptop CPU without GPU; pretrained network YOLO11s-cls (ImageNet).

## 3. Procedure

1. Train every method on clean images.
2. Test on clean images and under 11 changed conditions at 4 levels each (44 conditions): under- and overexposure, low contrast, uneven light, defocus, motion blur, sensor noise, lower resolution, rotation, camera tilt, JPEG compression.
3. Train with 2 % to 100 % of the training images (three random subsets each).
4. Measure computing time per image and training time.
5. Counter-measure: retrain with random lighting, blur and noise variation.
6. Integration: handshake with a simulated PLC over OPC UA (normal operation, camera faults, station failure) and camera calibration for converting pixels to millimetres.

## 4. Results

### Steel surface (6 defect types)

| Method | clean | mean under changed conditions | worst case |
|---|---|---|---|
| classical features + SVM | 99.7 % | 49.4 % | 16.7 % (defocus) |
| pretrained CNN features + linear classifier | 99.4 % | 79.6 % | 40.8 % (motion blur) |
| fine-tuned CNN | 99.7 % | 71.8 % | 31.7 % (defocus) |
| classical + SVM trained with variation | 99.4 % | 87.2 % | 20.8 % (sensor noise) |
| CNN features trained with variation | 98.9 % | 93.9 % | 71.7 % (defocus) |
| fine-tuned CNN trained with variation | 100.0 % | 94.4 % | 59.2 % (JPEG compression) |
Chance level: 16.7 % (6 classes).

**Tolerance** (highest level 0-4 within 5 points of the clean accuracy)

| | underexposure | overexposure | low contrast | uneven light | defocus | motion blur | sensor noise | lower resolution | rotation | camera tilt | JPEG compression |
|---|---|---|---|---|---|---|---|---|---|---|---|
| classical features + SVM | 2 | 0 | 1 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| pretrained CNN features + linear classifier | 3 | 1 | 3 | 4 | 0 | 0 | 1 | 0 | 3 | 4 | 0 |
| fine-tuned CNN | 1 | 1 | 1 | 4 | 0 | 0 | 0 | 0 | 3 | 4 | 0 |
| classical + SVM trained with variation | 3 | 1 | 2 | 3 | 0 | 0 | 1 | 0 | 3 | 4 | 0 |
| CNN features trained with variation | 4 | 1 | 3 | 4 | 2 | 1 | 1 | 4 | 3 | 4 | 2 |
| fine-tuned CNN trained with variation | 3 | 2 | 3 | 4 | 2 | 2 | 2 | 4 | 3 | 4 | 2 |

**Training data** (accuracy on the fixed test set)

| images per class | classical features + SVM | pretrained CNN features + linear classifier | fine-tuned CNN |
|---|---|---|---|
| 5 | 86.5 % | 85.2 % | 68.1 % |
| 12 | 95.2 % | 97.2 % | 72.5 % |
| 24 | 99.4 % | 98.7 % | 90.6 % |
| 60 | 99.4 % | 98.9 % | 99.2 % |
| 120 | 99.5 % | 99.3 % | 100.0 % |
| 240 | 99.7 % | 99.4 % | 99.7 % |

| Method | ms per image (CPU) | training time (s) |
|---|---|---|
| classical features + SVM | 26.2 | 38.8 |
| pretrained CNN features + linear classifier | 7.2 | 16.5 |
| fine-tuned CNN | 5.0 | 384.1 |

### Cast impeller (OK / defective)

| Method | clean | mean under changed conditions | worst case |
|---|---|---|---|
| classical features + SVM | 99.2 % | 67.0 % | 54.4 % (overexposure) |
| pretrained CNN features + linear classifier | 99.7 % | 83.5 % | 50.8 % (defocus) |
| fine-tuned CNN | 100.0 % | 89.9 % | 45.4 % (low contrast) |
| anomaly detection | 92.8 % | 71.7 % | 60.0 % (underexposure) |
| classical + SVM trained with variation | 96.9 % | 73.9 % | 37.2 % (sensor noise) |
| CNN features trained with variation | 99.7 % | 89.4 % | 60.8 % (JPEG compression) |
| fine-tuned CNN trained with variation | 99.5 % | 96.8 % | 76.9 % (JPEG compression) |
Calling every part defective would score about 60 % here (share of defective parts in the test set).

**Tolerance** (highest level 0-4 within 5 points of the clean accuracy)

| | underexposure | overexposure | low contrast | uneven light | defocus | motion blur | sensor noise | lower resolution | rotation | camera tilt | JPEG compression |
|---|---|---|---|---|---|---|---|---|---|---|---|
| classical features + SVM | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| pretrained CNN features + linear classifier | 1 | 1 | 1 | 2 | 1 | 1 | 0 | 1 | 2 | 4 | 0 |
| fine-tuned CNN | 1 | 2 | 1 | 4 | 1 | 1 | 2 | 2 | 4 | 4 | 1 |
| anomaly detection | 2 | 0 | 1 | 3 | 1 | 0 | 0 | 1 | 1 | 2 | 0 |
| classical + SVM trained with variation | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| CNN features trained with variation | 3 | 2 | 2 | 4 | 1 | 1 | 0 | 1 | 2 | 4 | 0 |
| fine-tuned CNN trained with variation | 3 | 2 | 4 | 4 | 3 | 3 | 3 | 3 | 4 | 4 | 3 |

| Method (clean) | AUROC | missed defects | false rejects |
|---|---|---|---|
| classical features + SVM | 1.000 | 0.9 % | 0.6 % |
| pretrained CNN features + linear classifier | 1.000 | 0.4 % | 0.0 % |
| fine-tuned CNN | 1.000 | 0.0 % | 0.0 % |
| anomaly detection | 0.981 | 7.3 % | 7.1 % |
| classical + SVM trained with variation | 0.996 | 2.6 % | 3.8 % |
| CNN features trained with variation | 1.000 | 0.4 % | 0.0 % |
| fine-tuned CNN trained with variation | 1.000 | 0.9 % | 0.0 % |

**Training data** (accuracy on the fixed test set)

| images per class | classical features + SVM | pretrained CNN features + linear classifier | fine-tuned CNN |
|---|---|---|---|
| 7 | 71.5 % | 73.6 % | 71.0 % |
| 18 | 76.3 % | 80.3 % | 91.8 % |
| 36 | 83.2 % | 88.0 % | 93.1 % |
| 91 | 92.3 % | 96.8 % | 96.9 % |
| 182 | 97.4 % | 98.0 % | 99.5 % |
| 363 | 99.2 % | 99.7 % | 100.0 % |

| Method | ms per image (CPU) | training time (s) |
|---|---|---|
| classical features + SVM | 39.8 | 36.7 |
| pretrained CNN features + linear classifier | 6.9 | 9.1 |
| fine-tuned CNN | 6.0 | 248.5 |
| anomaly detection | 14.4 | 93.4 |

### Part markings: read rate

| Reader | clean | mean changed | ms |
|---|---|---|---|
| qr: OpenCV | 91.7 % | 75.4 % | 5.4 |
| qr: ZBar | 100.0 % | 77.6 % | 2.2 |
| qr: ZXing-C++ | 96.7 % | 82.4 % | 0.9 |
| datamatrix: ZXing-C++ | 93.3 % | 80.9 % | 0.9 |
| text: RapidOCR | 100.0 % | 83.1 % | 117.6 |

### Station integration (OPC UA)

Simulated PLC (OPC UA server) and vision station (OPC UA client) with the handshake of a real inspection cell: Trigger → Busy → result, pick position, angle, error code → Done → Trigger = 0. PLC scan time 2 ms.

| Scenario | parts | response ms (mean / p95 / max) | PLC reaction |
|---|---|---|---|
| normal operation | 200 | 35.8 / 42.6 / 48.7 | none needed |
| camera misses every 10th image | 50 | 33.5 / 45.3 / 46.3 | 5 parts rejected with an error code (fail-safe) |
| station crashes during a cycle | 11 | 125.6 / 523.2 / 1000.5 | no answer within timeout: part rejected, line stopped |
| station crashes between parts | 5 | 35.7 / 40.0 / 40.2 | station heartbeat lost: line stopped |

Normal operation: inspection 29.9 ms, OPC UA writes 2.5 ms; 1 missed defect and 0 false rejects in 200 parts.

### Camera calibration

Source: self-check with rendered views of a known camera. Reprojection error 0.27 px (15 views). Position error on the work plane: 0.153 mm mean (p95 0.493 mm) with lens correction, 0.346 mm without.

## 5. Evaluation and recommendation

**1. Clean images do not decide.** On clean test images every method is between 92.8 % and 100.0 % accurate. A demonstration under lab conditions would not show which one to choose.

**2a. Steel surface (6 defect types):** when conditions change, the pretrained CNN features + linear classifier keeps 79.6 % on average over all 44 changed conditions, the classical features + SVM only 49.4 %. The hardest conditions are defocus, lower resolution, motion blur.

**2b. Cast impeller (OK / defective):** when conditions change, the fine-tuned CNN keeps 89.9 % on average over all 44 changed conditions, the classical features + SVM only 67.0 %. The hardest conditions are JPEG compression, sensor noise, defocus.

**3. Train with the variation you expect.** Retraining with random lighting, blur and noise raises the best mean accuracy under changed conditions from 79.6 % to 94.4 % (steel) and from 89.9 % to 96.8 % (casting); the largest gain is +38 points for the classical features + SVM on steel. Best on both tasks: fine-tuned CNN trained with variation (94.4 % steel, 96.8 % casting).

**4. Few training images: the best method depends on the task.** With only 12 images per class (steel), the pretrained features reach 97.2 %, the classical method 95.2 % and the fine-tuned network 72.5 %. On the castings it is the other way round: with 18 images per class the fine-tuned network reaches 91.8 %, the pretrained features 80.3 %, the classical method 76.3 %. Which method needs the fewest images depends on the task, so it has to be tested.

**5. No defect images? Anomaly detection.** Trained on good parts only, it separates OK and defective castings with AUROC 0.976 (303 good images); with 50 good images 0.903, with 5 only 0.669. With its automatic threshold it misses 7.3 % of the defects and rejects 7.1 % of the good parts, clearly behind the supervised methods. It is the right start for a new product while defect images are still rare, but it reacts to every change of the image, so lighting and camera must be kept stable.

**6. Speed is not the bottleneck.** Every method needs at most 40 ms per image on a laptop CPU, far below the cycle time of a typical line.

**7. Part markings:** on clean images the readers decode QR (OpenCV) 92%, QR (ZBar) 100%, QR (ZXing-C++) 97%, Data Matrix (ZXing-C++) 93%, text (RapidOCR) 100%. Hardest for every reader: sensor noise, motion blur, defocus.

**Recommendation for a new station:** fix lighting, focus and camera mounting first (changed conditions cost far more accuracy in the tests than the choice between methods on clean images), use the fine-tuned CNN trained with the expected variation, start with anomaly detection while defect images are rare, and confirm with trial images on the real station.


## 6. Limitations

- The changed conditions are simulated; they do not replace trials on the real station.
- Public datasets with controlled imaging; real lines add dust, oil and part variants.
- The markings are generated (known content); the surface is real.
- One run with a fixed random seed; differences below about 1 point are not meaningful.
- PLC and robot are simulated (OPC UA on the same computer); a real network adds latency.
