# Kaizen Vision: test results

Run: 2026-11-07 16:23, full mode, 90.1 min on x86_64, 2 threads

## Steel surface: which defect? (6 classes)

| Method | Clean accuracy | Mean accuracy over all 44 degraded conditions | ms per image |
|---|---|---|---|
| Classical features + SVM | 99.7% | 49.4% | 26.2 |
| Pretrained CNN features + linear classifier | 99.4% | 79.6% | 7.2 |
| Fine-tuned CNN | 99.7% | 71.8% | 5.0 |
| Classical + SVM, trained with lighting/blur/noise variation | 99.4% | 87.2% |  |
| CNN features, trained with variation | 98.9% | 93.9% |  |
| Fine-tuned CNN, trained with variation | 100.0% | 94.4% |  |

## Cast impeller: OK or defective?

| Method | Clean accuracy | Mean accuracy over all 44 degraded conditions | ms per image |
|---|---|---|---|
| Classical features + SVM | 99.2% | 67.0% | 39.8 |
| Pretrained CNN features + linear classifier | 99.7% | 83.5% | 6.9 |
| Fine-tuned CNN | 100.0% | 89.9% | 6.0 |
| Classical + SVM, trained with lighting/blur/noise variation | 96.9% | 73.9% |  |
| CNN features, trained with variation | 99.7% | 89.4% |  |
| Fine-tuned CNN, trained with variation | 99.5% | 96.8% |  |
| Anomaly detection (good parts only) | 92.8% | 71.7% | 14.4 |

## Part markings: read rate

| Reader | Clean | Mean over degraded conditions | ms |
|---|---|---|---|
| qr: OpenCV | 92% | 75% | 5.4 |
| qr: ZBar | 100% | 78% | 2.2 |
| qr: ZXing-C++ | 97% | 82% | 0.9 |
| datamatrix: ZXing-C++ | 93% | 81% | 0.9 |
| text: RapidOCR | 100% | 83% | 117.6 |
