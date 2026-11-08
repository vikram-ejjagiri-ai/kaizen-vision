# Kaizen Vision: test results

Run: 2026-11-07 14:52, quick mode, 18.9 min on x86_64, 2 threads

## Steel surface: which defect? (6 classes)

| Method | Clean accuracy | Mean accuracy over all 44 degraded conditions | ms per image |
|---|---|---|---|
| Classical features + SVM | 99.2% | 43.9% | 27.0 |
| Pretrained CNN features + linear classifier | 100.0% | 75.8% | 6.4 |
| Fine-tuned CNN | 100.0% | 67.7% | 4.7 |
| Classical + SVM, trained with lighting/blur/noise variation | 100.0% | 83.2% |  |
| CNN features, trained with variation | 100.0% | 92.4% |  |
| Fine-tuned CNN, trained with variation | 100.0% | 85.8% |  |

## Cast impeller: OK or defective?

| Method | Clean accuracy | Mean accuracy over all 44 degraded conditions | ms per image |
|---|---|---|---|
| Classical features + SVM | 99.2% | 67.3% | 39.0 |
| Pretrained CNN features + linear classifier | 99.2% | 78.6% | 6.6 |
| Fine-tuned CNN | 98.3% | 85.7% | 4.7 |
| Classical + SVM, trained with lighting/blur/noise variation | 99.2% | 72.8% |  |
| CNN features, trained with variation | 99.2% | 85.7% |  |
| Fine-tuned CNN, trained with variation | 98.3% | 90.0% |  |
| Anomaly detection (good parts only) | 92.5% | 70.5% | 15.3 |

## Part markings: read rate

| Reader | Clean | Mean over degraded conditions | ms |
|---|---|---|---|
| qr: OpenCV | 87% | 71% | 5.3 |
| qr: ZBar | 100% | 71% | 2.2 |
| qr: ZXing-C++ | 87% | 72% | 1.0 |
| datamatrix: ZXing-C++ | 73% | 72% | 0.9 |
| text: RapidOCR | 100% | 76% | 115.3 |
