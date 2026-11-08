**1. Clean images do not decide.** On clean test images every method is between 92.8 % and 100.0 % accurate. A demonstration under lab conditions would not show which one to choose.

**2a. Steel surface (6 defect types):** when conditions change, the pretrained CNN features + linear classifier keeps 79.6 % on average over all 44 changed conditions, the classical features + SVM only 49.4 %. The hardest conditions are defocus, lower resolution, motion blur.

**2b. Cast impeller (OK / defective):** when conditions change, the fine-tuned CNN keeps 89.9 % on average over all 44 changed conditions, the classical features + SVM only 67.0 %. The hardest conditions are JPEG compression, sensor noise, defocus.

**3. Train with the variation you expect.** Retraining with random lighting, blur and noise raises the best mean accuracy under changed conditions from 79.6 % to 94.4 % (steel) and from 89.9 % to 96.8 % (casting); the largest gain is +38 points for the classical features + SVM on steel. Best on both tasks: fine-tuned CNN trained with variation (94.4 % steel, 96.8 % casting).

**4. Few training images: the best method depends on the task.** With only 12 images per class (steel), the pretrained features reach 97.2 %, the classical method 95.2 % and the fine-tuned network 72.5 %. On the castings it is the other way round: with 18 images per class the fine-tuned network reaches 91.8 %, the pretrained features 80.3 %, the classical method 76.3 %. Which method needs the fewest images depends on the task, so it has to be tested.

**5. No defect images? Anomaly detection.** Trained on good parts only, it separates OK and defective castings with AUROC 0.976 (303 good images); with 50 good images 0.903, with 5 only 0.669. With its automatic threshold it misses 7.3 % of the defects and rejects 7.1 % of the good parts, clearly behind the supervised methods. It is the right start for a new product while defect images are still rare, but it reacts to every change of the image, so lighting and camera must be kept stable.

**6. Speed is not the bottleneck.** Every method needs at most 40 ms per image on a laptop CPU, far below the cycle time of a typical line.

**7. Part markings:** on clean images the readers decode QR (OpenCV) 92%, QR (ZBar) 100%, QR (ZXing-C++) 97%, Data Matrix (ZXing-C++) 93%, text (RapidOCR) 100%. Hardest for every reader: sensor noise, motion blur, defocus.

**Recommendation for a new station:** fix lighting, focus and camera mounting first (changed conditions cost far more accuracy in the tests than the choice between methods on clean images), use the fine-tuned CNN trained with the expected variation, start with anomaly detection while defect images are rare, and confirm with trial images on the real station.
