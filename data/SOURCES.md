# Data sources

The images are not stored in this repository. `python scripts/get_data.py` downloads them into `data/`.

## 1. NEU surface defect database (steel)

- Hot-rolled steel strip surface, 6 defect classes (crazing, inclusion, patches, pitted surface, rolled-in scale,
  scratches), 300 greyscale images of 200 x 200 px per class.
- Source: K. Song and Y. Yan, Northeastern University, China.
  https://faculty.neu.edu.cn/songkc/en/zdylm/263265/list/index.htm
- Citation requested by the authors: K. Song, Y. Yan, "A noise robust method based on completed local binary
  patterns for hot-rolled steel strip surface defects", Applied Surface Science 285 (2013) 858-864.
- No formal licence is stated; the database is offered for research use. Used here for a non-commercial
  portfolio project, with citation.
- The NEU-DET version is used (with its train/validation split of 240/60 images per class). The bounding-box
  annotations are not used; only the class.

## 2. Casting product image data for quality inspection

- Top-view images of cast submersible-pump impellers, taken under controlled lighting at Pilot Technocast,
  Rajkot, India. The 512 x 512 px version without augmentation is used: 519 OK and 781 defective parts.
- Source: R. Dabhi, Kaggle: "casting product image data for quality inspection"
  https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product
- Licence as listed on Kaggle: CC BY-NC-ND 4.0 (attribution, non-commercial, no derivatives); please check the
  Kaggle page. Therefore the images are not redistributed here and the project is non-commercial.
- Own stratified split: 70 % training, 30 % test (seed 0). The larger, augmented version of the dataset is not
  used, because it contains rotated copies of the same parts that could end up in both training and test.

## 3. Part markings (generated)

- QR codes, Data Matrix codes and printed serial numbers with known content, marked onto real steel-surface
  crops from dataset 1 (see `vision/codes.py`). Generated, so the correct answer is always known.

## Pretrained network

- Ultralytics YOLO11s classification model (ImageNet-pretrained), downloaded automatically from the Ultralytics
  GitHub release on first run. Ultralytics is licensed under AGPL-3.0.
