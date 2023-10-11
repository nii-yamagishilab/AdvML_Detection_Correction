# Detecting and Correcting Adversarial Images

Implementation of the paper:  <a href="https://www.jstage.jst.go.jp/article/transinf/E105.D/1/E105.D_2021MUP0005/_article/-char/en">Effects of Image Processing Operations on Adversarial Noise and Their Use in Detecting and Correcting Adversarial Images</a> (IEICE Transactions on Information and Systems 2022, best paper award).

You can clone this repository into your favorite directory:

    $ git clone https://github.com/nii-yamagishilab/AdvML_Detection_Correction.git

## 1. Requirement
- pytorch 1.3
- torchvision
- scikit-learn
- numpy
- foolbox 1.8.0

## 2. Project organization
1. Folder with data creation scripts, used to prepare the required dataset for experiments:

        data_creation
1. Folder with statistical feature extraction scripts, used to extract counting features and differences features:

        stats_feature_extraction
1. Folder with scripts for detectors using traditional machine learning algorithm on statistical features:
   
        detection_stats_traditional_ml
1. Folder with scripts for detectors using CNN on statistical features:
   
        detection_stats_cnn
1. Folder for end-to-end CNN detectors:
   
        detection_cnn
1. Folder with script for adversarial correction algorithms:
   
        correction
   
## 3. Authors
- Huy H. Nguyen (https://researchmap.jp/nhhuy/?lang=english)
- Minoru Kuribayashi (https://researchmap.jp/read0102027)
- Junichi Yamagishi (https://researchmap.jp/read0205283/?lang=english)
- Isao Echizen (https://researchmap.jp/echizenisao/?lang=english)

## Acknowledgement
This research was supported by JSPS KAKENHI Grants JP16H06302, JP17H04687, JP18H04120, JP18H04112, JP18KT0051, and JP19K22846 and by JST CREST Grants JPMJCR18A6 and JPMJCR20D3, Japan.

## Reference
H. H. Nguyen, M. Kuribayashi, J. Yamagishi, and I. Echizen, “Effects of Image Processing Operations on Adversarial Noise and Their Use in Detecting and Correcting Adversarial Images,” IEICE Transactions on Information and Systems (2022), 65-77.
