# SIIM-ISIC-Melanoma-Classification
### Melanoma Classification [Competition](https://www.kaggle.com/c/siim-isic-melanoma-classification)
#### DATA SETUP : Assumes that you have [kaggleAPI](https://github.com/Kaggle/kaggle-api) installed
```
mkdir ./JPEG Melanoma 256x256
mkdir ./JPEG Melanoma 2019 256x256
kaggle datasets download -d cdeotte/jpeg-melanoma-256x256
kaggle datasets download -d cdeotte/jpeg-isic2019-256x256
unzip -q jpeg-melanoma-jpeg-melanoma-256x256.zip -d JPEG Melanoma 256x256
unzip -q jpeg-isic2019-jpeg-isic2019-256x256.zip -d JPEG Melanoma 2019 256x256
```
## Usage 
```
# Train Model:
python train.py
```
You can run the [notebook](https://github.com/Med-Amine-saighi/SIIM-ISIC-Melanoma-Classification/blob/main/melanoma-pytorch-train-bceloss.ipynb) to showcase the training process.

### OverView of the Model Architecture :
![](https://github.com/Med-Amine-saighi/SIIM-ISIC-Melanoma-Classification/blob/main/figures/Model%20Architecure.PNG)


