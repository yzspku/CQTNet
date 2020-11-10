# CQTNet
LEARNING A REPRESENTATION FOR COVER SONG IDENTIFICATION USING CONVOLUTIONAL NEURAL NETWORK. ICASSP2020 

## Environment
python  --  3
pytorch --  1.0
librosa --  0.63

## Dataset
Second Hand Songs 100K (SHS100K), which is collected from Second Hand Songs website. 

## Generate CQT
You can utilize "gencqt.py" to get CQT features from your own audio.

## Train 
python main.py multi_train --model='CQTNet' --batch_size=32 --load_latest=False --notes='experiment0'

## Test

python main.py test --model='CQTNet' --load_model_path = 'check_points/CQTNet.pth'


## Paramters 
https://drive.google.com/file/d/1Rv-NuiAKW2rUlNZj8SOs2Iqidqkx30M8/view?usp=sharing



## Spectrum Augmentation
After using Spectrum Augmentation in training stage, the model performance has a great improvement. 

Specaugment: A simple data augmentation method for automatic speech recognition.


|  Dataset   | MAP  |
|  ----  | ----  |
| YouTube350  | 0.933 |
| Covers80  | 0.860 |
| Mazurkas  | 0.933 |
| SHS100K-TEST  | 0.71 |
