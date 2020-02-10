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
