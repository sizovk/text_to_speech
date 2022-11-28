# Text-to-Speech

This repository contains code for training of FastSpeech and FastSpeech2 models, which have been described in the articles https://arxiv.org/abs/1905.09263 and https://arxiv.org/abs/2006.04558 respectively.

## [FastSpeech2 training report](https://wandb.ai/k_sizov/fastspeech/reports/FastSpeech2-training-report--VmlldzozMDQ4NjE1)


## Reproduce results
### Setup data
```bash
pip install -r requirements.txt
bash data_setup.sh
python data_preprocess.py
```

### Train FastSpeech
```bash
python train.py -c configs/fastspeech.json 
```

### Train FastSpeech2
```bash
python train.py -c configs/fastspeech2.json 
```
