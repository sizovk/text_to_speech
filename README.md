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

### Inference FastSpeech2
```bash
python inference.py <PATH TO CHECKPOINT>
```

This will generate sound examples in the `audio_examples` folder with the following texts

* `A defibrillator is a device that gives a high energy electric shock to the heart of someone who is in cardiac arrest`
* `Massachusetts Institute of Technology may be best known for its math, science and engineering education`
* `Wasserstein distance or Kantorovich Rubinstein metric is a distance function defined between probability distributions on a given metric space`

And with the following configuration
* usual generated audio
* audio with +20%/-20% for pitch/speed/energy
* audio with +20/-20% for pitch,speed and energy together