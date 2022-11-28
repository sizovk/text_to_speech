import argparse
import numpy as np
import os
import torch
from tqdm import tqdm

from dataloader.text import text_to_sequence
from model import FastSpeech2
import glow
import waveglow

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

validation_texts = [
    "A defibrillator is a device that gives a high energy electric shock to the heart of someone who is in cardiac arrest",
    "Massachusetts Institute of Technology may be best known for its math, science and engineering education",
    "Wasserstein distance or Kantorovich Rubinstein metric is a distance function defined between probability distributions on a given metric space"
]

alphas = [
    (1.0, 1.0, 1.0),
    (1.2, 1.0, 1.0),
    (1.0, 1.2, 1.0),
    (1.0, 1.0, 1.2),
    (0.8, 1.0, 1.0),
    (1.0, 0.8, 1.0),
    (1.0, 1.0, 0.8),
    (1.2, 1.2, 1.2),
    (0.8, 0.8, 0.8)
]


def generate_model_input(validation_text):
    seq = text_to_sequence(validation_text, ['english_cleaners'])
    seq = np.array(seq)
    seq = np.stack([seq])
    src_pos = np.array([i+1 for i in range(seq.shape[1])])
    src_pos = np.stack([src_pos])
    src_seq = torch.from_numpy(seq).long()
    src_pos = torch.from_numpy(src_pos).long()
    return src_seq, src_pos


def get_waveglow():
    wave_glow = torch.load("./waveglow/pretrained_model/waveglow_256channels.pt", map_location="cpu")['model']
    wave_glow = wave_glow.remove_weightnorm(wave_glow)
    wave_glow.eval()
    for m in wave_glow.modules():
        if 'Conv' in str(type(m)):
            setattr(m, 'padding_mode', 'zeros')

    return wave_glow


def main(checkpoint_path):
    os.makedirs("./audio_examples", exist_ok=True)
    model = FastSpeech2(dict())
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    for alpha in tqdm(alphas):
        for i_text, validation_text in enumerate(tqdm(validation_texts)):
            src_seq, src_pos = generate_model_input(validation_text)
            output = model(src_seq, src_pos, length_alpha=alpha[0], pitch_alpha=alpha[1], energy_alpha=alpha[2])
            wg_input = output.transpose(1, 2)
            wg_model = get_waveglow()
            waveglow.inference.inference(
                wg_input, wg_model,
                f"./audio_examples/text{i_text}_length{alpha[0]:.1f}_pitch{alpha[1]}_energy{alpha[2]}.wav"
            )


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('checkpoint', type=str, help='path to checkpoint')
    args = args.parse_args()
    main(args.checkpoint)
