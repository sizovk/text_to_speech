from dataclasses import dataclass
from dacite import from_dict
import torch
from torch import nn

from .variance_adaptor import VarianceAdaptor
from .transformer import Encoder, Decoder


@dataclass
class FastSpeechConfig2:
    n_bins: int = 256
    pitch_min: float = -1.1868396117310913
    pitch_max: float = 6.822635708394184
    energy_min: float = -0.5493442
    energy_max: float = 78.29372

    num_mels: int = 80
    
    vocab_size: int = 300
    max_seq_len: int = 3000

    encoder_dim: int = 256
    encoder_n_layer: int = 4
    encoder_head: int = 2
    encoder_conv1d_filter_size: int = 1024

    decoder_dim: int = 256
    decoder_n_layer: int = 4
    decoder_head: int = 2
    decoder_conv1d_filter_size: int = 1024

    fft_conv1d_kernel: list = (9, 1)
    fft_conv1d_padding: list = (4, 0)

    duration_predictor_filter_size: int = 256
    duration_predictor_kernel_size: int = 3
    dropout: float = 0.1
    
    PAD: int = 0
    UNK: int = 1
    BOS: int = 2
    EOS: int = 3

    PAD_WORD: str = '<blank>'
    UNK_WORD: str = '<unk>'
    BOS_WORD: str = '<s>'
    EOS_WORD: str = '</s>'


def get_mask_from_lengths(lengths, max_len=None):
    if max_len == None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len, 1, device=lengths.device)
    mask = (ids < lengths.unsqueeze(1)).bool()

    return mask


class FastSpeech2(nn.Module):
    """ FastSpeech2 """

    def __init__(self, model_config):
        super(FastSpeech2, self).__init__()

        self.model_config = from_dict(data_class=FastSpeechConfig2, data=model_config)

        self.encoder = Encoder(self.model_config)
        self.variance_adaptor = VarianceAdaptor(self.model_config)
        self.decoder = Decoder(self.model_config)

        self.mel_linear = nn.Linear(self.model_config.decoder_dim, self.model_config.num_mels)

    def mask_tensor(self, mel_output, position, mel_max_length):
        lengths = torch.max(position, -1)[0]
        mask = ~get_mask_from_lengths(lengths, max_len=mel_max_length)
        mask = mask.unsqueeze(-1).expand(-1, -1, mel_output.size(-1))
        return mel_output.masked_fill(mask, 0.)

    def forward(self, src_seq, src_pos, mel_pos=None, mel_max_length=None, length_target=None, energy_target=None, pitch_target=None, length_alpha=1.0, pitch_alpha=1.0, energy_alpha=1.0):
        x, non_pad_mask = self.encoder(src_seq, src_pos)

        if self.training:
            output, duration, energy, pitch = self.variance_adaptor(x, length_target=length_target,energy_target=energy_target,pitch_target=pitch_target,mel_max_length=mel_max_length,length_alpha=length_alpha,pitch_alpha=pitch_alpha,energy_alpha=energy_alpha)
            output = self.decoder(output, mel_pos)
            output = self.mask_tensor(output, mel_pos, mel_max_length)
            energy = self.mask_tensor(energy.unsqueeze(2), mel_pos, mel_max_length).squeeze(2)
            pitch = self.mask_tensor(pitch.unsqueeze(2), mel_pos, mel_max_length).squeeze(2)
            output = self.mel_linear(output)
            return output, duration, energy, pitch
        else:
            output, mel_pos, energy, pitch = self.variance_adaptor(x, length_alpha=length_alpha,pitch_alpha=pitch_alpha,energy_alpha=energy_alpha)
            output = self.decoder(output, mel_pos)
            output = self.mel_linear(output)
            return output

