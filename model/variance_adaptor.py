import torch
from torch import nn
import torch.nn.functional as F

from .length_regulator import Transpose, create_alignment


class VariancePredictor(nn.Module):
    """ Variance Predictor """

    def __init__(self, model_config):
        super(VariancePredictor, self).__init__()

        self.input_size = model_config.encoder_dim
        self.filter_size = model_config.duration_predictor_filter_size
        self.kernel = model_config.duration_predictor_kernel_size
        self.conv_output_size = model_config.duration_predictor_filter_size
        self.dropout = model_config.dropout

        self.conv_net = nn.Sequential(
            Transpose(-1, -2),
            nn.Conv1d(
                self.input_size, self.filter_size,
                kernel_size=self.kernel, padding=1
            ),
            Transpose(-1, -2),
            nn.LayerNorm(self.filter_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            Transpose(-1, -2),
            nn.Conv1d(
                self.filter_size, self.filter_size,
                kernel_size=self.kernel, padding=1
            ),
            Transpose(-1, -2),
            nn.LayerNorm(self.filter_size),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )

        self.linear_layer = nn.Linear(self.conv_output_size, 1)
        self.relu = nn.ReLU()

    def forward(self, encoder_output):
        encoder_output = self.conv_net(encoder_output)
            
        out = self.linear_layer(encoder_output)
        out = self.relu(out)
        out = out.squeeze()
        if not self.training:
            out = out.unsqueeze(0)
        return out


class LengthRegulator2(nn.Module):
    """ Length Regulator with log length prediction"""

    def __init__(self, model_config):
        super(LengthRegulator2, self).__init__()
        self.duration_predictor = VariancePredictor(model_config)

    def LR(self, x, duration_predictor_output, mel_max_length=None):
        expand_max_len = torch.max(
            torch.sum(duration_predictor_output, -1), -1)[0]
        alignment = torch.zeros(duration_predictor_output.size(0),
                                expand_max_len,
                                duration_predictor_output.size(1)).numpy()
        alignment = create_alignment(alignment,
                                     duration_predictor_output.cpu().numpy())
        alignment = torch.from_numpy(alignment).to(x.device)

        output = alignment @ x
        if mel_max_length:
            output = F.pad(
                output, (0, 0, 0, mel_max_length-output.size(1), 0, 0))
        return output

    def forward(self, x, alpha=1.0, target=None, mel_max_length=None):
        duration_predictor_output = torch.exp(self.duration_predictor(x)) - 1
        
        if target is not None:
            output = self.LR(x, target, mel_max_length)
            return output, duration_predictor_output
        else:
            duration_predictor_output = ((duration_predictor_output + 0.5) * alpha).int()
    
            output = self.LR(x, duration_predictor_output)
            
            mel_pos = torch.stack([torch.Tensor([i+1 for i in range(output.size(1))])]).long().to(x.device)
        return output, mel_pos


class VarianceAdaptor(nn.Module):
    def __init__(self, model_config):
        super(VarianceAdaptor, self).__init__()

        self.length_regulator = LengthRegulator2(model_config)

        self.energy_predictor = VariancePredictor(model_config)
        self.energy_bins = nn.Parameter(
            torch.linspace(model_config.energy_min, model_config.energy_max, model_config.n_bins - 1),
            requires_grad=False,
        )
        self.energy_embedding = nn.Embedding(
            model_config.n_bins, model_config.encoder_dim,
        )


        self.pitch_predictor = VariancePredictor(model_config)
        self.pitch_bins = nn.Parameter(
            torch.linspace(model_config.pitch_min, model_config.pitch_max, model_config.n_bins - 1),
            requires_grad=False,
        )
        self.pitch_embedding = nn.Embedding(
            model_config.n_bins, model_config.encoder_dim,
        )


    def forward(self, x, length_target=None, energy_target=None, pitch_target=None, mel_max_length=None, length_alpha=1.0, pitch_alpha=1.0, energy_alpha=1.0):
        x, duration = self.length_regulator(x, alpha=length_alpha, target=length_target, mel_max_length=mel_max_length)

        energy = self.energy_predictor(x)
        if energy_target is not None:
            embedding_energy = self.energy_embedding(torch.bucketize(energy_target, self.energy_bins))
        else:
            embedding_energy = self.energy_embedding(torch.bucketize(energy * energy_alpha, self.energy_bins))
        x = x + embedding_energy

        pitch = self.pitch_predictor(x)
        if pitch_target is not None:
            embedding_pitch = self.pitch_embedding(torch.bucketize(pitch_target, self.pitch_bins))
        else:
            embedding_pitch = self.pitch_embedding(torch.bucketize(pitch * pitch_alpha, self.pitch_bins))
        x = x + embedding_pitch

        return x, duration, energy, pitch
