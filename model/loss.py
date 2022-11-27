import torch
import torch.nn as nn


class FastSpeechLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

    def forward(self, mel, duration_predicted, mel_target, duration_predictor_target):
        mel_loss = self.mse_loss(mel, mel_target)

        duration_predictor_loss = self.l1_loss(duration_predicted,
                                               duration_predictor_target.float())

        return mel_loss, duration_predictor_loss


class FastSpeech2Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

    def forward(self, mel, duration_predicted, energy_predicted, pitch_predicted, mel_target, duration_predictor_target, pitch_target, energy_target):
        mel_loss = self.l1_loss(mel, mel_target)

        duration_predictor_loss = self.mse_loss(torch.log(duration_predicted.float() + 1), torch.log(duration_predictor_target.float() + 1))

        energy_predicted_loss = self.mse_loss(energy_target, energy_predicted)

        pitch_predicted_loss = self.mse_loss(pitch_target, pitch_predicted)

        return mel_loss, duration_predictor_loss, energy_predicted_loss, pitch_predicted_loss
