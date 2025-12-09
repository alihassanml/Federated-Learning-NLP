# server: adaptive_privacy.py
import math

class AdaptivePrivacy:
    def __init__(self, base_noise=0.1, min_noise=0.01, decay=0.95):
        self.base_noise = base_noise
        self.min_noise = min_noise
        self.decay = decay
        self.last_loss = None

    def update_loss(self, loss):
        self.last_loss = float(loss)

    def get_noise_multiplier(self, round_idx):
        # simple scheme: reduce noise as loss drops; combine with decay over rounds
        if self.last_loss is None:
            return self.base_noise * (self.decay ** round_idx)
        # example schedule: noise ~ base * sigmoid(loss - threshold)
        threshold = 1.0
        factor = 1.0 / (1.0 + math.exp(-(self.last_loss - threshold)))
        noise = max(self.min_noise, self.base_noise * factor * (self.decay ** round_idx))
        return float(noise)
