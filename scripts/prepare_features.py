import os
import sys
import torch
import logging
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from templates.speech_recognition.mini_librispeech_prepare import prepare_mini_librispeech
from speechbrain.utils.data_utils import download_file
from speechbrain.utils.distributed import run_on_main

def prepare_features(self, stage, wavs):
    """Prepare features for computation on-the-fly

    Arguments
    ---------
    stage : sb.Stage
        Currently executing stage.
    wavs : tuple
        The input signals (tensor) and their lengths (tensor).
    """
    wavs, wav_lens = wavs

    # Add augmentation if specified. In this version of augmentation, we
    # concatenate the original and the augment batches in a single bigger
    # batch. This is more memory-demanding, but helps to improve the
    # performance. Change it if you run OOM.
    if stage == sb.Stage.TRAIN:
        if hasattr(self.modules, "env_corrupt"):
            wavs_noise = self.modules.env_corrupt(wavs, wav_lens)
            wavs = torch.cat([wavs, wavs_noise], dim=0)
            wav_lens = torch.cat([wav_lens, wav_lens])

        if hasattr(self.hparams, "augmentation"):
            wavs = self.hparams.augmentation(wavs, wav_lens)

    # Feature computation and normalization
    feats = self.hparams.compute_features(wavs)
    feats = self.modules.normalize(feats, wav_lens)

    return feats, wav_lens