import os
from os.path import join
import torch
import speechbrain as sb
from speechbrain.pretrained import WhisperASR
from speechbrain.pretrained import EncoderDecoderASR

class TranscribeSpeechbrain:
    def __init__(self):        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"        
        self.pipe = None

    def initialize_pipe(self):
        self.asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-crdnn-rnnlm-librispeech", savedir="pretrained_models/asr-crdnn-rnnlm-librispeech")
        

    def transcribe(self, audio_file :str):
        self.asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-crdnn-rnnlm-librispeech", savedir="pretrained_models/asr-crdnn-rnnlm-librispeech")
        transcription = self.asr_model.transcribe_file(audio_file)
        return transcription, "Hindi audio transcribed from SpeechBrain"            