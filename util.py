import numpy as np
import re
from dataclasses import dataclass
from typing import List
from pydub import AudioSegment

@dataclass
class WordBoundary:
    start_sec: float
    end_sec: float
    label: str

@dataclass 
class TrialData:
    transcript: str
    outcome: str 
    word_boundaries: List[WordBoundary]
    condition: str
    continuum_step: int
    target_word: str
    target_array: np.ndarray = None

def load_to_array(path: str) -> np.ndarray:
    """Loads audio, resamples to 16kHz mono, and returns a normalized float32 array."""
    seg = AudioSegment.from_wav(path).set_frame_rate(16000).set_channels(1)
    return np.array(seg.get_array_of_samples(), dtype=np.float32) / 32768.0

def transcribe_batch(arrays, processor, model, device):
    """Feeds a batch of numpy arrays directly to Whisper."""
    inputs = processor(arrays, sampling_rate=16000, return_tensors="pt", padding=True)
    input_features = inputs["input_features"].to(device)
    generated_ids = model.generate(input_features)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)