import numpy as np
import librosa
from pydub import AudioSegment

sr=16000

def audiosegment_to_array(seg, target_sr=sr):
    samples = np.array(seg.get_array_of_samples())
    max_val = float(1 << (8 * seg.sample_width - 1))
    y = samples.astype(np.float32) / max_val
    if seg.frame_rate != target_sr:
        y = librosa.resample(y, orig_sr=seg.frame_rate, target_sr=target_sr)
    return y


def transcribe_batch(segments, processor, model, sr=16000, device=None, batch_size=8):
    model.eval()
    texts = []
    for i in range(0, len(segments), batch_size):
        chunk = segments[i:i+batch_size]
        arrays = [audiosegment_to_array(seg, target_sr=sr) for seg in chunk]
        inputs = processor(arrays, sampling_rate=sr, return_tensors="pt", padding=True)
        input_features = inputs["input_features"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        gen_ids = model.generate(input_features=input_features, attention_mask=attention_mask, task="transcribe")  
        texts.extend(processor.batch_decode(gen_ids, skip_special_tokens=True))
    return [t.strip() for t in texts]

def create_filler(target, frames, talker, n_continuum):
    silence = AudioSegment.silent(duration=500)
    combined = AudioSegment.silent(duration=0)
    for f in frames:
        audio = AudioSegment.from_wav(f'audio/fillers/{talker}/{f}')
        combined += audio + silence

    trials = [combined + AudioSegment.from_wav(f'audio/MP/{talker}/continuum/{target}_1_{i}.wav') + silence for i in range(n_continuum)]
    return trials, frames