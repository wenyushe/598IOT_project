# enroll_user.py

import sounddevice as sd
import soundfile as sf
import scipy.io.wavfile as wav
import numpy as np
import torchaudio
torchaudio.set_audio_backend("soundfile")
from speechbrain.pretrained import SpeakerRecognition

# Load pretrained speaker recognition model
model = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

# Configuration
DURATION = 5  # seconds
SAMPLE_RATE = 16000
PHRASE = "Open the lights and lock the door"
EMBEDDING_PATH = "embedding/authorized_user_embedding.npy"
AUDIO_PATH = "audio/enrollment_audio.wav"


def record_audio(filename, duration, samplerate):
    print(f"\n🎙️  Please say: '{PHRASE}'")
    print("Recording...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()
    sf.write(filename, audio, samplerate)
    print(f"✅ Recorded and saved to {filename}")


def enroll_user():
    record_audio(AUDIO_PATH, DURATION, SAMPLE_RATE)

    # Load and process audio
    signal, fs = torchaudio.load(AUDIO_PATH)
    signal = signal.squeeze(0)

    # Extract speaker embedding
    embedding = model.encode_batch(signal.unsqueeze(0)).squeeze(0).detach().numpy()

    # Save embedding
    np.save(EMBEDDING_PATH, embedding)
    print(f"✅ Voice enrolled and embedding saved to {EMBEDDING_PATH}")

if __name__ == "__main__":
    enroll_user()
