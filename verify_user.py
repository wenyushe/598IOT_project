# verify_user.py

import sounddevice as sd
import soundfile as sf
import scipy.io.wavfile as wav
import numpy as np
import torchaudio
from scipy.spatial.distance import cosine
from speechbrain.pretrained import SpeakerRecognition

# Load pretrained model
model = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

# Config
DURATION = 5  # seconds
SAMPLE_RATE = 16000
EMBEDDING_PATH = "embedding/authorized_user_embedding.npy"
AUDIO_PATH = "audio/verification_audio.wav"
SIMILARITY_THRESHOLD = 0.6


def record_audio(filename, duration, samplerate):
    print("\n🎙️  Please speak now...")
    print("Recording...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()
    sf.write(filename, audio, samplerate)
    print(f"✅ Recorded and saved to {filename}")


def verify_user():
    record_audio(AUDIO_PATH, DURATION, SAMPLE_RATE)

    enrolled_embedding = np.load(EMBEDDING_PATH).flatten()

    # Load and process new audio
    signal, fs = torchaudio.load(AUDIO_PATH)
    signal = signal.squeeze(0)
    test_embedding = model.encode_batch(signal.unsqueeze(0)).squeeze().detach().numpy()
    test_embedding = test_embedding.flatten()

    # Cosine similarity
    similarity = 1 - cosine(enrolled_embedding, test_embedding)
    print(f"Similarity score: {similarity:.4f}")

    if similarity >= SIMILARITY_THRESHOLD:
        print("✅ Authorized")
    else:
        print("❌ Unauthorized")

if __name__ == "__main__":
    verify_user()
