from flask import Flask, render_template, request, redirect, url_for, session
import os
import sounddevice as sd
import soundfile as sf
import torchaudio
import numpy as np
from speechbrain.pretrained import SpeakerRecognition
from scipy.spatial.distance import cosine

app = Flask(__name__)
app.secret_key = 'your-secret-key'  # Needed for session
model = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

DURATION = 5
SAMPLE_RATE = 16000

AUDIO_DIR = "audio"
EMBEDDING_DIR = "embedding"
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(EMBEDDING_DIR, exist_ok=True)

def record_audio(filename):
    print("🎙️ Recording...")
    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    sf.write(filename, audio, SAMPLE_RATE)
    print(f"✅ Saved to {filename}")

def enroll_user(name):
    filename = f"{AUDIO_DIR}/{name}.wav"
    record_audio(filename)

    signal, _ = torchaudio.load(filename)
    signal = signal.squeeze(0)
    embedding = model.encode_batch(signal.unsqueeze(0)).squeeze().detach().numpy()
    np.save(f"{EMBEDDING_DIR}/{name}.npy", embedding)

def verify_user():
    test_filename = f"{AUDIO_DIR}/test.wav"
    record_audio(test_filename)

    signal, _ = torchaudio.load(test_filename)
    signal = signal.squeeze(0)
    test_embedding = model.encode_batch(signal.unsqueeze(0)).squeeze().detach().numpy().flatten()

    best_match = None
    best_score = 0
    THRESHOLD = 0.35

    for fname in os.listdir(EMBEDDING_DIR):
        if fname.endswith(".npy"):
            user_name = fname[:-4].replace("_", " ")
            enrolled = np.load(os.path.join(EMBEDDING_DIR, fname)).flatten()
            similarity = 1 - cosine(test_embedding, enrolled)
            print((similarity, fname))

            if similarity > THRESHOLD and similarity > best_score:
                best_score = similarity
                best_match = user_name

    if best_match:
        return f"👋 Hey {best_match}! Command accepted."
    else:
        return "❌ Unauthorized command."



@app.route('/')
def main_page():
    return render_template('main.html')

@app.route('/issue_command', methods=['POST'])
def issue_command():
    result = verify_user()
    return render_template('main.html', result=result)

@app.route('/add_user')
def add_user_page():
    return render_template('add_user.html')

@app.route('/submit_user', methods=['POST'])
def submit_user():
    name = request.form['username'].strip().replace(" ", "_")
    session['username'] = name
    return render_template('add_user.html', prompt_message=f"Please say: Turn on the lights and lock the door", show_record=True)

@app.route('/record_user', methods=['POST'])
def record_user():
    name = session.get('username')
    if not name:
        return redirect(url_for('add_user_page'))

    enroll_user(name)
    return render_template('add_user.html', message=f"Successfully added {name.replace('_', ' ')} as authorized user")

@app.route('/remove_user')
def remove_user_page():
    users = []
    for fname in os.listdir(EMBEDDING_DIR):
        if fname.endswith(".npy"):
            name = fname[:-4].replace("_", " ")
            users.append(name)
    return render_template('remove_user.html', users=users)

@app.route('/delete_user/<username>', methods=['POST'])
def delete_user(username):
    safe_name = username.replace(" ", "_")
    audio_path = os.path.join(AUDIO_DIR, f"{safe_name}.wav")
    embedding_path = os.path.join(EMBEDDING_DIR, f"{safe_name}.npy")

    if os.path.exists(audio_path):
        os.remove(audio_path)
    if os.path.exists(embedding_path):
        os.remove(embedding_path)

    return redirect(url_for('remove_user_page'))



if __name__ == '__main__':
    app.run(debug=True)
