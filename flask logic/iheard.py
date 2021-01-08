import flask
from flask import request, jsonify
from flask_cors import CORS
from google.cloud import speech_v1p1beta1 as speech
import soundfile as sf
import io
import scipy.io.wavfile as wf
import os
import torch
import torch.nn as nn
import torchaudio
import numpy as np
import librosa

app = flask.Flask(__name__)
CORS(app)

#setting environment path for the Google Speech-to-Text API
#os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = 'path goes here'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Permute(nn.Module):
    def forward(self, x):
        return x.permute(0, 2, 1)
        
class Unsqueeze(nn.Module):
    def forward(self, x):
        return x.unsqueeze(1)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            Unsqueeze(),

            nn.Conv1d(1, 64, 80, 5),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(64, 64, 3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(64, 128, 3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, 3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(128, 256, 3),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 256, 3),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 256, 3),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(256, 512, 3),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 512, 3),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            
            nn.AvgPool1d(19),
            Permute(),
            nn.Linear(512, 10),
            nn.LogSoftmax(dim=2)
            )
    def forward(self, xb):
        return self.network(xb)

model = Net()
model.load_state_dict(torch.load('deep_net_fixed.pth', map_location='cpu'))

def conversion(wav_file):
    torchaudio.USE_SOUNDFILE_LEGACY_INTERFACE = False
    torchaudio.set_audio_backend("soundfile")

    data, samplerate = librosa.load(wav_file, 44100, mono=True)
    sound = torch.from_numpy(data)

    # Fix all wav files to 176400 samples
    padded_data = torch.zeros(176400) #tempData accounts for audio clips that are too short

    if sound.numel() < 176400:
        padded_data[:sound.numel()] = sound[:]
    else:
        padded_data[:] = sound[:176400]
        

    final_data = torch.zeros(32000)
    every_n = 176400 // 32000

    count = 0
    for i in range(32000):
        final_data[i] = padded_data[count]
        count += every_n
        
    return final_data

#app route for ambient noise detection
@app.route('/ambient', methods=['GET', 'POST'])
def ambient():
    if request.method == 'POST':
        
        if os.path.exists("myfile.wav"):
              os.remove("myfile.wav")
              
        f = request.files['file']
        content = f.read()
        
        with open('myfile.wav', mode='bx') as file:
           file.write(content)
        
        data = conversion("myfile.wav")

        datax = data.unsqueeze(0)
        model.eval()
        output = model(datax)
        
        output_numpy = output.detach().numpy()[0, 0, :]

        labels = {0: "an air conditioner",
                    1: "a car horn",
                    2: "children playing",
                    3: "a dog bark",
                    4: "drilling",
                    5: "an engine idling",
                    6: "a gun shot",
                    7: "a jackhammer",
                    8: "a siren",
                    9: "street music"}

        i = np.argmax(output_numpy)
        winner = labels[i]
        winner_string = "I heard " + winner + "."
        return jsonify({'name':winner_string})

#app route for speech-to-text
@app.route('/google', methods=['GET', 'POST'])
def google():
    if request.method == 'POST':
        if os.path.exists("speechtotext.wav"):
              os.remove("speechtotext.wav")
        if os.path.exists("monosound.wav"):
              os.remove("monosound.wav")
              
        f = request.files['file']
        content = f.read()
        
        with open('speechtotext.wav', mode='bx') as file:
           file.write(content)

        client = speech.SpeechClient()
        speech_file = "speechtotext.wav"

        rate, data = wf.read(speech_file)
        data0 = data[:,0]

        wf.write("monosound.wav", 48000, data0)

        with io.open("monosound.wav", "rb") as audio_file:
                content = audio_file.read()

        audio = speech.RecognitionAudio(content=content)

        ob = sf.SoundFile(speech_file)

        first_lang = "en-US"
        second_lang = "es-US"
        third_lang = "zh-cmn-Hans-CN"
        fourth_lang = "hi-IN"

        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=ob.samplerate,
            language_code="en-US",
            alternative_language_codes=[second_lang, third_lang, fourth_lang]
        )

        response = client.recognize(config=config, audio=audio)

        text = ""
        for i, result in enumerate(response.results):
            alternative = result.alternatives[0]
            text = text + alternative.transcript + "\n"
              
        return jsonify({'text':text})
    
if __name__ == '__main__':
    app.run(host='0.0.0.0')
