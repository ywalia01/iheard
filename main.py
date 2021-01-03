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
import torch.nn.functional as F
import torch.optim as optim
from torchvision.datasets.utils import download_url as download_url
from torch.utils.data import random_split
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import torchaudio
import pandas as pd
import numpy as np
import tarfile
import csv
import librosa
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F

app = flask.Flask(__name__)
CORS(app)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

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
model.load_state_dict(torch.load('completed.pth', map_location='cpu'))
print('done')
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
print('done2')
counter = 0
@app.route('/ambient', methods=['GET', 'POST'])
def ambient():
    if request.method == 'POST':
        """
        if os.path.exists("myfile.wav"):
              os.remove("myfile.wav")
        f = request.files['file']
        content = f.read()
        
        with open('myfile.wav', mode='bx') as file:
           file.write(content)

        client = speech.SpeechClient()
        speech_file = "speechtotext.wav"

        rate, data = wf.read(speech_file)
        data0 = data[:,0]

        wf.write("monosound.wav", 44100, data0)

        with io.open("monosound.wav", "rb") as audio_file:
                content = audio_file.read()
        """
        global counter
        songs = ["100795-3-1-0.wav", "143651-2-0-59.wav"]
        data = conversion(songs[counter])
        counter += 1
        print(data.shape)
        datax = data.unsqueeze(0)
        output = model(datax)

        output_numpy = output.detach().numpy()[0, 0, :]
        print(output_numpy.shape)
        #probs = np.exp(output_numpy) / (np.exp(output_numpy)).sum()
        labels = {0: "an air conditioner",
                    1: "a car horn",
                    2: "children_playing",
                    3: "a dog bark",
                    4: "drilling",
                    5: "an engine idling",
                    6: "a gun shot",
                    7: "a jackhammer",
                    8: "a siren",
                    9: "street music"}
        #print(probs)
        i = np.argmax(output_numpy)
        winner = labels[i]
        print(winner)
        winner_string = "I heard " + winner + "."
        return jsonify({'name':winner_string})

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
    
#if __name__ == '__main__':
#    app.run(host='0.0.0.0', port=8080)
