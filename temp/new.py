from flask import Flask, request, jsonify   # flask
from markupsafe import escape # flask
import json # flask

from os import path
from pydub import AudioSegment
import tensorflow as tf
import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder
import random
import math
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

def preprocess_and_predict(file_path, model_path, num_segments, num_mfcc=13, n_fft=2048, hop_length=512):
    randomno=[1,2,3,4,5,6]
    labels = ['disco', 'rock', 'country', 'classical', 'metal', 'jazz', 'hiphop', 'blues', 'reggae', 'pop']

    # create a label encoder
    le = LabelEncoder()

    # fit and transform the labels to encode them
    encoded_labels = le.fit_transform(labels)

    # to decode, you can use inverse_transform
    decoded_labels = le.inverse_transform(encoded_labels)
    SAMPLE_RATE = 22050
    TRACK_DURATION = 30 # measured in seconds
    SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION

    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)
    #num_mfcc_vectors_per_segment = 216
    # load the saved model
    model = tf.keras.models.load_model(model_path)

    # dictionary to store mfcc
    data = {
        "mfcc": []
    }

    # load audio file
    signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)

    # process all segments of audio file
    for d in range(num_segments):

        # calculate start and finish sample for current segment
        start = samples_per_segment * d
        finish = start + samples_per_segment

        # extract mfcc
        mfcc = librosa.feature.mfcc(y=signal[start:finish], sr=sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
        mfcc = mfcc.T

        # store only mfcc feature with expected number of vectors
        if len(mfcc) == num_mfcc_vectors_per_segment:
            data["mfcc"].append(mfcc.tolist())
            #print("{}, segment:{}".format(file_path, d+1))

    # convert data to np array for model input
    input_data = np.array(data["mfcc"])

    input_data = np.expand_dims(input_data, axis=-1)
    print("Input Data Shape:", input_data.shape)
    # make prediction
    predictions = model.predict(input_data)
    predicted_indices = np.argmax(predictions, axis=1)
    final_output=predicted_indices[1]
    predicted_label = le.inverse_transform([final_output])
    print(predicted_label[0])
    # limit the output to the specified number of segments
    client_id = 'f8475da0781a4dc3a3f43986532c5bc3'
    client_secret = 'e758ade35c314e3c81c89c5e32c3d0d4'

    client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

    genre = predicted_label[0]

    playlists = sp.search(q=f'{genre}', type='playlist', limit=5)
    data = { 'message': 'error', 'songs': []}
    if playlists['playlists']['items']:

        #print("Found Playlists:")
        #for playlist in playlists['playlists']['items']:
            #print(f"- {playlist['name']} (ID: {playlist['id']})")
        number=random.uniform(1,6)
        playlist_index = round(number) - 1

        selected_playlist_id = playlists['playlists']['items'][playlist_index]['id']
        chart = sp.playlist_tracks(selected_playlist_id)

        song_names = [track['track']['name'] for track in chart['items']]
        data['message'] = (f"Top 10 Songs in the {genre} genre from '{playlists['playlists']['items'][playlist_index]['name']}':")
        for i, song_name in enumerate(song_names[:10]):
            data["songs"].append(f"{song_name}")
    else:
        data['message'] = f"No playlist found for the {genre} genre."
    # print(f"No playlist found for the {genre} genre.")


    return data

def genregen(src):
    # files
    dst = "test.wav"
    sound = AudioSegment.from_mp3(src)
    sound.export(dst, format="wav")
    file_path = 'test.wav'
    model_path = 'prodii_model (1).h5'
    num_segments = 6
    predictions = preprocess_and_predict(file_path, model_path, num_segments)
    return predictions

src = './music/Sunflower-(Spider-Man_-Into-the-Spider-Verse)(PagalWorld).mp3'
# predictions = genregen(src)


# Flask API starts here
app = Flask(__name__)

@app.route("/", methods=["GET"])  # dont touch
def home():
    return jsonify({"message": "API is working perfectly"})


@app.route("/mlData/<string:filename>", methods=["GET"])
def api(filename):
    predictions = genregen('./music/' + filename+".mp3")
    # result = {"message": "Received data successfully", "data": f"Hello, {escape(filename)}!", "data2":predictions}
    return jsonify(predictions)
    
if __name__ == "__main__":
    app.run(port=5000, debug=False, passthrough_errors=True, use_debugger=False, use_reloader=False)
