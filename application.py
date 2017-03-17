from gevent.wsgi import WSGIServer
from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import librosa
from structify import segment, plot_segmented_signal, create_segmented_audio

application = Flask(__name__)

def is_wav(filename):
    return '.' in filename and filename.split('.')[-1] == 'wav'

@application.route('/')
def index():
    return render_template('index.html')

@application.route('/upload', methods=['GET', 'POST'])
def file_upload():
    if request.method == 'POST':
    	# check if the post request has the file part
        if 'audio' not in request.files:
            return redirect(request.url)
        f = request.files['audio']
        # if user does not select file, browser also
        # submit a empty part without filename
        if f.filename == '':
            return redirect(request.url)
        if f and is_wav(f.filename):
            f.save(os.path.join('audio', 'temp.wav'))
            signal, sr = librosa.load('audio/temp.wav')

            hop_len = 1024
            alpha = 1.3
            timestamps = segment(signal, sr, hop_len, alpha)

            # Saves as 'static/segmented_signal.png'
            image_filename = plot_segmented_signal(signal, sr, timestamps, f.filename)

            beep, _ = librosa.load('audio/beep.wav')
            # Saves as 'audio/segmented.wav'
            audio_filename = create_segmented_audio(signal, sr, timestamps, beep)

            return jsonify({'image': image_filename, 'audio': audio_filename})

    if request.method == 'GET':
        return 'upload'

if __name__ == '__main__':
    http_server = WSGIServer(('', 5000), application)
    http_server.serve_forever()
