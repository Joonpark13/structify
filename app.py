from flask import Flask, render_template, request, redirect, url_for
import os
import librosa

app = Flask(__name__)

def is_wav(filename):
    return '.' in filename and filename.split('.')[-1] == 'wav'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
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
            f.save(os.path.join('static', 'temp.wav'))
            signal, sr = librosa.load('static/temp.wav')
            return 'success'
    if request.method == 'GET':
        return 'upload'

if __name__ == '__main__':
    app.run()
