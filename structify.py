import librosa
import librosa.display # Must separately be imported
import matplotlib.pyplot as plt

def sim_matrix(signal, sr):
    mfcc = librosa.feature.mfcc(y=signal, sr=sr)
    return librosa.segment.recurrence_matrix(mfcc)

def main():
    # TEMP: create and save a similarity matrix
    signal, sr = librosa.load('audio/call_me_maybe.mp3')
    # Only use an 8th of the song for testing
    matrix = sim_matrix(signal[:len(signal) / 8], sr)

    plt.figure()
    librosa.display.specshow(matrix)
    plt.savefig('temp.png')

if __name__ == "__main__":
    main()

