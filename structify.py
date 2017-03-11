import librosa
import librosa.display # Must separately be imported
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

def beat_track(music, sr, hop_length):
    """
        input: 
            music: an audio signal, single channel array containing all the samples of the signal.
            sr: sample rate to use
            hop_length: the hop length for the stft - without specifying this, the frames output by the beat tracker 
            may not match your feature vector sequence.
        output:
            beats: a list of all the frame indices found by the beat tracker
    """
    tempo, beats = librosa.beat.beat_track(music, sr=sr, hop_length=hop_length)
    return beats

def beat_sync_features(feature_vectors, beats, aggregator = np.median, display = True):
    """
        input:
            feature_vectors: a numpy ndarray MxN, where M is the number of features in each vector and 
            N is the length of the sequence.
            beats: frames given by the beat tracker
            aggregator: how to summarize all the frames within a beat (e.g. np.median, np.mean). Defaults to np.median.
            display: if True, displays the beat synchronous features.
        output:
            beat_synced_features: a numpy ndarray MxB, where M is the number of features in each vector
            and B is the number of beats. Each column of this matrix represents a beat synchronous feature
            vector.
    """
    transposed = np.transpose(feature_vectors)
    sm = np.empty((beats.size, transposed.shape[1]))
    
    split_vectors = np.split(transposed, beats)
    for ind, split_vector in enumerate(split_vectors[1:]):
        # Don't include first split because audio's 0:00 doesn't necessarily mean first beat starts then
        sm[ind] = aggregator(split_vector, axis=0)
        
    sm = np.transpose(sm)
        
    if display:
        plt.pcolor(sm)
        plt.xlabel('Beat frames')
        plt.ylabel('Pitch Class')
        plt.title('Beat Synched Features')
        plt.show()
    
    return sm

def beat_sync_features_alt(feature_vectors, beats, aggregator=np.median, display=True, sr=None, hop_length=None):
    """
        input:
            feature_vectors: a numpy ndarray MxN, where M is the number of features in each vector and 
            N is the length of the sequence.
            beats: frames given by the beat tracker
            aggregator: how to summarize all the frames within a beat (e.g. np.median, np.mean). Defaults to np.median.
            display: if True, displays the beat synchronous features.
        output:
            beat_synced_features: a numpy ndarray MxB, where M is the number of features in each vector
            and B is the number of beats. Each column of this matrix represents a beat synchronous feature
            vector.
    """
    
    # Find boundaries of the samples that make up each beat. Calculate halfway mark between each beat, which will
    # represent the boundaries of all beats except the first and the last. Prepend 0, and append the last sample,
    # so that we then have boundaries for every beat.
    beat_boundaries = [0] + [(beats[i] + beats[i+1]) / 2 for i in range(len(beats) - 1)] + [beats[-1]]
    
    beat_synced_features = np.zeros((feature_vectors.shape[0], beats.size))
    
    # Aggregate features based on indices of beat_boundaries
    for i, start in enumerate(beat_boundaries[:-1]):
        end = beat_boundaries[i + 1]
        beat_synced_features[:, i] = aggregator(feature_vectors[:, start:end], axis = 1)
    
    # Modified version of plotting, from start of #1
    if display:
        plt.figure(figsize=(20, 4))
        librosa.display.specshow(beat_synced_features, sr = sr, hop_length = hop_length,
                                 y_axis = "chroma", x_axis = "frames")
        plt.xlabel("Beat Number")
        
    return beat_synced_features 

def cost(features):
    features_list = np.transpose(features)
    sim = librosa.segment.recurrence_matrix(features, mode='distance')
    return (1.0 / len(features_list)) * (np.sum(sim) / 2.0)

def segment(signal, sr, hop_len, alpha, aggregator=np.median):
    mfcc = librosa.feature.mfcc(y=signal, sr=sr) 
    tempo, beats = librosa.beat.beat_track(signal, sr=sr, hop_length=hop_len)

    bsf = beat_sync_features(mfcc, beats, aggregator, display=False)
    assert beats.size == bsf.shape[1]

    features = np.transpose(bsf)

    DG = nx.DiGraph()

    # Beat frames are nodes
    DG.add_nodes_from(range(len(features)))
    assert len(DG.nodes()) == beats.size

    # Add edges
    for i in range(len(features) - 2):  # - 2 to account for j being i + 1 and
                                        # we add edges between i and j + 1 (aka i and i + 2)
        for j in range(i + 1, len(features) - 1):
            if j - i == 1:
                # librosa's recurrance matrix can't calculate distance for one feature vector
                # But we know distance to itself is 0
                cost_value = alpha
            else:
                cost_value = alpha + cost(np.transpose(features[i:j]))

            DG.add_edge(i, j + 1, weight=cost_value)

    path = nx.dijkstra_path(DG, 0, len(features) - 1)

    # Convert beat frames to time
    beat_frames = []
    for index in path:
        beat_frames.append(beats[index])
    return librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_len)

def main():
    signal, sr = librosa.load('audio/toy.wav')
    signal = signal[:len(signal) / 2] # Half length for testing
    print segment(signal, sr, 1024, 1.3)

if __name__ == "__main__":
    main()

