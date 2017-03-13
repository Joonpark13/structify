import librosa
import librosa.display # Must separately be imported
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import scipy as sp

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


def beat_sync_features(feature_vectors, beats, aggregator=np.median, display=True, sr=None, hop_length=None):
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
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(beat_synced_features, sr = sr, hop_length = hop_length,
                                 x_axis = "frames")
        plt.xlabel("Beat Number")
        plt.ylabel("Some Feature idk")
        plt.show()
        
    return beat_synced_features


def cost(i, j, sim):
    return (1.0 / sim.shape[0]) * (np.sum(sim[i:j, i:j]) / 2.0)


def sim_matrix(feature_vectors, sample_rate, hop_length, distance_metric = 'cosine', display = True):
    """
        Input:
            feature_vectors - a numpy ndarray MxN, where M is the number of features in each vector and 
            N is the length of the sequence.
            sample_rate - sample rate of the original audio
            hop_length - the length of the hop used in the representation
            distance_metric - which distance metric to use to compute similarity. Defaults to cosine.
            display - whether or not to display the similarity matrix after computing it. Defaults to True.
        Output:
            if display is True, plot the similarity matrix. Along the x and y axis of the similarity matrix, 
            the ticks should be in seconds not in samples. 
            returns sim - an NxN matrix with the pairwise distance between every feature vector.
                                 0 is no distance, 1 is max distance
    """
    
    # Compute distance matrix
    sim = sp.spatial.distance.cdist(feature_vectors.T, feature_vectors.T, distance_metric)
    
    # Normalize by max distance
    max_distance = np.amax(sim)
    sim /= max_distance
    
    if display:
        plt.imshow(sim)
        plt.colorbar()
        plt.title('Similarity Matrix - {0}'.format(distance_metric))

    return sim


def segment(signal, sr, hop_len, alpha, aggregator=np.median):
    mfcc = librosa.feature.mfcc(signal, sr)
    tempo, beats = librosa.beat.beat_track(signal, sr=sr, hop_length=hop_len)
        
    bsf = beat_sync_features(mfcc, beats, aggregator, display=False)
    assert beats.size == bsf.shape[1]
    
    # Compute (and show for testing) similarity matrix)
    sim = sim_matrix(bsf, sr, hop_len, "cityblock")
    plt.show()

    DG = nx.DiGraph()

    # Beat frames are nodes
    DG.add_nodes_from(range(sim.shape[0]))
    assert len(DG.nodes()) == beats.size

    # Add edges
    for i in range(sim.shape[0] - 2):  # - 2 to account for j being i + 1 and
                                        # we add edges between i and j + 1 (aka i and i + 2)
        for j in range(i + 1, sim.shape[0] - 1):
            if j - i == 1:
                # librosa's recurrance matrix can't calculate distance for one feature vector
                # But we know distance to itself is 0
                cost_value = alpha
            else:
                cost_value = alpha + cost(i, j, sim)

            DG.add_edge(i, j + 1, weight=cost_value)

    path = nx.dijkstra_path(DG, 0, sim.shape[0] - 1)

    # Convert beat frames to time
    beat_frames = []
    for index in path:
        beat_frames.append(beats[index])
    return librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_len)


def main():
    signal, sr = librosa.load('audio/toy2.wav')
    print segment(signal, sr, 1024, 700.0)

if __name__ == "__main__":
    main()

