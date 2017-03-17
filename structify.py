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


def beat_sync_features(feature_vectors, beats, aggregator=np.median, display=False, sr=None, hop_length=None):
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
    
    # Prepend 0 to beat_boundaries to account for first beat
    beat_boundaries = [0] + beats
    
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
    cost = (1.0 / (j - i + 1)) * (np.sum(sim[i:j+1, i:j+1]) / 2.0)
    return cost


def sim_matrix(feature_vectors, sample_rate, hop_length, distance_metric='cityblock', display=False):
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
    sim /= np.amax(sim)
    
    if display:
        plt.imshow(sim)
        plt.colorbar()
        plt.title('Similarity Matrix - {0}'.format(distance_metric))

    return sim


def segment(signal, sr, hop_len, alpha, aggregator=np.median, distance_metric='cityblock'):
    mfcc = librosa.feature.mfcc(signal, sr)
    tempo, beats = librosa.beat.beat_track(signal, sr=sr, hop_length=hop_len)
        
    bsf = beat_sync_features(mfcc, beats, aggregator, display=False)
    assert beats.size == bsf.shape[1]
    
    # Compute beat synchronous similarity matrix
    sim = sim_matrix(bsf, sr, hop_len, distance_metric)

    # Build directed graph to segment song
    DG = nx.DiGraph()

    N = sim.shape[0]  # number of beats

    # Beat frames are nodes
    DG.add_nodes_from(range(N))
    assert len(DG.nodes()) == beats.size

    # Edges are costs of segments between node i and node j
    # If graph has N nodes, starts of paths should be 0, 1, ... , N - 2
    for i in range(N - 1):
        # For some i, ends of paths should be i+1, i+2, ... , N - 1
        for j in range(i + 1, N):
            cost_value = alpha + cost(i, j - 1, sim)
            DG.add_edge(i, j, weight = cost_value)

    path = nx.dijkstra_path(DG, 0, N - 1)

    # Convert beat frames to time
    beat_frames = []
    for index in path:
        beat_frames.append(beats[index])

    return librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_len)


def evaluate(correct_timestamps, result_timestamps, threshold=2.0):
    # threshold refers to the max difference (in absolute value seconds) a
    # returned timestamp can have from our estimated correct timestamp to be
    # considered accurate

    current_timestamps = correct_timestamps
    total = 0.0
    hits = 0
    misses = 0
    i = 0
    while i < len(result_timestamps) and len(current_timestamps) > 0:
        index, val = min(
            enumerate(current_timestamps),
            key=lambda x: abs(x[1] - result_timestamps[i])
        )
        diff = abs(val - result_timestamps[i])
            
        if diff < threshold:
            total += 1
            hits += 1
            current_timestamps = np.delete(current_timestamps, index)
        else:
            total -= 0.5
            misses += 1

        i += 1

    return max((total / len(correct_timestamps)), 0), hits, misses


def auto_test_alpha(signal, sr, correct_timestamps, start_alpha, end_alpha, num_samples, threshold=2.0):
    # threshold refers to the max difference (in absolute value seconds) a
    # returned timestamp can have from our estimated correct timestamp to be
    # considered accurate

    best_alpha = start_alpha
    best_eval = 0
    best_hits = 0
    best_misses = 0
    best_timestamps = []
    test_alphas = np.linspace(start_alpha, end_alpha, num=num_samples)

    for alpha in test_alphas:
        result_timestamps = segment(signal, sr, 1024, alpha)
        value, hits, misses = evaluate(correct_timestamps, result_timestamps, threshold)
        if (value > best_eval):
            best_eval = value
            best_alpha = alpha
            best_hits = hits
            best_misses = misses
            best_timestamps = result_timestamps

    return {
        'test_alphas': test_alphas,
        'best_eval':  best_eval,
        'best_alpha': best_alpha,
        'best_timestamps': best_timestamps,
        'best_hits': best_hits,
        'best_misses': best_misses
    }


def plot_segmented_signal(signal, sr, segments, song_title):
    """Plot audio signal with lines representing segmentations.

        input:
            signal: audio signal of segmented song (1D numpy array)
            sr: sample rate
            segments: 1D numpy array denoting the segment boundaries
            song_title: name of song
        
        output:
            None, but saves image to disk.
    """


    times = librosa.samples_to_time(range(len(signal)), sr=sr)

    plt.figure(figsize=(10, 4))

    plt.plot(times, signal, color='b')

    for timestamp in segments:
        plt.axvline(timestamp, color='r')

    plt.title(song_title)
    plt.xlabel('Time (s)')
    plt.savefig('segmented_signal.png')


def create_segmented_audio(signal, sr, segments, beep_signal, song_title):
    """Add beeps to audio signal to represent segmentations.

        input:
            signal: audio signal of segmented song (1D numpy array)
            sr: sample rate
            segments: 1D numpy array denoting the segment boundaries
            beep_signal: audio signal of beep (1D numpy array)
            song_title: name of song

        output:
            None, but saves audio file to disk
    """

    # Times to add beeps
    segments = librosa.time_to_samples(segments, sr = sr)

    # Make signal softer and shorter
    beep_signal /= 2.0
    beep_signal = beep_signal[ : len(beep_signal) / 2]
    beep_len = len(beep_signal)

    # Add beep at start of every segment
    for start in segments:
        signal[start : start+beep_len] += beep_signal

    # Save to disk as new wav file
    fname = "audio/{0}_segmented.wav".format(song_title)
    librosa.output.write_wav(fname, signal, sr)


def main():
    song = 'audio/call_me_maybe.wav'
    signal, sr = librosa.load(song)
    print 'Loaded song from {0}.'.format(song)

    hop_len = 1024
    alpha = 1.3
    segments = segment(signal, sr, hop_len, alpha, distance_metric = 'euclidean')
    print 'Successfully segmented song at times:'
    print segments

    # Create output image showing segmentations
    song_title = song[6:-4]  # song begins with 'audio/' and ends with '.wav'
    plot_segmented_signal(signal, sr, segments, song_title)
    print 'Generated image of segmented audio signal.'

    # Create output audio with beeps denoting segmentations
    beep, _ = librosa.load('audio/beep.wav')
    create_segmented_audio(signal, sr, segments, beep, song_title)
    print 'Generated segmented audio in audio/' + song_title + '_segmented.wav.'


if __name__ == "__main__":
    main()

