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
    
    # Find boundaries of the samples that make up each beat. Calculate halfway mark between each beat, which will
    # represent the boundaries of all beats except the first and the last. Prepend 0, and append the last sample,
    # so that we then have boundaries for every beat.
    beat_boundaries = [0] + beats # [(beats[i] + beats[i+1]) / 2 for i in range(len(beats) - 1)] + [beats[-1]]
    
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
    max_distance = np.amax(sim)
    sim /= max_distance
    
    if display:
        plt.imshow(sim)
        plt.colorbar()
        plt.title('Similarity Matrix - {0}'.format(distance_metric))

    return sim


def segment(signal, sr, hop_len, alpha, aggregator=np.median, distance_metric='cityblock', features='mfcc'):
    if features == 'mfcc':
        features = librosa.feature.mfcc(signal, sr)
    elif features == 'chroma':
        features = librosa.feature.chroma_stft(signal, sr)
    elif features == 'tempo':
        features = librosa.feature.tempogram(signal, sr)
    else:
        raise Exception('Segment not called with valid distance measure')

    tempo, beats = librosa.beat.beat_track(signal, sr=sr, hop_length=hop_len)
        
    bsf = beat_sync_features(features, beats, aggregator, display=False)
    assert beats.size == bsf.shape[1]
    
    # Compute (and show for testing) similarity matrix)
    sim = sim_matrix(bsf, sr, hop_len, distance_metric)
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

def evaluate(correct_timestamps, result_timestamps, threshold=2.0):
    # threshold refers to the max difference (in absolute value seconds) a
    # returned timestamp can have from our estimated correct timestamp to be
    # considered accurate

    current_timestamps = correct_timestamps
    precision = 0.
    recall = 0.
    f1 = 0.
    hits = 0.
    i = 0
    while i < len(result_timestamps) and len(current_timestamps) > 0:
        index, val = min(
            enumerate(current_timestamps),
            key=lambda x: abs(x[1] - result_timestamps[i])
        )
        diff = abs(val - result_timestamps[i])
            
        if diff < threshold:
            hits += 1
            current_timestamps = np.delete(current_timestamps, index)

        i += 1
        precision = hits / len(result_timestamps)
        recall = hits / len(correct_timestamps)
        if (precision + recall) != 0:
            f1 = 2 * (precision * recall) / (precision + recall)

    return f1, precision, recall

def auto_test_alpha(signal, sr, correct_timestamps, start_alpha, end_alpha, num_samples, threshold=2.0, 
                    dist_measure='cityblock', feature='mfcc'):
    # threshold refers to the max difference (in absolute value seconds) a
    # returned timestamp can have from our estimated correct timestamp to be
    # considered accurate

    best_alpha = start_alpha
    best_f1 = 0.
    best_precision = 0.
    best_recall = 0.
    best_timestamps = []
    test_alphas = np.linspace(start_alpha, end_alpha, num=num_samples)
    alphax = 0.

    for alpha in test_alphas:
        print "testing alpha ", alpha
        result_timestamps = segment(signal, sr, 1024, alpha, distance_metric=dist_measure, features=feature)
        f1, precision, recall = evaluate(correct_timestamps, result_timestamps, threshold)
        if (f1 > best_f1):
            best_f1 = f1
            best_alpha = alpha
            best_precision = precision
            best_recall = recall
            best_timestamps = result_timestamps
            alphax = alpha

    return {
        'test_alphas': test_alphas,
        'best_f1':  best_f1,
        'best_alpha': best_alpha,
        'best_timestamps': best_timestamps,
        'best_precision': best_precision,
        'best_recall': best_recall,
        'alpha': alphax
    }

def auto_test_distance(signal, sr, correct_timestamps, start_alpha, end_alpha, num_samples, threshold=2.0, 
                       distances=['cityblock'], feature='mfcc'):
    print "testing distances"
    best_data = {
        'test_alphas': [],
        'best_dist': '',
        'best_f1':  0,
        'best_alpha': 0,
        'best_timestamps': [],
        'best_precision': 0,
        'best_recall': 0,
        'alpha': 0.
    }
    for dist_measure in distances:
        test_data = auto_test_alpha(signal, sr, correct_timestamps, start_alpha, end_alpha, num_samples, threshold, dist_measure, feature)
        if test_data['best_f1'] > best_data['best_f1']:
            best_data = test_data
            best_data['best_dist'] = dist_measure
            best_data['alpha'] = test_data['alpha']
    best_data['test_dists'] = distances
    return best_data

def auto_test_features(signal, sr, correct_timestamps, start_alpha, end_alpha, num_samples, threshold=2.0, 
                       distances=['cityblock'], features=['mfcc', 'chroma', 'tempo']):
    print "testing features"
    best_data = {
        'test_alphas': [],
        'test_dists': distances,
        'best_dist': '',
        'best_feature': '',
        'best_f1':  0,
        'best_alpha': 0,
        'best_timestamps': [],
        'best_precision': 0,
        'best_recall': 0
    }
    extra_data = {
        'f1_mfcc': 0.,
        'f1_chroma': 0.,
        'f1_tempo': 0.,
        'alpha_mfcc': 0.,
        'alpha_chroma': 0.,
        'alpha_tempo': 0.
    }
    for feature in features:
        test_data = auto_test_distance(signal, sr, correct_timestamps, start_alpha, end_alpha, num_samples, threshold, distances, feature)
        if test_data['best_f1'] > best_data['best_f1']:
            best_data = test_data
            best_data['best_feature'] = feature
        if feature == 'mfcc' and test_data['best_f1'] > extra_data['f1_mfcc']:
            extra_data['f1_mfcc'] = test_data['best_f1']
            extra_data['alpha_mfcc'] = test_data['alpha']
        if feature == 'chroma' and test_data['best_f1'] > extra_data['f1_chroma']:
            extra_data['f1_chroma'] = test_data['best_f1']
            extra_data['alpha_chroma'] = test_data['alpha']
        if feature == 'tempo' and test_data['best_f1'] > extra_data['f1_tempo']:
            extra_data['f1_tempo'] = test_data['best_f1']
            extra_data['alpha_tempo'] = test_data['alpha']
    best_data['test_features'] = features
    return best_data, extra_data

def main():
    f1_mfcc = 0.
    f1_chroma = 0.
    f1_tempo = 0.
    alpha_mfcc = 0.
    alpha_chroma = 0.
    alpha_tempo = 0.
    songs = ['audio/bohemian_rhapsody.wav',
             'audio/call_me_maybe.wav',
             'audio/come_on.wav',
             'audio/firework.wav',
             'audio/happy_together.wav',
             'audio/hotel_california.wav',
             'audio/raspberry_beret.wav',
             'audio/rolling_in_the_deep.wav',
             'audio/titanium.wav',
             'audio/when_doves_cry.wav']
    correct_timestamps = {
        'audio/bohemian_rhapsody.wav': [5., 60., 120., 163., 188., 252., 299., 354.],
        'audio/call_me_maybe.wav': [.65, 3., 27., 60., 87., 136., 151., 183.],
        'audio/come_on.wav': [0., 6., 45., 84., 102., 106., 146., 172., 212., 252.],
        'audio/firework.wav': [6., 15., 45., 61., 92., 123., 139., 170., 185., 232.],
        'audio/happy_together.wav': [1., 42., 56., 72., 88., 104., 122., 169.],
        'audio/hotel_california.wav': [0., 52., 104., 182., 208., 260., 399.],
        'audio/raspberry_beret.wav': [0., 8., 56., 80., 112., 136., 169., 213],
        'audio/rolling_in_the_deep.wav': [0., 59., 78., 113., 150., 169., 188., 225.],
        'audio/titanium.wav': [0., 32., 65., 99., 162., 203.],
        'audio/when_doves_cry.wav': [1., 19., 34., 65., 95., 110., 155., 171., 202., 216., 225.]
    }
    for song in songs:
        signal, sr = librosa.load(song)
        test_data, extra_data = auto_test_features(signal, sr, correct_timestamps[song], 0.5, 2.0, 5)
        print 'For the song ', song
        print 'The best feature out of {0} is {1}'.format(test_data['test_features'], test_data['best_feature'])
        print 'For which the best distance metric out of {0} is {1}'.format(test_data['test_dists'], test_data['best_dist'])
        print 'For which the best alpha value out of {0} is {1}'.format(test_data['test_alphas'], test_data['best_alpha'])
        print 'It gives an F1 of {0}, with a precision of {1} and recall of {2}'.format(
            test_data['best_f1'],
            test_data['best_precision'],
            test_data['best_recall']
        )
        print 'Our timestamps: {0}'.format(correct_timestamps)
        print 'Best returned timestamps: {0}'.format(test_data['best_timestamps'])
        print 'The best F1 of MFCC is {0}, alpha of {1}'.format(
            extra_data['f1_mfcc'],
            extra_data['alpha_mfcc']
        )
        f1_mfcc += extra_data['f1_mfcc']
        alpha_mfcc += extra_data['alpha_mfcc']
        print 'The best F1 of Chroma is {0}, alpha of {1}'.format(
            extra_data['f1_chroma'],
            extra_data['alpha_chroma']
        )
        f1_chroma += extra_data['f1_chroma']
        alpha_chroma += extra_data['alpha_chroma']
        print 'The best F1 of Tempo is {0}, alpha of {1} \n \n'.format(
            extra_data['f1_tempo'],
            extra_data['alpha_tempo']
        )
        f1_tempo += extra_data['f1_tempo']
        alpha_tempo += extra_data['alpha_tempo']

    f1_mfcc /= len(songs)
    f1_chroma /= len(songs)
    f1_tempo /= len(songs)
    alpha_mfcc /= len(songs)
    alpha_chroma /= len(songs)
    alpha_tempo /= len(songs)
    print 'Average F1 for MFCC is {0}, recommended alpha is {1}'.format(
        f1_mfcc,
        alpha_mfcc
    )
    print 'Average F1 for Chroma is {0}, recommended alpha is {1}'.format(
        f1_chroma,
        alpha_chroma
    )
    print 'Average F1 for Tempo is {0}, recommended alpha is {1}'.format(
        f1_tempo,
        alpha_tempo
    )

if __name__ == "__main__":
    main()

