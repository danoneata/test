""" Use SIFT descriptors for the videos. """
import cPickle
from ipdb import set_trace
import numpy as np
import os
import random
from sklearn import cross_validation

import descriptors
from yael.yael import gmm_read

from fisher_vectors.evaluation import Evaluation
from fisher_vectors.per_slice.discriminative_detection import aggregate
from fisher_vectors.per_slice.discriminative_detection import _normalize
from fisher_vectors.model import FVModel
from fisher_vectors.model.utils import standardize
from fisher_vectors.model.utils import power_normalize
from fisher_vectors.model.utils import L2_normalize


def get_per_video_sift_data(split, subsample=1, nr_clusters=64, color=10):
    """ Loads the SIFT descriptors per video. `color` is a special flag; if it
    is set to 10 the descriptors are loaded unnormalized.

    """
    filename = ("/home/lear/oneata/tmp/trecvid11_sift/"
                "sift_per_video_%s_subsample%d_k%d_color%d.raw" % (
                    split, subsample, nr_clusters, color))

    if os.path.exists(filename):
        # Load data from cache file.
        print "Load per video data", filename
        with open(filename, "r") as ff:
            video_data = np.load(ff)
            video_labels = np.load(ff)
            video_names = cPickle.load(ff)
        return video_data, video_labels, video_names

    data, labels, limits = descriptors.load_dan_split(
        split, subsample, nr_clusters, color)
    # Get uniform weights within each video.
    weights = _normalize(np.ones_like(labels, dtype=np.float), limits, 'L1')
    # Load or compute data.
    video_data = aggregate(data, weights, limits)
    video_labels = np.array([labels[low] for low in limits[:-1]])
    video_names = descriptors.vid_names_dan_split(split)
    # Save data to file.
    with open(filename, "w") as ff:
        np.save(ff, video_data)
        np.save(ff, video_labels)
        cPickle.dump(video_names, ff)

    return video_data, video_labels, video_names


def get_per_video_mbh_data(split, suffix=''):
    """ Returns the 'train' or 'test' video descriptors. The `suffix` argument
    can be '_morenull' to load the data with 5296 null samples. 
    
    """
    base_path = ('/home/clear/oneata/data/trecvid12/features/'
                 'dense5.track15mbh.small.skip_1/')
    sstats_fn = os.path.join(
        base_path, 'statistics_k_256', '%s%s.dat' % (split, suffix))
    labels_fn = os.path.join(
        base_path, 'statistics_k_256', 'labels_%s%s.info' % (split, suffix))
    info_fn = os.path.join(
        base_path, 'statistics_k_256', 'info_%s%s.info' % (split, suffix))
    gmm_fn = os.path.join(base_path, 'gmm', 'gmm_256')

    sstats = np.fromfile(sstats_fn, dtype=np.float32)
    labels = np.array([tuple_label[0]
              for tuple_label in cPickle.load(open(labels_fn, 'r'))])
    video_names = cPickle.load(open(info_fn, 'r'))['video_names']

    # Convert sufficient statistics to Fisher vectors.
    gmm = gmm_read(open(gmm_fn, 'r'))
    data = FVModel.sstats_to_features(sstats, gmm)

    return data, labels, video_names


def data_to_kernels(tr_data, te_data):
    tr_data, mu, sigma = standardize(tr_data)
    tr_data = power_normalize(tr_data, 0.5)
    tr_data = L2_normalize(tr_data)

    te_data, _, _ = standardize(te_data, mu, sigma)
    te_data = power_normalize(te_data, 0.5)
    te_data = L2_normalize(te_data)

    tr_kernel = np.dot(tr_data, tr_data.T)
    te_kernel = np.dot(te_data, tr_data.T)

    return tr_kernel, te_kernel


def get_dummy_data(split):
    D = 200
    N = 1050 if split == 'train' else 1000
    data = np.random.rand(N, D)
    labels = np.random.randint(0, 16, N)
    return data, labels


def label_as_matthijs(labels, idx_to_cls):
    """ Remap labels such that they match Matthijs'. """
    mappings = {
        "birthday_party": 0,
        "board_trick": 1,
        "changing_vehicle_tire": 2,
        "feeding_an_animal": 3,
        "flash_mob_gathering": 4,
        "grooming_an_animal": 5,
        "landing_a_fish": 6,
        "making_a_sandwich": 7,
        "parade": 8,
        "parkour": 9,
        "repairing_an_appliance": 10,
        "sewing_project": 11,
        "unstuck_vehicle": 12,
        "wedding_ceremony": 13,
        "woodworking_project": 14,
        "null": 15}
    new_labels = [mappings[idx_to_cls[label]] for label in labels]
    return np.array(new_labels)


def subsample_null_class(labels, proportion, random_state=0):
    null_class_idx = 0
    other_idxs = np.where(labels != null_class_idx)[0]
    null_idxs = np.where(labels == null_class_idx)[0]
    nr_null_samples = len(labels[null_idxs])
    rs = cross_validation.ShuffleSplit(
        nr_null_samples, n_iterations=1, test_size=proportion,
        random_state=random_state)
    _, split_idxs = iter(rs).next()
    idxs = np.hstack((other_idxs, null_idxs[split_idxs]))
    return idxs


def get_data(features, split, **kwargs):
    """ Loads data for the specified features. """
    if features == 'sift' or features == 'color':
        subsample = kwargs.get('subsample')
        nr_clusters = kwargs.get('nr_clusters')
        color = kwargs.get('color')
        data, labels, _ = get_per_video_sift_data(
            split, subsample, nr_clusters, color)
    elif features == 'mbh':
        suffix = kwargs.get('suffix', '')
        data, labels, _ = get_per_video_mbh_data(split, suffix)
    elif features == 'dummy':
        data, labels = get_dummy_data(split)
    return data, labels


def vary_nr_negatives():
    null_class_idx = 0
    feature = 'mbh'
    params = {
        'dummy': {},
        'mbh': {'suffix': '_morenull'},
        'sift': {'subsample': 10, 'nr_clusters': 64, 'color': 0}}

    tr_data, tr_labels = get_data(feature, 'train', **params[feature])
    te_data, te_labels = get_data(feature, 'test', **params[feature])

    outfilename = '/home/lear/oneata/data/trecvid12/results/tmp.txt'
    with open(outfilename, 'a') as ff:
        ff.write('%s %s\n' % (feature, params[feature].__str__()))
        ii, nr_repeats = 0, 5
        for ii in xrange(nr_repeats):
            for proportion in (0.02, 0.04, 0.08, 0.1, 0.2, 0.4, 0.8, 1.):
                if proportion < 1.0:
                    idxs = subsample_null_class(tr_labels, proportion, ii)
                else:
                    idxs = np.arange(len(tr_labels))
                    random.seed(ii)
                    random.shuffle(idxs)
                _tr_data, _tr_labels = tr_data[idxs], tr_labels[idxs]

                tr_kernel, te_kernel = data_to_kernels(_, te_data)
                #eval = Evaluation('trecvid12', eval_type='trecvid11')
                #score = eval.fit(tr_kernel, tr_labels).score(te_kernel, te_labels)

                from fisher_vectors.evaluation import trecvid12_parallel as eval
                fit_out = eval.fit(tr_kernel, _tr_labels)
                score = eval.score(te_kernel, te_labels, fit_out)

                print score
                ff.write('%1.2f %d %2.3f \n' % (
                    proportion,
                    len(idxs) - len(_tr_labels[_tr_labels != null_class_idx]),
                    score))


def evaluate():
    params = {
        'mbh': {},
        'dummy': {},
        'sift': {'subsample': 10, 'nr_clusters': 64, 'color': 0},
        'color': {'subsample': 10, 'nr_clusters': 64, 'color': 1}}
    features = ['dummy'] #, 'sift']

    for feature in features:
        tr_data, tr_labels = get_data(feature, 'train', **params[feature])
        te_data, te_labels = get_data(feature, 'test', **params[feature])

        Kxx, Kyx = data_to_kernels(tr_data, te_data)
        try:
            tr_kernel += Kxx
            te_kernel += Kyx
        except NameError:
            tr_kernel = Kxx
            te_kernel = Kyx

    from fisher_vectors.evaluation import trecvid12_parallel as eval
    fit_out = eval.fit(tr_kernel, tr_labels)
    print eval.score(te_kernel, te_labels, fit_out)


def main():
    vary_nr_negatives()
    #evaluate()


if __name__ == '__main__':
    main()
