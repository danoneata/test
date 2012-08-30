import cPickle
from ipdb import set_trace
import itertools
import multiprocessing as mp
import numpy as np
from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV
import sys

from per_video import data_to_kernels
from per_video import get_data
from per_video import remap_descriptors
from per_slice import binarize_labels
from fisher_vectors.evaluation.utils import average_precision


null_class_idx = 0
idx_to_class = {
     0: 'null',
     1: 'board_trick',
     2: 'feeding_an_animal',
     3: 'landing_a_fish',
     4: 'wedding_ceremony',
     5: 'woodworking_project',
     6: 'birthday_party',
     7: 'changing_vehicle_tire',
     8: 'flash_mob_gathering',
     9: 'unstuck_vehicle',
    10: 'grooming_an_animal',
    11: 'making_a_sandwich',
    12: 'parade',
    13: 'parkour',
    14: 'repairing_an_appliance',
    15: 'sewing_project',
    21: 'attempting_a_bike_trick',
    22: 'cleaning_an_appliance',
    23: 'dog_show',
    24: 'giving_directions_to_a_location',
    25: 'marriage_proposal',
    26: 'renovating_a_home',
    27: 'rock_climbing',
    28: 'town_hall_meeting',
    29: 'winning_a_race_without_a_vehicle',
    30: 'working_on_a_metal_crafts_project',
}


combinations = {
    'sift_mu':       ('sift2', {'dimensions' : (255, 255 + 32 * 256) }),
    'sift_sigma':    ('sift2', {'dimensions' : (255 + 32 * 256, 255 + 32 * 256 * 2) }),
    'sift_mu_sigma': ('sift2', {'dimensions' : (255, 255 + 32 * 256 * 2) }),
    'mbh':           ('mbh', {}),
    'audio':         ('heng_audio', {'derivative': ''}),
    'audio_D1':      ('heng_audio', {'derivative': '_D1'}),
    'audio_D2':      ('heng_audio', {'derivative': '_D2'}),
    'jochen_audio':  ('jochen_audio',{'dimslice': 0})}


late_fusion_params = {
    'score_type': 'scores'}


class LateFusion(object):
    def __init__(self, score_type):
        self.score_type = score_type

    def fit(self, kernels, labels):
        scores, self.clf = [], []
        kernels = list(kernels)

        # Get a hold-out data for fitting the late fusion weights.
        ss = StratifiedShuffleSplit(labels, 1, test_size=0.25, random_state=0)
        tr_idxs, val_idxs = iter(ss).next()
        # nr_samples = len(labels)
        # tr_idxs, val_idxs = np.arange(nr_samples), np.arange(nr_samples)
        k_tr_idxs = np.ix_(tr_idxs, tr_idxs)
        k_val_idxs = np.ix_(val_idxs, tr_idxs)

        for kernel in kernels:
            self.clf.append(SVM())
            self.clf[-1].fit(kernel[k_tr_idxs], labels[tr_idxs])
            scores.append(self.predict_clf(self.clf[-1], kernel[k_val_idxs]))
        self.weights = self._get_late_fusion_weights(scores, labels[val_idxs])

        # Retrain on all the data.
        for ii, kernel in enumerate(kernels):
            self.clf[ii].fit(kernel, labels)

        return self

    def predict(self, te_kernels):
        scores = []
        for ii, kernel in enumerate(te_kernels):
            scores.append(self.predict_clf(self.clf[ii], kernel))
        fused_scores = self._fuse_scores(scores, self.weights)
        return fused_scores

    def score(self, te_kernels, te_labels):
        return average_precision(te_labels, self.predict(te_kernels))

    def predict_clf(self, clf, kernel):
        if self.score_type == 'probas':
            return clf.predict_proba(kernel)
        elif self.score_type == 'scores':
            return clf.decision_function(kernel)

    def get_weights_str(self):
        return ' '.join(['%.2f' % ww for ww in self.weights])

    def _get_late_fusion_weights(self, scores, tr_labels):
        #return np.array([1. / len(scores)] * len(scores))  # Equal weights.
        best_ap = 0
        for weights in self.weights_grid(len(scores)):
            ap = average_precision(
                tr_labels, self._fuse_scores(scores, weights))
            if ap > best_ap:
                best_ap = ap
                best_weights = weights
        return best_weights

    @staticmethod
    def _fuse_scores(scores, weights):
        return np.sum(np.vstack(scores).T * weights, 1)

    @staticmethod
    def weights_grid(dd, step = 0.02):
        """ Generates weights on a regular grid. """
        for ww in itertools.product(
            *(np.arange(0, 1 + step, step) for ii in xrange(dd - 1))):
            last_weight = 1 - sum(ww)
            if last_weight < 0:
                continue
            yield ww + (last_weight, )


class MySVC(SVC):
    def predict(self, X):
        return self.decision_function(X)


class SVM(object):
    def __init__(self, **kwargs):
        self.nr_processes = kwargs.get('nr_processes', 1)
        c_values = np.power(3.0, np.arange(-2, 8))
        self.parameters = [{'C': c_values}]

    def fit(self, tr_kernel, tr_labels):
        splits = StratifiedShuffleSplit(
            tr_labels, 3, test_size=0.25, random_state=0)
        my_clf = MySVC(kernel='precomputed',probability=True,
                         class_weight='auto')
        self.clf = (
            GridSearchCV(my_clf, self.parameters,
                         score_func=average_precision,
                         cv=splits, n_jobs=self.nr_processes))
        self.clf.fit(tr_kernel, tr_labels)
        return self

    def decision_function(self, te_kernel):
        return np.squeeze(
            self.clf.best_estimator_.decision_function(te_kernel))

    def predict_proba(self, te_kernel):
        return self.clf.predict_proba(te_kernel)[:, 1]

    def score(self, te_kernel, te_labels):
        return average_precision(te_labels, self.predict_proba(te_kernel))


def get_kernels_given_class(tr_kernels, te_kernels, tr_labels, te_labels,
                            class_idx):
    tr_idxs = ((tr_labels == class_idx) |
               (tr_labels == null_class_idx))
    te_idxs = ((te_labels == class_idx) |
               (te_labels == null_class_idx))

    tr_kernel_idxs = np.ix_(tr_idxs, tr_idxs)
    te_kernel_idxs = np.ix_(te_idxs, tr_idxs)

    for tr_kernel, te_kernel in itertools.izip(tr_kernels, te_kernels):
        yield tr_kernel[tr_kernel_idxs], te_kernel[te_kernel_idxs]


def get_labels_given_class(tr_labels, te_labels, class_idx):
    tr_idxs = ((tr_labels == class_idx) |
               (tr_labels == null_class_idx))
    te_idxs = ((te_labels == class_idx) |
               (te_labels == null_class_idx))

    cls_tr_labels = binarize_labels(tr_labels[tr_idxs], pos_label=class_idx,
        neg_label=null_class_idx)
    cls_te_labels = binarize_labels(te_labels[te_idxs], pos_label=class_idx,
        neg_label=null_class_idx)

    return cls_tr_labels, cls_te_labels


def get_kernels():
    selection = sys.argv[1:]

    ref_tr_vidnames = None
    ref_te_vidnames = None

    for cname in selection:
        feature, params = combinations[cname]

        print "load feature", cname
        
        tr_data, tr_labels, tr_vidnames = get_data(feature, 'train_balanced', **params)
        te_data, te_labels, te_vidnames = get_data(feature, 'test_balanced', **params)

        print "compute kernels train %d*%d test %d*%d" % (
            tr_data.shape + te_data.shape)

        if ref_tr_vidnames != None:
            print "remapping names"
            # pdb.set_trace()

            te_data = remap_descriptors(te_data, te_vidnames, ref_te_vidnames)
            tr_data = remap_descriptors(tr_data, tr_vidnames, ref_tr_vidnames)           
        else:
            ref_tr_vidnames = te_vidnames
            ref_te_vidnames = tr_vidnames
          
        tr_kernel, te_kernel = data_to_kernels(tr_data, te_data)
        yield tr_kernel, te_kernel


def get_labels():
    selection = sys.argv[1:]
    for cname in selection:
        feature, params = combinations[cname]
        _, tr_labels, _ = get_data(feature, 'train_balanced', **params)
        _, te_labels, _ = get_data(feature, 'test_balanced', **params)
        return tr_labels, te_labels


def per_class_worker(result_queue, class_idx):
    # Load data.
    tr_kernels, te_kernels = get_kernels()
    tr_labels, te_labels = get_labels()

    # Binarize labels + slice kernels.
    cls_tr_kernels, cls_te_kernels = get_kernels_given_class(
        tr_kernels, te_kernels, tr_labels, te_labels, class_idx)
    cls_tr_labels, cls_te_labels = get_labels_given_class(
        tr_labels, te_labels, class_idx)

    # Train SVC for each channel, then late fuse.
    late_fusion = LateFusion(**late_fusion_params)
    late_fusion.fit(cls_tr_kernels, cls_tr_labels)
    score = 100 * late_fusion.score(cls_te_kernels, cls_te_labels)
    result_queue.put((class_idx, score, late_fusion.get_weights_str()))


def late_fusion_master():
    processes, result_queue = [], mp.Queue()
    for class_idx in xrange(1, 16):
        processes.append(
            mp.Process(
                target=per_class_worker,
                args=(result_queue, class_idx)))
        processes[-1].start()
    for process in processes:
        process.join()
    results = sorted([result_queue.get() for ii in xrange(15)])
    for (class_idx, score, weights) in results:
        print '%s%2.3f %s' % (
            '{0:34}'.format(idx_to_class[class_idx]), score, weights)
    print '{0:34}'.format('MAP') + '%2.3f' % np.mean(
        [score for _, score, _ in results])


late_fusion_master()

# One processor, deprecated.
#scores = []
#tr_kernels, te_kernels, tr_labels, te_labels = get_kernels()
#for class_idx in xrange(15):
#    late_fusion = LateFusion()
#    # Binarize labels + slice kernels.
#    (cls_tr_kernels, cls_te_kernels,
#     cls_tr_labels, cls_te_labels) = kernels_given_class(
#        tr_kernels, te_kernels, tr_labels, te_labels, class_idx)
#    late_fusion.fit(cls_tr_kernels, cls_tr_labels)
#    scores.append(100 * late_fusion.score(cls_te_kernels, cls_te_labels))
#    print '%d %2.3f %s' % (
#        class_idx, scores[-1], late_fusion.get_weights_str())
#print 'MAP %2.3f' % np.mean(scores)
