import cPickle
from ipdb import set_trace
import itertools
import multiprocessing as mp
import numpy as np
from sklearn.linear_model import Lasso
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


def weights_grid(dd, step = 0.02):
    """ Generates weights on a regular grid. """
    for ww in itertools.product(
        *(np.arange(0, 1 + step, step) for ii in xrange(dd - 1))):
        last_weight = 1 - sum(ww)
        if last_weight < 0:
            continue
        yield ww + (last_weight, )


class LateFusion(object):
    def __init__(self, score_type):
        self.score_type = score_type

    def fit(self, kernels, labels):
        self.clf = []
        kernels = list(kernels)
        nr_kernels = len(kernels)

        # Get a hold-out data for fitting the late fusion weights.
        self.weight_scores = {}
        ss = StratifiedShuffleSplit(labels, 3, test_size=0.25, random_state=0)
        #tr_idxs, val_idxs = iter(ss).next()
        # nr_samples = len(labels)
        # tr_idxs, val_idxs = np.arange(nr_samples), np.arange(nr_samples)
        for ii in xrange(nr_kernels):
            self.clf.append(SVM())

        for tr_idxs, val_idxs in ss:
            k_tr_idxs = np.ix_(tr_idxs, tr_idxs)
            k_val_idxs = np.ix_(val_idxs, tr_idxs)

            scores = []
            for ii, kernel in enumerate(kernels):
                self.clf[ii].fit(kernel[k_tr_idxs], labels[tr_idxs])
                scores.append(
                    self.predict_clf(self.clf[ii], kernel[k_val_idxs]))

            scores = np.vstack(scores).T
            self.fit_late_fusion(scores, labels[val_idxs])

        self.weights = max(self.weight_scores,
                           key=lambda key:
                           np.mean(self.weight_scores[key]))

        # Retrain on all the data.
        for ii, kernel in enumerate(kernels):
            self.clf[ii].fit(kernel, labels)

        return self

    def predict(self, te_kernels):
        scores = []
        for ii, kernel in enumerate(te_kernels):
            scores.append(self.predict_clf(self.clf[ii], kernel))
        scores = np.vstack(scores).T
        fused_scores = self.predict_late_fusion(scores)
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

    def fit_late_fusion(self, scores, tr_labels):
        # Equal weights.
        #D = scores.shape[1]
        #self.weights = np.array([1. / D] * D)

        # Dumb crossvalidation.
        best_ap = 0
        D = scores.shape[1]
        for self.weights in weights_grid(D):
            ap = average_precision(
                tr_labels, self.predict_late_fusion(scores))

            if self.weights in self.weight_scores:
                self.weight_scores[self.weights].append(ap)
            else:
                self.weight_scores[self.weights] = [ap]

        #    if ap > best_ap:
        #        best_ap = ap
        #        best_weights = self.weights
        #self.weights = best_weights

        # Fit small regressor, linear model.
        #self.lm = Lasso()
        #self.lm.fit(scores, tr_labels)

    def predict_late_fusion(self, scores):
        return np.sum(scores * self.weights, 1)
        #return self.lm.predict(scores)


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

    cls_tr_kernels, cls_te_kernels = [], []

    for tr_kernel, te_kernel in itertools.izip(tr_kernels, te_kernels):
        cls_tr_kernels.append(tr_kernel[tr_kernel_idxs])
        cls_te_kernels.append(te_kernel[te_kernel_idxs])

    return cls_tr_kernels, cls_te_kernels


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


def get_kernels_and_labels():
    selection = sys.argv[1:]

    ref_tr_vidnames = None
    ref_te_vidnames = None

    tr_kernels, te_kernels = [], []

    for cname in selection:
        feature, params = combinations[cname]

        print "load feature", cname
        
        tr_data, tr_labels, tr_vidnames = get_data(feature, 'train', **params)
        te_data, te_labels, te_vidnames = get_data(feature, 'test', **params)

        print "compute kernels train %d*%d test %d*%d" % (
            tr_data.shape + te_data.shape)

        if ref_tr_vidnames != None:
            print "remapping names"
            # pdb.set_trace()

            te_data = remap_descriptors(te_data, te_vidnames, ref_te_vidnames)
            tr_data = remap_descriptors(tr_data, tr_vidnames, ref_tr_vidnames)           
            tr_labels_, te_labels_ = tr_labels, te_labels
        else:
            ref_tr_vidnames = tr_vidnames
            ref_te_vidnames = te_vidnames
          
        tr_kernel, te_kernel = data_to_kernels(tr_data, te_data)
        tr_kernels.append(tr_kernel)
        te_kernels.append(te_kernel)

    return tr_kernels, te_kernels, tr_labels_, te_labels_


def per_class_worker(result_queue, tr_kernels, te_kernels, tr_labels,
                     te_labels, class_idx):
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

    # Load data.
    tr_kernels, te_kernels, tr_labels, te_labels = get_kernels_and_labels()
    for class_idx in xrange(1, 16):
        processes.append(
            mp.Process(
                target=per_class_worker,
                args=(result_queue, tr_kernels, te_kernels, tr_labels,
                      te_labels, class_idx)))
        processes[-1].start()
    for process in processes:
        process.join()
    results = sorted([result_queue.get() for ii in xrange(15)])
    for (class_idx, score, weights) in results:
        print '%s%2.3f %s' % (
            '{0:34}'.format(idx_to_class[class_idx]), score, weights)
    print '{0:34}'.format('MAP') + '%2.3f' % np.mean(
        [score for _, score, _ in results])


def late_fusion_test():
    qq = mp.Queue()
    tr_kernels, te_kernels, tr_labels, te_labels = get_kernels_and_labels()
    per_class_worker(qq, tr_kernels, te_kernels, tr_labels, te_labels, 1)


def main():
    #late_fusion_test()
    late_fusion_master()


if __name__ == '__main__':
    main()
