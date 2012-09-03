""" Use SIFT descriptors for the videos. """
import cPickle
from ipdb import set_trace
import numpy as np
import os
import pdb
import random
from sklearn import cross_validation
from sklearn.preprocessing import Scaler
import sys

import descriptors
from yael.yael import gmm_read
from yael import ynumpy

from fisher_vectors.evaluation import Evaluation
from fisher_vectors.evaluation.trecvid12_parallel import chunker
from fisher_vectors.per_slice.discriminative_detection import aggregate
from fisher_vectors.per_slice.discriminative_detection import _normalize
from fisher_vectors.model import FVModel
from fisher_vectors.model.utils import standardize
from fisher_vectors.model.utils import power_normalize
from fisher_vectors.model.utils import L2_normalize


all_jochen_cache = {}
label_map = {
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
    31: 'unknown',
}
rev_label_map = {name:no for no, name in label_map.items()}


combinations = {
    'sift_mu':       ('sift2', {'dimensions' : (255, 255 + 32 * 256) }),
    'sift_sigma':    ('sift2', {'dimensions' : (255 + 32 * 256, 255 + 32 * 256 * 2) }),
    'sift_mu_sigma': ('sift2', {'dimensions' : (255, 255 + 32 * 256 * 2) }),
    'sift_mu_sigma_T2t1': ('sift2', {'dimensions' : (255, 255 + 32 * 256 * 2), 'temporal_spm': (2, 0) }),
    'sift_mu_sigma_T2t2': ('sift2', {'dimensions' : (255, 255 + 32 * 256 * 2), 'temporal_spm': (2, 1) }),
    'mbh':           ('mbh', {}),
    'mbh_2xnorm':    ('mbh', {'double_norm': True}),
    'mbh_1024':      ('mbh', {'K': 1024}),
    'audio':         ('heng_audio', {'derivative': ''}),
    'audio_D1':      ('heng_audio', {'derivative': '_D1'}),
    'audio_D2':      ('heng_audio', {'derivative': '_D2'}),
    'jochen_audio':  ('jochen_audio',{'dimslice': 0}),
    }
 

def normalize_fisher(X):
    """ Power normalization """
    X = np.sign(X) * np.sqrt(np.abs(X))
    # L2     
    X = X / np.sqrt((X * X).sum(axis=1).reshape((X.shape[0],1)))
    X[np.isnan(X)] = 0
    return X


def get_per_video_sift_data_old(split, subsample=1, nr_clusters=64, color=10):
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


def get_per_video_sift_data(split, desc_key = 'framestep60_dense_siftnonorm_lpca32_gmm256_w_mu_sigma',
                            dimensions = (255, 255 + 32 * 256), **kwargs):
    """
    The descriptor key is a directory name. The dimensions can be used to select mu/w/sigma
    """
    temporal_spm = kwargs.get('temporal_spm', None)
    if temporal_spm:
        suffix = '_T%dt%d' % temporal_spm
        nr_bags, bag_idx = temporal_spm
    else:
        suffix = ''
        nr_bags, bag_idx = 1, 0

    # cache
    filename = ("/home/lear/oneata/data/trecvid12/scripts/"
                "fusion/data/per_video_cache/SIFT_%s_%s%s.raw" %
                (desc_key, split, suffix))

    if os.path.exists(filename):
        # Load data from cache file.
        print "Load per video data", filename
        with open(filename, "r") as ff:
            video_data = np.load(ff)
            video_labels = np.load(ff)
            video_names = cPickle.load(ff)
        d0, d1 = dimensions
        return video_data[:, d0:d1], video_labels, video_names

    basedir = "/scratch2/clear/dataset/trecvid11/med//desc/frames/" + desc_key
    name_map = {
        fname[:9] : d0 + '/' + d1
        for d0 in ['training', 'validation']
        for d1 in ['NULL'] + ['E%03d' % no for no in range(1, 16)]
        for fname in os.listdir(basedir + '/' + d0 + '/' + d1)
        if fname.endswith('_0.fvecs')}      

    video_names = []
    video_labels = []
    video_data = []

    for l in open("data/%s.list" % split, "r"):
        l = l[:-1]
        vidname = l.split('-')[0]  
        classname = l.split(' ')[-1]
        video_names.append(vidname)
        video_labels.append(rev_label_map[classname])

        # Fisher descriptors of a video may be spread over several blocks
            
        block = 0
        fv = None
        while True:
            fname = "%s/%s/%s_%d.fvecs" % (basedir, name_map[vidname], vidname, block)
            if os.access(fname, os.R_OK):
                fvi = ynumpy.fvecs_read(fname)
                if fvi.shape[0] == 0:
                    pass
                elif fv == None:
                    fv = fvi
                else:
                    try:
                        fv = np.vstack((fv, fvi))
                    except ValueError:
                        pdb.set_trace()
            else:
                break
            block += 1 
        assert block > 0
        print vidname, "%d * %d" % fv.shape     

        fv = normalize_fisher(fv)

        # average of descriptors        
        nr_slices = fv.shape[0]
        bag_idx = np.minimum(bag_idx, nr_slices - 1)
        idxs = list(chunker(np.arange(nr_slices), nr_bags))[bag_idx]
        desc = fv[idxs].sum(axis = 0) / fv[idxs].shape[0]
        video_data.append(desc)
    
    video_data = np.vstack(video_data)
    video_labels = np.array(video_labels)
    
    with open(filename, "w") as ff:
        np.save(ff, video_data)
        np.save(ff, video_labels)
        cPickle.dump(video_names, ff)
    d0, d1 = dimensions
    return video_data[:, d0:d1], video_labels, video_names


def all_jochen(dimslice):
    if dimslice in all_jochen_cache:
        return all_jochen_cache[dimslice]

    data = []
    video_names = []
    for split2 in 'training', 'validation':
        datadir = (
            "/scratch/clear/douze/data_tv12/jochen_audio_descs/TV_MED_2011/" +
            split2)
        
        for eno in range(0, 16):
            ename = "E%03d" % eno if eno > 0 else "NULL"
            fname = datadir + "/fisher_vectors_%s.fvecs" % ename
            fname_txt = datadir + "/%s_%s.txt" % (split2, ename)
            print "load", fname
            descs = ynumpy.fvecs_read(fname)
            descs = descs[:, dimslice * 5055 : (dimslice + 1) * 5055]       
            ntxt = 0
            for l in open(fname_txt, "r"):
                vidname = l.split('/')[-1].split('.')[0]
                video_names.append(vidname)
                ntxt += 1
            if (split2, eno) != ('training', 0): 
                if ntxt != descs.shape[0]:
                    print "ntxt=%d shape=%d" % (ntxt, descs.shape[0])
                    video_names = video_names[:descs.shape[0] - ntxt]
            else:
                # train/fisher_vectors_NULL.fvecs just a link....
                descs = descs[:ntxt, :]
            data.append(descs)
    data = np.vstack(data)
            
    all_jochen_cache[dimslice] = video_names, data

    return video_names, data


def get_audio_data_jochen(split, dimslice = 0):
    " read Jochen's audio descriptors"

    descvids, descs = all_jochen(dimslice)

    descvids = {name:no for no, name in enumerate(descvids)}

    missing = '0'
    d = descs.shape[1]

    video_names = []
    labels = []
    data = []
    miss = []
    n_wanted = 0
    for l in open("data/%s_balanced.list" % split, "r"):
        l = l[:-1]
        vidname = l.split('-')[0]  
        classname = l.split(' ')[-1]
        n_wanted += 1
        
        if vidname in descvids:     
            video_names.append(vidname)
            labels.append(rev_label_map[classname])
            data.append(descs[descvids[vidname]])
        else:
            miss.append(vidname)
            if missing == 'skip':
                pass
            else:
                video_names.append(vidname)
                labels.append(rev_label_map[classname])
                data.append(np.zeros(d))
                                
    print "  missing %d/%d descriptors" % (len(miss), n_wanted)
    
    data = np.vstack(data)
    labels = np.array(labels)
    return data, labels, video_names

    
def get_audio_data_heng(split, derivative = '', missing = '0'):
    " read Heng's audio descriptors (in txt format....)"
    
    cachefname = "data/audio_cache/mix%s.raw" % derivative

    if os.access(cachefname, os.R_OK):
        print "using cache", cachefname
        ff = open(cachefname, "r")
        descvids = cPickle.load(ff)        
        descs = np.load(ff)
    else: 
        
        filename = "/home/clear/hewang/trecvid2012/trecvid11/result_MFCC%s_final/fisher/features_k256.gz" % derivative
        print "loading", filename
        descs = []
        descvids = []        
        for l in os.popen("gunzip -c %s " % filename, "r"):
            if l == '\n':
                pass            
            elif l[0] == '#':
                vidname = l[2:-1]
            else:
                descvids.append(vidname)                
                descs.append(np.fromstring(l, sep = ' '))
                if len(descs) % 1000 == 0:
                    print "%d descs..." % len(descs)

        descs = np.vstack(descs)

        print "writing cache", cachefname
        ff = open(cachefname, "w")
        cPickle.dump(descvids, ff)        
        np.save(ff, descs)        
        
    print "  select descriptors for", split

    descvids = {name:no for no, name in enumerate(descvids)}

    # find dimension from an arbitrary descriptor
    d = descs.shape[1]

    video_names = []
    labels = []
    data = []
    miss = []
    n_wanted = 0
    for l in open("data/%s.list" % split, "r"):
        l = l[:-1]
        vidname = l.split('-')[0]  
        classname = l.split(' ')[-1]
        n_wanted += 1
        
        if vidname in descvids:     
            video_names.append(vidname)
            labels.append(rev_label_map[classname])
            data.append(descs[descvids[vidname]])
        else:
            miss.append(vidname)
            if missing == 'skip':
                pass
            else:
                video_names.append(vidname)
                labels.append(rev_label_map[classname])
                data.append(np.zeros(d))
                                
    print "  missing %d/%d descriptors" % (len(miss), n_wanted)
    
    data = np.vstack(data)
    labels = np.array(labels)
    return data, labels, video_names


def get_per_video_mbh_data_given_list(list_name, **kwargs):
    """ Loads the Fisher vectors corresponding to the samples found in the
    list specified by `list_name`.

    """
    K = kwargs.get('K', 256)
    sstats_path = ('/home/clear/oneata/data/trecvid12/features'
                   '/dense5.track15mbh.small.skip_1/statistics_k_%d' % K)

    # Default base directories.
    list_base_path = kwargs.get(
        'list_base_path', '/home/lear/douze/src/experiments/trecvid12/data')
    cache_base_path = kwargs.get(
        'cache_base_path', sstats_path)
    double_norm = kwargs.get('double_norm', False)

    suffix = '.double_norm' if double_norm else ''

    # If this file exists load them directly.
    cache_filename = os.path.join(cache_base_path, 'mbh_' + list_name + suffix + '.raw')
    if os.path.exists(cache_filename):
        print "Loading Fisher vectors from MBH descriptors for list %s..." % (
            list_name)
        with open(cache_filename, 'r') as ff:
            data = np.load(ff)
            labels = np.load(ff)
            names = cPickle.load(ff)
        return data, labels, names

    D = 64
    data, labels, names = [], [], []
    sstats_generic_name = os.path.join(sstats_path, 'stats.tmp' + suffix, '%s.dat')
    list_path = os.path.join(list_base_path, list_name + '.list')
    gmm_path = os.path.join(sstats_path, 'gmm_%d' % K)
    gmm = gmm_read(open(gmm_path, 'r'))

    # Get descriptors for the files in list.
    print "Creating cache file from list %s..." % list_name
    for line in open(list_path, 'r'):
        fields = line.split()

        sstats_filename = fields[0]
        video_name = sstats_filename.split('-')[0]
        sys.stdout.write("%s\t\r" % video_name)

        try:
            class_name = fields[1]
        except IndexError:
            class_name = 'unknown'

        try:
            sstats = np.fromfile(sstats_generic_name % sstats_filename,
                                 dtype=np.float32)
            if double_norm:
                fv = sstats
            else:
                fv = FVModel.sstats_to_features(sstats, gmm)
        except IOError:
            print ('Sufficient statistics for video %s are missing;'
                   'replacing with zeros.') % video_name
            fv = np.zeros(K + 2 * K * D)

        data.append(fv)
        names.append(video_name)
        labels.append(rev_label_map[class_name])

    data = np.vstack(data)
    labels = np.array(labels)

    assert data.shape[0] == len(labels), "Data size doesn't match nr of labels"

    # Cache results to file.
    with open(cache_filename, 'w') as ff:
        np.save(ff, data)
        np.save(ff, labels)
        cPickle.dump(names, ff)

    return data, labels, names


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


def get_dummy_data(split):
    D = 200
    N = 1050 if split == 'train' else 1000
    data = np.random.rand(N, D)
    labels = np.random.randint(0, 16, N)
    return data, labels


def get_data(features, split, **kwargs):
    """ Loads data for the specified features. """
    if features == 'sift' or features == 'color':
        subsample = kwargs.get('subsample')
        nr_clusters = kwargs.get('nr_clusters')
        color = kwargs.get('color')
        data, labels, vidnames  = get_per_video_sift_data_old(
            split, subsample, nr_clusters, color)               
    elif features == 'sift2':
        data, labels, vidnames = get_per_video_sift_data(split, **kwargs)
    elif features == 'mbh':
        data, labels, vidnames = get_per_video_mbh_data_given_list(split, **kwargs)        
    elif features == 'jochen_audio':
        data, labels, vidnames = get_audio_data_jochen(split, **kwargs)
    elif features == 'heng_audio':
        data, labels, vidnames = get_audio_data_heng(split, **kwargs)
    else:
        assert False
    # pdb.set_trace()
    return data, labels, vidnames


def data_to_kernels(tr_data, te_data):
    scaler = Scaler(copy=False)
    scaler.fit_transform(tr_data)
    #tr_data, mu, sigma = standardize(tr_data)
    tr_data = power_normalize(tr_data, 0.5)
    tr_data = L2_normalize(tr_data)

    #te_data, _, _ = standardize(te_data, mu, sigma)
    scaler.transform(te_data)
    te_data = power_normalize(te_data, 0.5)
    te_data = L2_normalize(te_data)

    tr_kernel = np.dot(tr_data, tr_data.T)
    te_kernel = np.dot(te_data, tr_data.T)

    return tr_kernel, te_kernel


def remap_descriptors(te_data, te_vidnames, ref_te_vidnames):
    """ reorder or select subset from te_data where each line corresponds to an entry in te_vidnames,
    so that the lines in the order given by ref_te_vidnames"""
    perm = []
    rev_te_vidnames = {name: no for no, name in enumerate(te_vidnames)}
    try:
        perm = [rev_te_vidnames[name] for name in ref_te_vidnames]
    except KeyError:
        pdb.set_trace()
    perm = np.array(perm)
    return te_data[perm, :]            


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


def vary_nr_negatives():
    null_class_idx = 0
    feature = 'mbh'
    params = {
        'dummy': {},
        'mbh': {'suffix': '_morenull'},
        'sift': {'subsample': 10, 'nr_clusters': 64, 'color': 0}}

    tr_data, tr_labels, _ = get_data(feature, 'train', **params[feature])
    te_data, te_labels, _ = get_data(feature, 'test', **params[feature])

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

                tr_kernel, te_kernel = data_to_kernels(tr_data, te_data)
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
    selection = sys.argv[1:]

    ref_tr_vidnames = None
    ref_te_vidnames = None

    for cname in selection:

        factor = 1.0
        if '*' in cname:
            factor, cname = cname.split('*')
            factor = float(factor)
        
        feature, params = combinations[cname]

        print "load feature", cname, "factor", factor
        
        tr_data, tr_labels, tr_vidnames = get_data(feature, 'train_balanced', **params)
        te_data, te_labels, te_vidnames = get_data(feature, 'test_balanced', **params)

        print "compute kernels train %d*%d test %d*%d" % (
            tr_data.shape + te_data.shape)

        if ref_tr_vidnames != None:
            print "remapping names"
            # pdb.set_trace()

            te_data = remap_descriptors(te_data, te_vidnames, ref_te_vidnames)
            tr_data = remap_descriptors(tr_data, tr_vidnames, ref_tr_vidnames)           
          
        
        Kxx, Kyx = data_to_kernels(tr_data, te_data)

        Kxx *= factor
        Kyx *= factor

        if ref_tr_vidnames == None:
            tr_kernel = Kxx
            te_kernel = Kyx
            ref_te_vidnames = te_vidnames
            ref_tr_vidnames = tr_vidnames
        else:
            tr_kernel += Kxx
            te_kernel += Kyx        

    from fisher_vectors.evaluation import trecvid12_parallel as eval
    fit_out = eval.fit(tr_kernel, tr_labels)
    print eval.score(te_kernel, te_labels, fit_out)


def main():
    #vary_nr_negatives()
    evaluate()


if __name__ == '__main__':
    main()
