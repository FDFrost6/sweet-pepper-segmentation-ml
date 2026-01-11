import argparse
import pickle
import numpy as np
import os
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from skimage.feature import hog
from skimage.io import imread
from skimage.color import rgb2gray
from scipy.stats import mode            
from sklearn.metrics import confusion_matrix

from libs.clustering import KMeans
# from libs.utils import mustdnorm



parser = argparse.ArgumentParser(add_help=True)
parser.add_argument("--exr", required=True)
parser.add_argument("--dataloc", action="store", default="data/texture_snippets/")
parser.add_argument("--ppc", action="store", type=int, default=8, help="Pixels per cell")
parser.add_argument("--orient", type=int, default=8, help="Number of orientations")
parser.add_argument("--cpb", type=int, default=1, help="Cells per block")
parser.add_argument("--nclusters", type=int, default=5)
parser.add_argument( '--classes', '-c', nargs="+", default=[] )
parser.add_argument( '--knn', action='store', type=int, default=100 )
flags = parser.parse_args()



if flags.exr=='exr0':
    
    def extract_train_eval_files(dataloc, testsplit):
        files_train = []
        files_test = []
        labels_train = []
        labels_test = []
        for i, tex in enumerate(sorted(os.listdir(dataloc))):
            files_tex = glob.glob(os.path.join(dataloc, tex, '*.jpg'))
            ## randomization can be disabled with random state here, makes it easier to track what's happening.
            ## if desired, remember to also do so in your KMeans class (call np.random.seed(XXX) before your .randint() call)
            files_tex_train, files_tex_test = train_test_split(files_tex, test_size=.3)#, random_state=20250603)
            files_train.extend(files_tex_train)
            files_test.extend(files_tex_test)
            labels_train.extend([i]*len(files_tex_train))
            labels_test.extend([i]*len(files_tex_test))
        return files_train, files_test, labels_train, labels_test
    
    def extract_hog_feats(file, orient, ppc, cpb):
        gray = rgb2gray(imread(file))
        # print(gray.shape)
        feat, hog_map = hog(
            gray,
            orientations=orient,
            pixels_per_cell=(ppc, ppc),
            cells_per_block=(cpb, cpb),
            visualize=True,
            feature_vector=True,
        )
        # print(feat)
        # print(feat.shape)
        feat = feat.reshape(-1, orient)
        # print(feat.shape)
        # exit()
        return feat

    def extract_train_hog_feature_vector(files, labels, orient, ppc, cpb):        
        feat_out = []
        labels_out = []
        for f, l in tqdm(zip(files, labels), total=len(files)):
            feat = extract_hog_feats(f, orient, ppc, cpb)
            feat_out.append(feat)
            labels_out.extend([l]*feat.shape[0])
        feat_out = np.vstack(feat_out)
        labels_out_ = np.hstack(labels_out)
        labels_out = np.stack(labels_out)  # my original
        return feat_out, labels_out
    
    class BoVW:
        def __init__(self, nclusters):
            self.nclusters = nclusters
        def fit(self, X):
            # self.km = KMeans(n_clusters=self.nclusters, imax=5)  # lowered for fatser prototyping
            self.km = KMeans(n_clusters=self.nclusters)  # play around with how many k you need. This is your number of visual words, an aimportant prameter!
            self.km.fit(X)
        def predict(self, X):
            feats = self.km.predict(X)
            # print(feats)
            counts, _ = np.histogram( feats, bins=self.nclusters )
            return counts

    files_train, files_test, labels_train, labels_test = extract_train_eval_files(dataloc=flags.dataloc, testsplit=.3)
    ## deactivate to load from disk and skip during prototyping
    if False:
        feats_train, flattened_labels_train = extract_train_hog_feature_vector(files_train, labels_train, flags.orient, flags.ppc, flags.cpb)
        feats_test, flattened_labels_test = extract_train_hog_feature_vector(files_test, labels_test, flags.orient, flags.ppc, flags.cpb)
        with open('data/practical07exc0.pkl', 'wb') as f:
            pickle.dump((feats_train, flattened_labels_train, feats_test, flattened_labels_test), f)
    else:
        with open('data/practical07exc0.pkl', 'rb') as f:
            feats_train, flattened_labels_train, feats_test, flattened_labels_test = pickle.load(f)
    print(feats_train.shape, flattened_labels_train.shape)
    print(feats_test.shape, flattened_labels_test.shape)

    bovw = BoVW(nclusters=flags.nclusters)
    bovw.fit(feats_train)
    # print(bovw.km.C)
    # print(bovw.km.C.shape)
    bovw.predict(feats_test)
    
    ## sample hist
    sample = files_test[0]
    # sample = files_test[99]  # from different class
    sample_feats = extract_hog_feats(sample, flags.orient, flags.ppc, flags.cpb)
    # print(sample_feats.shape)
    counts = bovw.predict(sample_feats)
    # print(counts)
    # print(counts.shape)
    plt.stairs(counts, fill=True)
    plt.show()
    plt.close()
    
    ## multiple:
    n_samples = 20
    unique_labels = np.unique(labels_test)
    textures = sorted(os.listdir(flags.dataloc))
    fig, ax = plt.subplots( n_samples, len(unique_labels), sharex=True, sharey=True )
    column = 0
    for u in unique_labels:
        idx = list(labels_test).index(u)  # get first occurance of that label
        files_test_u = files_test[idx:idx+n_samples]  # get files for that label
        row = 0
        for sample in files_test_u:
            sample_feats = extract_hog_feats(sample, flags.orient, flags.ppc, flags.cpb)
            counts = bovw.predict( sample_feats )
            ax[row,column].stairs(counts, fill=True)
            if row==0:
                ax[row,column].set_title(f'{column}_{textures[column]}')
            row += 1
            if row == n_samples:
                column += 1
                break
    plt.tight_layout()
    plt.show()
    plt.close()
    


if flags.exr=='exr1':
    
    if 'all' in flags.classes:
        flags.classes = ['black', 'blue', 'green', 'grey', 'red', 'white']
    print( flags.classes )

    def mustdnorm( image, mu=0.5, st=0.5 ):
        return (image-mu)/st

    first = True
    cat = 0
    print(sorted(os.listdir( flags.dataloc )))
    for i, c in enumerate(sorted(os.listdir(flags.dataloc))):
        if c in flags.classes:
            for f in sorted( glob.glob( os.path.join( flags.dataloc, c, '*.png' ) ) ):
                rgb = imread( f )/255.
                rgb = mustdnorm( rgb )
                # print(rgb.shape)
                rgb = rgb.reshape( -1, 3 )
                # print(rgb.shape)
                # exit()
                if first:
                    V = rgb
                    l = np.ones( (rgb.shape[0],) )*cat
                    first = False
                else:
                    V = np.vstack( (V,rgb) )
                    l = np.concatenate( (l, np.ones( (rgb.shape[0], ) )*cat) )
            cat+=1
        # print(V.shape, l.shape)

    class KNN():
        def __init__(self, K=1 ):
            self.K = K

        def fit(self, X, y ):
            self.X = X
            self.y = y

        def euclid(self, x ):
            resids = self.X - x
            sqrd = resids**2
            smmd = np.sum( sqrd, axis=1 )
            return np.sqrt( smmd )

        def predict(self, X ):
            cls = np.zeros( (X.shape[0],) )
            for i, x in enumerate( X ):
                dist = self.euclid( x )
                ids = dist.argsort()[:self.K]
                nn = self.y[ids]
                cls[i], _ = mode( nn )
            return cls

    Xt, Xe, yt, ye = train_test_split( V, l, test_size=.3 )
    obj = KNN( K=flags.knn )
    obj.fit( Xt, yt )
    preds = obj.predict( Xe )
    # print(preds)
    # print(preds.shape)
    cm = confusion_matrix( ye, preds, normalize='true' )
    cm = confusion_matrix( ye, preds )
    print( cm )
