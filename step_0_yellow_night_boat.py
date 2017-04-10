"""
Finding BoatID according to blog post:
https://www.kaggle.com/anokas/the-nature-conservancy-fisheries-monitoring/finding-boatids/notebook
"""
import pandas as pd
import numpy as np
import glob
from sklearn import cluster
import cPickle
from sklearn import mixture
import cv2
import random
import matplotlib.pyplot as plt

OPENCV_METHODS = (
	("Correlation", cv2.cv.CV_COMP_CORREL),
	("Chi-Squared", cv2.cv.CV_COMP_CHISQR),
	("Intersection", cv2.cv.CV_COMP_INTERSECT),
	("Hellinger", cv2.cv.CV_COMP_BHATTACHARYYA))

# Function to show 4 images
def show_four(imgs, title):
    #select_imgs = [np.random.choice(imgs) for _ in range(4)]
    select_imgs = [imgs[np.random.choice(len(imgs))] for _ in range(4)]
    _, ax = plt.subplots(1, 4, sharex='col', sharey='row', figsize=(20, 3))
    plt.suptitle(title, size=20)
    for i, img in enumerate(select_imgs):
        ax[i].imshow(img)


def show_16(imgs, title):
    _, ax = plt.subplots(4, 4, sharex='col', sharey='row', figsize=(20, 6))
    plt.suptitle(title, size=20)
    for i, img in enumerate(imgs):
        ax[i // 4, i % 4].imshow(img)


def main():

    # Data loading
    train_files = sorted(glob.glob('../train/*/*.jpg'), key=lambda x: random.random())
    train = np.array([cv2.imread(img) for img in train_files])
    print('Length of train {}'.format(len(train)))
    shapes = np.array([str(img.shape) for img in train])
    pd.Series(shapes).value_counts()
    show_16(train[:16], 'first_16')

    print('calculating image histogram...')
    hist_all = np.zeros((len(train), 8*8*8))
    for i in range(len(train)):
        hist_all[i] = cv2.normalize(cv2.calcHist([train[i]], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])).flatten()
        #hist_all[i] = cv2.calcHist([train[i]], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]).flatten()

    color_mean = np.zeros((len(train), 3))
    for i in range(len(train)):
        color_mean[i]= train[i].mean(axis=0).mean(axis=0)
    ## gmm
    print('GMM modelling')
    gmix_color_mean = mixture.GMM(n_components=2, covariance_type='full')
    gmix_color_mean.fit(color_mean)
    label_color_mean = gmix_color_mean.predict(color_mean)
    with open('./exp_dir/night_boat_classifier.pkl', 'wb') as fid:
        cPickle.dump(gmix_color_mean, fid)

    with open('./exp_dir/night_boat_classifier.pkl', 'rb') as fid:
        gmix = cPickle.load(fid)
    label = gmix.predict(hist_all)
    prob = gmix.predict_proba(hist_all)
    print('TSNE dimensionality reduction')
    from classes.TSNE import tsne
    Y = tsne(hist_all, 2, 50, 20.0)

    tsne_visualisation(train, Y[:,::-1])
    ## gmm
    print('GMM modelling')
    gmix = mixture.GMM(n_components=2, covariance_type='full')
    gmix.fit(hist_all)

    with open('./exp_dir/night_boat_classifier.pkl', 'wb') as fid:
        cPickle.dump(gmix, fid)

    print(gmix.means_)
    label = gmix.predict(hist_all)
    prob = gmix.predict_proba(hist_all)
    idx = np.argsort(prob[:,0])
    show_16(train[idx[-32:-16]], 'first_16')

    nightBoatImages = train[label==0]

    colors = ['r' if i == 0 else 'g' for i in gmix.predict(hist_all)]
    plt.figure()
    ax = plt.gca()
    ax.scatter(Y[:, 0], Y[:, 1], c=colors, alpha=0.8)
    plt.show()

    ## kmeans clustering
    # from sklearn.cluster import  KMeans
    # estimator = KMeans(n_clusters=2)
    # estimator.fit(hist_all)
    # labels = estimator.labels_
    # colors = ['r' if i == 0 else 'g' for i in labels]
    # plt.figure()
    # ax = plt.gca()
    # ax.scatter(Y[:, 0], Y[:, 1], c=colors, alpha=0.8)
    # plt.show()
    #
    # # spectral clustering
    # distances = np.zeros((len(hist_all), len(hist_all)))
    # for i in range(len(hist_all)):
    #     for j in range(i+1, len(hist_all)):
    #         distances[i, j] = cv2.compareHist(hist_all[i].astype('float32'), hist_all[j].astype('float32'), cv2.cv.CV_COMP_BHATTACHARYYA)
    #
    # distances = 0.5 * (distances + distances.T)
    # from sklearn import cluster
    # ward = cluster.AgglomerativeClustering(n_clusters=2, linkage='ward',
    #                                        connectivity=distances)
    #
    # spectral = cluster.SpectralClustering(n_clusters=2, affinity='precomputed')
    # spectral.fit(distances)
    # y_pred = spectral.labels_.astype(np.int)


def tsne_visualisation(X, Y, total_sz=(4000, 4000), resize_sz=(400,400), min_dist=0.01):
    import scipy

    plt.title("\\textbf{TSNE} -- Two-dimensional embedding of boatID")
    im_all = np.ones((total_sz[0], total_sz[1], 3)) * 255
    im_record = np.zeros(total_sz)
    gap = total_sz[0] * min_dist

    im_range_min = Y.min(axis=0)
    im_range_max = Y.max(axis=0)
    im_range = im_range_max - im_range_min

    for i in range(X.shape[0]):
        loc_norm = ((Y[i][0]-im_range_min[0])/im_range[0],  (Y[i][1]-im_range_min[1])/im_range[1])
        loc_norm *= np.asarray(total_sz)
        loc_norm = max(resize_sz[0]/2, min(int(loc_norm[0]), total_sz[0]-resize_sz[0]/2)), \
                   max(resize_sz[1]/2, min(int(loc_norm[1]), total_sz[1]-resize_sz[1]/2))

        loc_record = max(0, min(int(loc_norm[0]), total_sz[0])), \
                   max(0, min(int(loc_norm[1]), total_sz[1]))
        im_record[loc_record[0], loc_record[1]] = 1
        if np.sum(im_record[loc_record[0]-gap:loc_record[0]+gap, loc_record[1]-gap:loc_record[1]+gap])<2:

            im_resized = scipy.misc.imresize(X[i], resize_sz)
            im_all[loc_norm[0]-resize_sz[0]/2: loc_norm[0]+resize_sz[0]/2,
                loc_norm[1]-resize_sz[1]/2: loc_norm[1]+resize_sz[1]/2] = im_resized

    plt.imshow(im_all/255.)


if __name__ == '__main__':
    main()