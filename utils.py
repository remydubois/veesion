from keras.utils import to_categorical
import numpy
import os
import tensorflow as tf
from itertools import cycle
from scipy.ndimage import imread
from data import normalizer
import pandas
from sklearn.preprocessing import LabelEncoder


def path_parser(root, indices):
    """
    Just reads the images paths from root and yields lists of paths (one path per frame), or yield the path of the numpy array of the stacked frames if exists.
    """
    for video in indices:
        if os.path.exists(os.path.join(root, str(video), 'seq.npy')):
            yield os.path.join(root, str(video), 'seq.npy')
        else:
            yield [os.path.join(root, str(video), x) for x in os.listdir(os.path.join(root, str(video))) if x.endswith('.jpg')]


def video_generator(videos, length, width=150):
    """
    Get the frames corresponding to one video  and arrange them into one stack of videos. Pads the short videos. In the same fashion, downsamples the too long videos to a fixed size.

    In v2 all the videos are converted to numpy arrays, reason why the type is checked.
    TODO could be speed up by reading the pads only once.
    """

    for video in videos:

        if type(video) == list:
            L = len(video)

            if length > L:
                video.extend([video[-1], ] * (length - L))

            indices = numpy.linspace(0, L - 1, length)
            indices = numpy.round(indices).astype(int)

            # if not os.path.exists(path)
            sequence = numpy.stack([imread(video[i]) for i in indices], axis=2)
        else:
            sequence = numpy.load(video)

            L = sequence.shape[2]

            if L < length:
                sequence = numpy.pad(sequence, pad_width=[
                                     (0, 0), (0, 0), (0, length - L), (0, 0)], mode='edge')
            if L > length:
                indices = numpy.linspace(0, L - 1, length)
                indices = numpy.round(indices).astype(int)

                sequence = sequence[:, :, indices, :]

        if sequence.shape[1] > width:
            sequence = sequence[
                :, (sequence.shape[1] - width) // 2:-(sequence.shape[1] - width) // 2, :, :]
        elif sequence.shape[1] < width:
            sequence = numpy.pad(sequence, mode='mean', pad_width=[(
                0, 0), (-(sequence.shape[1] - width) // 2, -(sequence.shape[1] - width) // 2), (0, 0), (0, 0)])
        # print(sequence.shape)
        yield sequence


def data_generator(videos, labels, length=40, batch_size=5):
    """
    Just arranges the videos into batches of equal size, calls data augmentation things and preprocessing blabla
    """

    # Get the videos:
    videogenerator = video_generator(videos, length=length)

    videogenerator = map(lambda x: x / x.max(), videogenerator)

    # Form epochs only of the same size (better for reproducibility):
    # total = (len(labels) // batch_size) * batch_size
    # labels = labels[:total]
    zipped = zip(videogenerator, labels)

    # Make it cycle
    zipped = cycle(zipped)

    # Data Augment
    # Nothing for now

    # Pre-process
    # zipped = map(lambda pair: (normalizer(pair[0]), pair[1]), zipped)

    while True:
        images, ys = list(zip(*[next(zipped) for _ in range(batch_size)]))

        yield numpy.stack(images, axis=0), to_categorical(ys, numpy.unique(labels).shape[0])


def read_labels(root):
    paths = os.listdir(os.path.join(root, 'labels'))

    paths.sort()

    le = LabelEncoder()

    # mind the order
    sets = [pandas.read_csv(os.path.join(
        root, 'labels', p), sep=';', header=None) for p in paths]
    le.fit(pandas.concat([t[1] for t in sets[1:]]))

    sets[0][2] = 'unkown'
    sets[1][2] = le.transform(sets[1][1])
    sets[2][2] = le.transform(sets[2][1])

    return sets[1], sets[2], sets[0]
