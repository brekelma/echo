import numpy as np
import six
from six.moves.urllib.request import urlretrieve
from abc import ABC, abstractmethod
from scipy.misc import imread
from scipy.io import loadmat
import os 
import gzip
import pickle
import copy 

def get_file(fname,
             origin,
             untar=False,
             md5_hash=None,
             file_hash=None,
             cache_subdir='datasets',
             hash_algorithm='auto',
             extract=False,
             archive_format='auto',
             cache_dir=None):
    """Downloads a file from a URL if it not already in the cache.
    By default the file at the url `origin` is downloaded to the
    cache_dir `~/.keras`, placed in the cache_subdir `datasets`,
    and given the filename `fname`. The final location of a file
    `example.txt` would therefore be `~/.keras/datasets/example.txt`.
    Files in tar, tar.gz, tar.bz, and zip formats can also be extracted.
    Passing a hash will verify the file after download. The command line
    programs `shasum` and `sha256sum` can compute the hash.
    # Arguments
        fname: Name of the file. If an absolute path `/path/to/file.txt` is
            specified the file will be saved at that location.
        origin: Original URL of the file.
        untar: Deprecated in favor of 'extract'.
            boolean, whether the file should be decompressed
        md5_hash: Deprecated in favor of 'file_hash'.
            md5 hash of the file for verification
        file_hash: The expected hash string of the file after download.
            The sha256 and md5 hash algorithms are both supported.
        cache_subdir: Subdirectory under the Keras cache dir where the file is
            saved. If an absolute path `/path/to/folder` is
            specified the file will be saved at that location.
        hash_algorithm: Select the hash algorithm to verify the file.
            options are 'md5', 'sha256', and 'auto'.
            The default 'auto' detects the hash algorithm in use.
        extract: True tries extracting the file as an Archive, like tar or zip.
        archive_format: Archive format to try for extracting the file.
            Options are 'auto', 'tar', 'zip', and None.
            'tar' includes tar, tar.gz, and tar.bz files.
            The default 'auto' is ['tar', 'zip'].
            None or an empty list will return no matches found.
        cache_dir: Location to store cached files, when None it
            defaults to the [Keras Directory](/faq/#where-is-the-keras-configuration-filed-stored).
    # Returns
        Path to the downloaded file
    """  # noqa
    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser('~'), '.keras')
    if md5_hash is not None and file_hash is None:
        file_hash = md5_hash
        hash_algorithm = 'md5'
    datadir_base = os.path.expanduser(cache_dir)
    if not os.access(datadir_base, os.W_OK):
        datadir_base = os.path.join('/tmp', '.keras')
    datadir = os.path.join(datadir_base, cache_subdir)
    if not os.path.exists(datadir):
        os.makedirs(datadir)

    if untar:
        untar_fpath = os.path.join(datadir, fname)
        fpath = untar_fpath + '.tar.gz'
    else:
        fpath = os.path.join(datadir, fname)

    download = False
    if os.path.exists(fpath):
        pass
        # File found; verify integrity if a hash was provided.
        #if file_hash is not None:
            #if not validate_file(fpath, file_hash, algorithm=hash_algorithm):
            #    print('A local file was found, but it seems to be '
            #          'incomplete or outdated because the ' + hash_algorithm +
            #          ' file hash does not match the original value of ' +
            #          file_hash + ' so we will re-download the data.')
            #    download = True
    else:
        download = True

    if download:
        print('Downloading data from', origin)

        error_msg = 'URL fetch failure on {}: {} -- {}'
        try:
            try:
                urlretrieve(origin, fpath)# dl_progress)
            except HTTPError as e:
                raise Exception(error_msg.format(origin, e.code, e.msg))
            except URLError as e:
                raise Exception(error_msg.format(origin, e.errno, e.reason))
        except (Exception, KeyboardInterrupt):
            if os.path.exists(fpath):
                os.remove(fpath)
            raise


    if untar:
        if not os.path.exists(untar_fpath):
            _extract_archive(fpath, datadir, archive_format='tar')
        return untar_fpath

    if extract:
        _extract_archive(fpath, datadir, archive_format)

    return fpath

def mnist_load(path='mnist.npz'):
    """Loads the MNIST dataset.
    # Arguments
        path: path where to cache the dataset locally
            (relative to ~/.keras/datasets).
    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    path = get_file(path,
                    origin='https://s3.amazonaws.com/img-datasets/mnist.npz',
                    file_hash='8a61469f7ea1b51cbae51d4f78837e45')
    f = np.load(path)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    return (x_train, y_train), (x_test, y_test)

class Dataset(ABC): 
    #def __init__(self, per_label = None):
    #    if per_label is not None
        
    @abstractmethod
    def get_data(self):
        pass

    def shrink_supervised(self, per_label, include_labels = None, shuffle = True):
        data = None
        labels = None
        per_label = int(per_label)
        for i in np.unique(self.y_train[:self.x_train.shape[0]]):
            if include_labels is None or (include_labels is not None and i in include_labels):
                l = np.squeeze(np.array(np.where(self.y_train[:self.x_train.shape[0]] == i)))
                if shuffle:
                    np.random.shuffle(l)
                if data is not None:
                    data = np.vstack((data, self.x_train[l[:per_label], :]))
                    labels = np.hstack((labels, self.y_train[l[:per_label]]))
                else:
                    data = self.x_train[l[:per_label], :]
                    labels = self.y_train[l[:per_label]] 
        self.x_train = data
        self.y_train = labels

    def generator(self, data = None, test = None, target_len = 1, unsupervised = True, batch = 100, mode = 'train', echo = False):
        if data is None:
            data = self.x_train if mode == 'train' else self.x_test
            labels = self.y_train if mode == 'train' else self.y_test
            #data = copy.copy(self.x_train) if mode == 'train' else copy.copy(self.x_test)
            #labels = copy.copy(self.y_train) if mode == 'train' else copy.copy(self.y_test)

        samples_per_epoch = data.shape[0]
        number_of_batches = int(samples_per_epoch/batch)
        counter=0

        while 1:
            x_batch = np.array(data[batch*counter:batch*(counter+1)]).astype('float32')
            y_batch = x_batch if unsupervised else np.array(labels[batch*counter:batch*(counter+1)]).astype('float32') 
            if echo:
                pass
                # TO DO : efficient way of sampling other data points?  would have to do forward pass
                
            counter += 1
            yield x_batch, [y_batch]*target_len

            #restart counter to yeild data in the next epoch as well
            if counter >= number_of_batches:
                counter = 0
                np.random.shuffle(data)
                np.random.shuffle(labels)

    #def num_samples(self):

class DatasetWrap(Dataset):
    def __init__(self, x_train, x_val = None, y_train = None, y_val = None, x_test = None, y_test = None, dim1 = None, dim2 = None):
        self.x_train = x_train
        self.x_val = x_val
        self.y_train = y_train
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test
        self.dim = x_train.shape[-1]
        if dim1 is None:
            self.dim1 = np.sqrt(self.dim)
            if int(self.dim1) **2 == self.dim: # check integer sqrt
                self.dim2 = self.dim1
                self.dims = [self.dim1, self.dim2] 
            else:
                print("Please enter dim1, dim2 ")
        else:
            self.dim1 = dim1
            self.dim2 = dim2
            self.dims = [self.dim1, self.dim2]  
    def get_data(self):
        return self.x_train, self.x_val, self.y_train, self.y_val

class MNIST(Dataset):
    def __init__(self, binary = False, val = 0, per_label = None):
        self.dims = [28,28]
        self.dim1 = 28
        self.dim2 = 28
        self.dim = 784
        self.binary = binary
        self.name = 'mnist' if not binary else 'binary_mnist'

        self.x_train, self.x_test, self.y_train, self.y_test = self.get_data()
        self.x_train = self.x_train[:(self.x_train.shape[0]-val), :]
        self.y_train = self.y_train[:(self.y_train.shape[0]-val)]
        self.x_val = self.x_train[(self.x_train.shape[0]-val):, :]

        if per_label is not None:
            pass

    def get_data(self, onehot = False, path = None):
        if path is None and self.binary:
            path = './datasets/mnist/MNIST_binary'
        elif path is None:
            path = 'MNIST_data/mnist.npz'

        def lines_to_np_array(lines):
            return np.array([[int(i) for i in line.split()] for line in lines])

        if not self.binary:
            #from keras.datasets import mnist
            #from tensorflow.keras.datasets.mnist import load_data
            (x_train, y_train), (x_test, y_test) = mnist_load()
            x_train = x_train.astype('float32') / 255.
            x_test = x_test.astype('float32') / 255.
            x_train = x_train.reshape((len(x_train), self.dim))
            x_test = x_test.reshape((len(x_test), self.dim))
            # Clever one hot encoding trick
            if onehot:
                y_train = np.eye(10)[y_train]
                y_test = np.eye(10)[y_test]
            return x_train, x_test, y_train, y_test

        else:
            with open(os.path.join(path, 'binarized_mnist_train.amat')) as f:
                lines = f.readlines()
            train_data = lines_to_np_array(lines).astype('float32')
            with open(os.path.join(path, 'binarized_mnist_valid.amat')) as f:
                lines = f.readlines()
            validation_data = lines_to_np_array(lines).astype('float32')
            with open(os.path.join(path, 'binarized_mnist_test.amat')) as f:
                lines = f.readlines()
            test_data = lines_to_np_array(lines).astype('float32')

            #from tensorflow.keras.datasets.mnist import load_data
            (x_train, y_train), (x_test, y_test) = mnist_load()
            #print(train_data.shape, y_train.shape)
            return train_data, test_data, y_train, y_test
            #return train_data, validation_data, test_data

class fMNIST(Dataset):
    def __init__(self, binary = False, val = 0, per_label = None):
        self.dims = [28,28]
        self.dim1 = 28
        self.dim2 = 28
        self.dim = 784
        #unused
        self.binary = binary
        self.name = 'fmnist' if not self.binary else 'binary_fmnist'

        #self.x_train, self.x_test, self.y_train, self.y_test = self.get_data()
        self.x_train, self.y_train = self.get_data(kind='train')
        self.x_test, self.y_test = self.get_data(kind='test')
        self.x_train = self.x_train[:(self.x_train.shape[0]-val), :]
        self.y_train = self.y_train[:(self.y_train.shape[0]-val)]
        self.x_val = self.x_train[(self.x_train.shape[0]-val):, :]

    def get_data(self, kind = 'train', path = None):
        if path is None:
            path = './datasets/mnist/fMNIST'

        """Load MNIST data from `path`"""
        labels_path = os.path.join(path,
                                   '%s-labels-idx1-ubyte.gz'
                                   % kind)
        images_path = os.path.join(path,
                                   '%s-images-idx3-ubyte.gz'
                                   % kind)

        with gzip.open(labels_path, 'rb') as lbpath:
            labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                                   offset=8)

        with gzip.open(images_path, 'rb') as imgpath:
            images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                                   offset=16).reshape(len(labels), 784)
        
        images = images.astype('float32') / 255.
        images = images.reshape((len(images), self.dim))
        if self.binary:
            np.random.seed(0)
        images = np.random.binomial(1,images, size = images.shape) if self.binary else images
        return images, labels



class DSprites(Dataset):
    def __init__(self):
        self.dims = [64,64]
        self.dim1 = 64
        self.dim2 = 64
        self.dim = 4096
        self.name = 'dsprites'
        self.x_train, _, self.y_train, _ = self.get_data()
        self.x_val = None
        self.x_test = None
        
    def get_data(self):
        dataset_zip = np.load('datasets/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz', encoding = 'bytes') # ‘latin1’
        imgs = dataset_zip['imgs']
        imgs = imgs.reshape((imgs.shape[0], -1))
        latent_ground_truth = dataset_zip['latents_values']
        latent_for_classification = dataset_zip['latents_classes']
        metadata = dataset_zip['metadata']
        return imgs, latent_ground_truth, latent_for_classification, metadata


class Omniglot(Dataset):
    def __init__(self, val = 0, per_label = None):
        self.dims = [28,28]
        self.dim1 = 28
        self.dim2 = 28
        self.dim = 784
        self.name = 'omniglot'

        self.x_train, self.x_test, self.lang_train, self.lang_test, self.y_train, self.y_test = self.get_data()
        self.x_train = self.x_train[:(self.x_train.shape[0]-val), :]
        self.y_train = self.y_train[:(self.y_train.shape[0]-val)]
        self.lang_train = self.lang_train[:(self.lang_train.shape[0]-val), :]
        self.x_val = self.x_train[(self.x_train.shape[0]-val):, :]
        self.y_val = self.y_train[(self.y_train.shape[0]-val):]
        self.lang_val = self.lang_train[(self.lang_train.shape[0]-val):, :]

        if per_label is not None:
            pass

    def get_data(self, onehot = False, path = None):
        if path is None:
            path = './datasets/omniglot'
        data = {}
        loadmat(os.path.join(path, 'chardata.mat'), data)
        x_train = data['data'].transpose()
        x_test = data['testdata'].transpose()
        lang_train = data['target'].transpose()
        y_train = data['targetchar'].transpose()
        lang_test = data['testtarget'].transpose()
        y_test = data['testtargetchar'].transpose()
        return x_train, x_test, lang_train, lang_test, y_train, y_test


class OmniglotOld(Dataset):
    def __init__(self):
        self.dim1 = 105
        self.dim2 = 105
        self.dim = 105*105
        self.dims = [105,105]
        self.name = 'omniglot'
        
        self.x_train= self.get_data()
        self.x_test = self.x_train[:10000]
        self.x_val = self.x_train[:10000]
        # TO DO: Create validation / test sets by resampling?
        #self.x_val = None
        #self.x_test = None 

    def get_data(self, file_dir = '/home/rcf-proj/gv/brekelma/autoencoders/datasets/omniglot'):
        with open(os.path.join(file_dir, 'omniglot_np.pickle'), "rb") as f:
            data = pickle.load(f)
        print()
        print("OMNIGLOT DATA", data.shape)
        print()
        data = data.astype('float32') / 255.
        return data

    def get_data2(self, per_char = None, total_chars = 964, imsize = 105, seed = 0, file_dir = 'datasets/omniglot/python/images_background'):
        try:
            with open(os.path.join(file_dir, 'omniglot_np.pickle'), "rb") as f:
                x_train = pickle.load(f)
            return x_train.astype('float32') / 255.
        except:
            for r, d, files in os.walk(file_dir):
                print("r split ", r.split("/")[-len(file_dir.split("/")):])
                print()
                if "/".join(r.split("/")[-len(file_dir.split("/")):]) == file_dir:
                    print("initial r ", r)
                    base_len = len(r.split("/"))
                    continue
                # actually need to go 2 levels down
                elif files and len(r.split("/")) == base_len + 2:
                    pc = len(files) if per_char is None else per_char

                    inds = np.random.randint(0, len(files), pc)
                    print(type(inds), inds.shape)
                    for f_ind in inds:
                        f = files[f_ind]
                        addimg = imread(os.path.join(r, f), flatten = True).flatten()
                        try:
                            print("img ", addimg.shape)
                        except:
                            print("img ", type(addimg))
                        try:
                            x_train = np.vstack([x_train, addimg])
                        except Exception as e:
                            print()
                            print(e)
                            print("trying single x_train") 
                            x_train = addimg
                else:
                    continue
            print("pickling omniglot data (per char = ", per_char, ")")
            #print(x_train.shape)
            #import IPython; IPython.embed()
            pickle.dump(x_train, open(os.path.join(file_dir, 'omniglot_np.pickle'), "wb"))
            return x_train.astype('float32') / 255.

        # imgs = np.random.randint(1, 20, (total_chars,per_char))
        # x_train = np.zeros((total_chars*per_char, imsize**2))
        # for i in range(total_chars):
        #     for j in range(per_char):
        #         k=imgs[i,j]
        #         fn = os.path.join(file_dir, str(i+1).zfill(4) + '_' + str(k).zfill(2) + '.png')
        #         x_train[i*per_char+j, :] = imread(fn, flatten = True).flatten()
        # return x_train


class EMNIST(Dataset):
    def __init__(self):
        raise NotImplementedError
    def get_data(self):
        raise NotImplementedError

class CelebA(Dataset):
    def __init__(self):
        raise NotImplementedError
    def get_data(self):
        raise NotImplementedError


class Cifar10(Dataset):
    def __init__(self):
        self.dim1 = 32
        self.dim2 = 32
        self.dim3 = 3
        self.dim = 32*32*3
        self.dims = [32,32,3]
        self.name = 'cifar10'
        self.x_train, self.x_test, self.y_train, self.y_test = self.get_data()
        
    def get_data(self):
        from keras.datasets import cifar10
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        return X_train, X_test, y_train, y_test
        #raise NotImplementedError
