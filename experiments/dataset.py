from itertools import product
import numpy as np
import six
from six.moves.urllib.request import urlretrieve
from abc import ABC, abstractmethod
#from scipy.misc import imread
from scipy.io import loadmat
import os 
import gzip
import pickle
import copy 
from evaluation import util
#from disentanglement_lib.data.ground_truth import util


#BASE_DATASETS = '/auto/rcf-proj/gv/brekelma/echo_sandbox/datasets
BASE_DATASETS = '/nas/home/brekelma/datasets'

#BASE_DATASETS = '/data/brekelma/echo_sandbox/datasets' #/home/ec2-user/echo_sandbox/datasets' #'/home/rcf-proj/gv/brekelma/datasets'
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
            path = os.path.join(BASE_DATASETS, 'mnist', 'MNIST_binary')
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
    def __init__(self, binary = False, val =10000, per_label = None):
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
            path = os.path.join(BASE_DATASETS, 'mnist', 'fMNIST')

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
        self.x_train, self.latent_ground_truth, self.y_train, _ = self.get_data()
        self.x_val = None
        self.x_test = self.x_train #None
        self.y_test = self.y_train

    def get_data(self):
        dataset_zip = np.load(os.path.join(BASE_DATASETS,'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'), encoding = 'bytes', allow_pickle=True) # ‘latin1’
        imgs = dataset_zip['imgs']
        imgs = imgs.reshape((imgs.shape[0], -1))
        latent_ground_truth = dataset_zip['latents_values']
        latent_for_classification = dataset_zip['latents_classes']
        
        try:
            self.factor_sizes = np.array(
                dataset_zip["metadata"][()]["latents_sizes"], dtype=np.int64)
            self.full_factor_sizes = [1, 3, 6, 40, 32, 32]
            self.state_space = util.SplitDiscreteStateSpace(self.factor_sizes,
                                                 self.latent_factor_indices)
        except:
            pass
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
            path =os.path.join(BASE_DATASETS, 'omniglot')
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
    def __init__(self, normalize = False):
        self.dim1 = 32
        self.dim2 = 32
        self.dim3 = 3
        self.dim = 3072
        self.dims = [32,32,3]
        self.name = 'cifar10'
        self.normalize = normalize
        self.x_train, self.x_test, self.y_train, self.y_test = self.get_data()

    def get_data(self):
        from keras.datasets import cifar10
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()

        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        X_train = X_train / 255.0
        X_test = X_test / 255.0

        norm_mu = [.4814, .4822, .4465]
        norm_std = [.246, .243, .261]

        if self.normalize:
            X_train[...,0] = np.divide((X_train[...,0] - norm_mu[0]), norm_std[0])
            X_train[...,1] = np.divide((X_train[...,1] - norm_mu[1]), norm_std[1])
            X_train[...,2] = np.divide((X_train[...,2] - norm_mu[2]), norm_std[2])
            X_test[...,0] = np.divide((X_test[...,0] - norm_mu[0]), norm_std[0])
            X_test[...,1] = np.divide((X_test[...,1] - norm_mu[1]), norm_std[1])
            X_test[...,2] = np.divide((X_test[...,2] - norm_mu[2]), norm_std[2])

        return X_train, X_test, y_train, y_test
        #raise NotImplementedError                                                                                                                                                                                

def get_coords_from_index(idx, factor_sizes):
    coords = []
    cur = idx
    for n in factor_sizes[::-1]:
        coords.append(cur % n)
        cur = cur // n
    return coords[::-1]

    


class DSpritesCorrelated(Dataset):
    def __init__(self, remove_prob = 0.2, n_blocks=8,
                data_dir = None, indices=None, classification=False, colored=False):    
        self.dims = [64,64]
        self.dim1 = 64
        self.dim2 = 64
        self.dim = 4096
        self.name = 'dsprites'

        self.valid_indices, self.invalid = self.get_correlated_indices(remove_prob=remove_prob, seed=42, classification=False,
                                      colored=False, n_blocks=n_blocks)

        self.x_train, self.y_train = self.get_data(indices = self.valid_indices)
        self.x_test, self.y_test = self.get_data(indices = self.invalid)

        print("DATASET Init as TRAIN: ", self.x_train.shape, "TEST : ", self.x_test.shape, ' invalid : ', np.array(self.invalid.shape))
        
        print("CHANGING to : ")

    def get_data(self, indices = None, classification = False, colored = False):
        data_dir = os.path.join(BASE_DATASETS,'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')

        #super(DSpritesDataset, self).__init__()
        data = np.load(data_dir, encoding = 'latin1', allow_pickle=True)
        #np.load(data_dir, encoding='latin1')

        indices = self.indices if indices is None else indices
        # color related stuff
        self.colored = colored
        self.colors = None
        self.n_colors = 1
        indices_without_color = indices
        if colored:
            color_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      'resources/rainbow-7.npy')
            self.colors = np.load(color_file)
            self.n_colors = len(self.colors)
            indices_without_color = [idx // self.n_colors for idx in indices]

        # factor_names and factor_sizes
        

        meta = data['metadata'].item()
        try:
            self.factor_names = list(meta['latents_names'][1:])
            self.factor_sizes = list(meta['latents_sizes'][1:])
        except:
            self.factor_names = ['shape', 'scale', 'orientation', 'posX', 'posY']
            self.factor_sizes = [3, 6, 40, 32, 32]


        if colored:
            self.factor_names.append('color')
            self.factor_sizes.append(self.n_colors)
        self.n_factors = len(self.factor_names)

        # save relevant part of the grid
        self.imgs = data['imgs'][indices_without_color]
        print()
        print("IMGS ", self.imgs.shape)
        print()
        # factor values, classes, possible_values
        self.factor_values = data['latents_values'][indices_without_color]
        self.factor_values = [arr[1:] for arr in self.factor_values]

        self.factor_classes = data['latents_classes'][indices_without_color]
        self.factor_classes = [arr[1:] for arr in self.factor_classes]

        self.possible_values = []
        for name in ['shape', 'scale', 'orientation', 'posX', 'posY']:
            #print("Factor : ", name, " . Values : ", meta['latents_possible_values'][name])
            self.possible_values.append(meta['latents_possible_values'][name])

        if colored:
            for i, idx in enumerate(indices):
                color_class = idx % self.n_colors
                color_value = color_class / (self.n_colors - 1.0)
                self.factor_classes[i] = np.append(self.factor_classes[i], color_class)
                self.factor_values[i] = np.append(self.factor_values[i], color_value)
            self.possible_values.append(list(np.arange(0, self.n_colors) / (self.n_colors - 1.0)))

        self.classification = classification

        # dataset name
        self.dataset_name = 'dsprites'
        self.name = 'dsprites'
        if self.colored: 
            self.dataset_name += '-colored'

        # factor types
        self.is_categorical = [True, False, False, False, False]
        if self.colored:
            self.is_categorical.append(True)
        
        self.imgs = np.squeeze(self.imgs)
        x_train = self.imgs.reshape((self.imgs.shape[0], -1))
        y_train = self.factor_classes

        return x_train, y_train

    def get_correlated_indices(self, remove_prob=0.2, seed=2, classification=False,
                                      colored=False, n_blocks=8):
        
        np.random.seed(seed)
        n_factors = 5
        factor_sizes = [3, 6, 40, 32, 32]
        is_categorical = [True, False, False, False, False]
        if colored:
            n_factors += 1
            factor_sizes.append(7)
            is_categorical.append(True)
        N = np.prod(factor_sizes).astype(int)
        invalid_pairs = [[None] * n_factors for _ in range(n_factors)]

        def split_into_blocks(n_states, n_blocks):
            block_lens = [n_states // n_blocks for _ in range(n_blocks)]
            for idx in range(n_states % n_blocks):
                block_lens[idx] += 1
            blocks = []
            start = 0
            for size in block_lens:
                blocks.append((start, start + size))
                start += size
            return blocks
        
        count = 0
        total = 0
        print("N FACTORS ", n_factors)
        for i in range(n_factors):
            for j in range(i+1, n_factors):
                ni = (factor_sizes[i] if is_categorical[i] else min(n_blocks, factor_sizes[i]))
                nj = (factor_sizes[j] if is_categorical[j] else min(n_blocks, factor_sizes[j]))
                a = split_into_blocks(factor_sizes[i], ni)
                b = split_into_blocks(factor_sizes[j], nj)
                pairs = list(product(a, b))
                invalid_pairs[i][j] = set()
                total += len(pairs)
                for ra, rb in pairs:
                    if np.random.rand() < remove_prob:
                        count +=1
                        for x in range(ra[0], ra[1]):
                            for y in range(rb[0], rb[1]):
                                invalid_pairs[i][j].add((x, y))
        invalid = np.zeros((N,), dtype=np.bool)
        for idx in range(N):
            coords = get_coords_from_index(idx, factor_sizes)

            for i in range(n_factors):
                for j in range(i+1, n_factors):
                    if (coords[i], coords[j]) in invalid_pairs[i][j]:
                        invalid[idx] = True

        valid_indices = np.where(~invalid)[0]
        print()
        print("VALID SIZE ", valid_indices.shape)
    #    return self.valid_indices

    #def calc_dist_info(self):
        #valid_indices = self.valid_indices
        # calculate statistics about the distribution
        dist_info = dict()
        probs = (~invalid).astype(np.float) / len(valid_indices)
        marginal_probs = [np.zeros((fk_states,)) for fk_states in factor_sizes]

        for state_idx in range(N):
            coords = get_coords_from_index(state_idx, factor_sizes)
            for c, k in zip(coords, range(n_factors)):
                marginal_probs[k][c] += probs[state_idx]

        marginal_entropies = np.zeros((n_factors,))
        for k in range(n_factors):
            for fk_val in range(factor_sizes[k]):
                p = marginal_probs[k][fk_val]
                if p > 0:
                    marginal_entropies[k] -= p * np.log(p)

        joint_entropy = np.log(len(valid_indices))

        dist_info['invalid_pairs'] = invalid_pairs
        dist_info['probs'] = probs
        dist_info['marginal_probs'] = marginal_probs
        dist_info['marginal_entropies'] = marginal_entropies
        dist_info['joint_entropy'] = joint_entropy
        dist_info['TC'] = np.sum(marginal_entropies) - joint_entropy


        print("*"*50, "DISTRIBUTION STATS ", "*"*50)
        for k in dist_info.keys():
            print(k, ": ", dist_info[k])
        print("*"*100)

        self.dist_info = dist_info

        #self.valid_indices = valid_indices
        invalid = np.where(invalid)[0]
        return valid_indices, invalid #dist_info

    def __len__(self):
        return len(self.imgs)
