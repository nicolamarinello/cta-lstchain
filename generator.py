import keras
import numpy as np
import h5py
import multiprocessing
import os


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, h5files, batch_size=32, shuffle=True):
        self.batch_size = batch_size
        self.h5files = h5files
        self.indexes = np.array([], dtype=np.int64).reshape(0, 2)
        self.shuffle = shuffle
        self.n_images = 0
        self.generate_indexes()
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        # TODO: check whether this method is called just one time, otherwise it starts to be a bit expensive
        # total number of images in the dataset
        return int(np.floor(self.n_images/self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # index goes from 0 to the number of batches
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        # list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        x, y = self.__data_generation(indexes)

        return x, y

    def chunkit(self, seq, num):

        avg = len(seq) / float(num)
        out = []
        last = 0.0

        while last < len(seq):
            out.append(seq[int(last):int(last + avg)])
            last += avg

        return out

    def worker(self, h5files, positions, i, return_dict):

        idx = np.array([], dtype=np.int64).reshape(0, 2)

        for l, f in enumerate(h5files):
            h5f = h5py.File(f, 'r')
            length = len(h5f['LST/LST_image_charge_interp'][:])
            h5f.close()
            r = np.arange(length)
            cp = np.dstack(np.meshgrid([positions[l]], r)).reshape(-1, 2)  # cartesian product
            idx = np.append(idx, cp, axis=0)
        return_dict[i] = idx

    def generate_indexes(self):

        cpu_n = multiprocessing.cpu_count()
        pos = self.chunkit(np.arange(len(self.h5files)), cpu_n)
        h5f = self.chunkit(self.h5files, cpu_n)

        manager = multiprocessing.Manager()
        return_dict = manager.dict()

        processes = []

        for i in range(cpu_n):
            p = multiprocessing.Process(target=self.worker, args=(h5f[i], pos[i], i, return_dict))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        for key, value in return_dict.items():
            self.indexes = np.append(self.indexes, value, axis=0)

        self.n_images = self.indexes.shape[0]

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle:
            np.random.shuffle(self.indexes)     # shuffle all the pairs (if, ii) - (index file, index image in the file)

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples'
        # Initialization
        x = np.empty([self.batch_size, 100, 100])
        y = np.empty([self.batch_size], dtype=int)

        # Generate data
        for i, row in enumerate(indexes):

            filename = self.h5files[row[0]]
            fn_basename = os.path.basename(os.path.normpath(filename))

            clas = 0                                 # class: proton by default

            if fn_basename.startswith('g'):
                clas = 1

            h5f = h5py.File(filename, 'r')
            image = h5f['LST/LST_image_charge_interp'][row[1]]
            h5f.close()

            # Store image
            x[i, ] = image
            # Store class
            y[i] = clas

        x = x.reshape(x.shape[0], 1, 100, 100)

        return x, y