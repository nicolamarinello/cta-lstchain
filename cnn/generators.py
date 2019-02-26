import keras
import numpy as np
import h5py
import multiprocessing
# import threading
import os


class DataGeneratorC(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, h5files, batch_size=32, arrival_time=False, val_per=0.2, shuffle=True):
        self.batch_size = batch_size
        self.h5files = h5files
        self.indexes = np.array([], dtype=np.int64).reshape(0, 3)
        self.shuffle = shuffle
        self.generate_indexes()
        self.arrival_time = arrival_time
        self.on_epoch_end()
        self.val_indexes = np.array([])
        if val_per > 0:
            # split into training and validation
            self.indexes, self.val_indexes = np.split(self.indexes, [int(self.indexes.shape[0]*(1-val_per))])
            # sort val_indexes by file index to read images faster from disk
            self.val_indexes = np.sort(self.val_indexes.view('i8,i8,i8'), order=['f1'], axis=0).view(np.int)

    def __len__(self):
        'Denotes the number of batches per epoch'
        # total number of images in the dataset
        return int(np.floor(self.indexes.shape[0]/self.batch_size))

    def __getitem__(self, index):

        # print("training idx: ", index, '/', self.__len__())

        # with self.lock:
        'Generate one batch of data'
        # index goes from 0 to the number of batches
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        # list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        x, y = self.__data_generation(indexes)

        # print("training idx: ", indexes)

        return x, y

    def get_indexes(self):
        return self.indexes, self.val_indexes

    def get_event(self, idx):

        filename = self.h5files[idx[0]]

        h5f = h5py.File(filename, 'r')
        image = h5f['LST/LST_image_charge'][idx[1]]
        time = h5f['LST/LST_image_peak_times'][idx[1]]
        lst_idx = h5f['LST/LST_event_index'][idx[1]]
        mc_energy = h5f['Event_Info/ei_mc_energy'][:][lst_idx]
        h5f.close()

        gt = idx[2]

        return image, time, gt, mc_energy

    def get_val(self):

        # Generate indexes of the batch
        indexes = self.val_indexes

        old_bs = self.batch_size

        self.batch_size = len(self.val_indexes)

        # Generate data
        x, y = self.__data_generation(indexes)

        self.batch_size = old_bs

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

        idx = np.array([], dtype=np.int64).reshape(0, 3)

        for l, f in enumerate(h5files):
            h5f = h5py.File(f, 'r')
            lst_idx = h5f['LST/LST_event_index'][1:]
            h5f.close()
            r = np.arange(len(lst_idx))

            fn_basename = os.path.basename(os.path.normpath(f))

            clas = 0  # class: proton by default

            if fn_basename.startswith('g'):
                clas = 1

            cp = np.dstack(np.meshgrid([positions[l]], r, clas)).reshape(-1, 3)  # cartesian product

            idx = np.append(idx, cp, axis=0)
        return_dict[i] = idx

    def generate_indexes(self):

        cpu_n = multiprocessing.cpu_count()
        pos = self.chunkit(np.arange(len(self.h5files)), cpu_n)
        h5f = self.chunkit(self.h5files, cpu_n)

        manager = multiprocessing.Manager()
        return_dict = manager.dict()

        processes = []

        if cpu_n >= len(self.h5files):
            # print('ncpus >= num_files')
            for i, f in enumerate(self.h5files):
                p = multiprocessing.Process(target=self.worker, args=([f], [i], i, return_dict))
                p.start()
                processes.append(p)
        else:
            # print('ncpus < num_files')
            for i in range(cpu_n):
                p = multiprocessing.Process(target=self.worker, args=(h5f[i], pos[i], i, return_dict))
                p.start()
                processes.append(p)

        for p in processes:
            p.join()

        for key, value in return_dict.items():
            self.indexes = np.append(self.indexes, value, axis=0)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle:
            np.random.shuffle(self.indexes)     # shuffle all the pairs (if, ii) - (index file, index image in the file)

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples'
        # Initialization
        x = np.empty([self.batch_size, self.arrival_time+1, 100, 100])
        y = np.empty([self.batch_size], dtype=int)

        # print('__data_generation', indexes)

        # Generate data
        for i, row in enumerate(indexes):

            # print(row[0])

            filename = self.h5files[row[0]]

            h5f = h5py.File(filename, 'r')
            # Store image
            x[i, 0] = h5f['LST/LST_image_charge_interp'][row[1]]
            if self.arrival_time:
                x[i, 0] = h5f['LST/LST_image_peak_times_interp'][row[1]]
            h5f.close()
            # Store class
            y[i] = row[2]

            # if y[i] == 0:
            #    x[i,] = np.full((100, 100), 0)
            # if y[i] == 1:
            #    x[i,] = np.full((100, 100), 1)

        x = x.reshape(x.shape[0], 1, 100, 100)

        return x, y


class DataGeneratorR(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, h5files, feature, batch_size=32, shuffle=True):
        self.batch_size = batch_size
        self.h5files = h5files
        self.feature = feature
        self.indexes = np.array([], dtype=np.int64).reshape(0, 3)
        self.shuffle = shuffle
        self.n_images = 0
        self.generate_indexes()
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
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

        # print("training idx: ", indexes)

        return x, y

    def get_indexes(self):
        return self.indexes

    def chunkit(self, seq, num):

        avg = len(seq) / float(num)
        out = []
        last = 0.0

        while last < len(seq):
            out.append(seq[int(last):int(last + avg)])
            last += avg

        return out

    def worker(self, h5files, positions, i, return_dict):

        idx = np.array([], dtype=np.int64).reshape(0, 3)

        for l, f in enumerate(h5files):
            h5f = h5py.File(f, 'r')
            lst_idx = h5f['LST/LST_event_index'][1:]
            h5f.close()

            r = np.arange(len(lst_idx))

            cp = np.dstack(([positions[l]] * len(r), r, lst_idx)).reshape(-1, 3)

            idx = np.append(idx, cp, axis=0)
        return_dict[i] = idx

    def generate_indexes(self):

        cpu_n = multiprocessing.cpu_count()
        pos = self.chunkit(np.arange(len(self.h5files)), cpu_n)
        h5f = self.chunkit(self.h5files, cpu_n)

        manager = multiprocessing.Manager()
        return_dict = manager.dict()

        processes = []

        if cpu_n >= len(self.h5files):
            # print('ncpus >= num_files')
            for i, f in enumerate(self.h5files):
                p = multiprocessing.Process(target=self.worker, args=([f], [i], i, return_dict))
                p.start()
                processes.append(p)
        else:
            # print('ncpus < num_files')
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
        y = np.empty([self.batch_size], dtype=float)

        # Generate data
        for i, row in enumerate(indexes):

            filename = self.h5files[int(row[0])]

            h5f = h5py.File(filename, 'r')

            # Store image
            x[i, ] = h5f['LST/LST_image_charge_interp'][int(row[1])]

            # Store features
            if self.feature == 'energy':
                y[i] = h5f['Event_Info/ei_mc_energy'][:][int(row[2])]
            elif self.feature == 'az':
                y[i] = h5f['Event_Info/ei_az'][:][int(row[2])]

            h5f.close()

        x = x.reshape(x.shape[0], 1, 100, 100)

        return x, y