import multiprocessing
import os

import h5py
import keras
import numpy as np


class DataGeneratorC(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, h5files, batch_size=32, arrival_time=False, shuffle=True):
        self.batch_size = batch_size
        self.h5files = h5files
        self.indexes = np.array([], dtype=np.int64).reshape(0, 4)
        self.shuffle = shuffle
        self.generate_indexes()
        self.arrival_time = arrival_time
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        # total number of images in the dataset
        return int(np.floor(self.indexes.shape[0] / self.batch_size))

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

        return x, y, w

    def get_indexes(self):
        return self.indexes[0:self.__len__() * self.batch_size]

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

    def chunkit(self, seq, num):

        avg = len(seq) / float(num)
        out = []
        last = 0.0

        while last < len(seq):
            out.append(seq[int(last):int(last + avg)])
            last += avg

        return out

    def worker(self, h5files, positions, i, return_dict):

        idx = np.array([], dtype=np.int64).reshape(0, 4)

        for l, f in enumerate(h5files):
            h5f = h5py.File(f, 'r')
            lst_idx = h5f['LST/LST_event_index'][1:]
            h5f.close()
            r = np.arange(len(lst_idx))

            fn_basename = os.path.basename(os.path.normpath(f))

            clas = np.zeros(len(r))  # class: proton by default

            if fn_basename.startswith('g'):
                clas = np.ones(len(r))

            # cp = np.dstack(np.meshgrid([positions[l]], r, clas, lst_idx)).reshape(-1, 4)  # cartesian product
            cp = np.dstack(([positions[l]] * len(r), r, clas, lst_idx)).reshape(-1, 4)

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
            np.random.shuffle(self.indexes)  # shuffle all the pairs (if, ii) - (index file, index image in the file)

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples'
        # Initialization
        x = np.empty([self.batch_size, self.arrival_time + 1, 100, 100])
        y = np.empty([self.batch_size], dtype=int)

        # print('__data_generation', indexes)

        # Generate data
        for i, row in enumerate(indexes):

            # print(row[0])

            filename = self.h5files[int(row[0])]

            h5f = h5py.File(filename, 'r')
            # Store image
            x[i, 0] = h5f['LST/LST_image_charge_interp'][int(row[1])]
            if self.arrival_time:
                x[i, 1] = h5f['LST/LST_image_peak_times_interp'][int(row[1])]
            # Store class
            y[i] = int(row[2])

            h5f.close()

        return x, y


class DataGeneratorR(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, h5files, feature, batch_size=32, arrival_time=False, shuffle=True):
        self.batch_size = batch_size
        self.h5files = h5files
        self.feature = feature
        self.indexes = np.array([], dtype=np.int64).reshape(0, 3)
        self.shuffle = shuffle
        self.generate_indexes()
        self.arrival_time = arrival_time
        self.outcomes = 1
        if feature == 'xy': self.outcomes += 1
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        # total number of images in the dataset
        return int(np.floor(self.indexes.shape[0] / self.batch_size))

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

        # if self.test_mode:
        #    self.feature_array = np.append(self.feature_array, y)

        return x, y

    def get_indexes(self):
        return self.indexes[0:self.__len__() * self.batch_size]

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

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle:
            np.random.shuffle(self.indexes)  # shuffle all the pairs (if, ii) - (index file, index image in the file)

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples'
        # Initialization
        x = np.empty([self.batch_size, self.arrival_time + 1, 100, 100])
        y = np.empty([self.batch_size, self.outcomes], dtype=float)

        # Generate data
        for i, row in enumerate(indexes):

            filename = self.h5files[int(row[0])]

            h5f = h5py.File(filename, 'r')

            # Store image
            x[i, 0] = h5f['LST/LST_image_charge_interp'][int(row[1])]
            if self.arrival_time:
                x[i, 1] = h5f['LST/LST_image_peak_times_interp'][row[1]]

            # Store features
            if self.feature == 'energy':
                y[i] = np.log10(h5f['Event_Info/ei_mc_energy'][:][int(row[2])])
            elif self.feature == 'xy':
                y[i, 0] = h5f['Event_Info/ei_core_x'][:][int(row[2])]
                y[i, 1] = h5f['Event_Info/ei_core_y'][:][int(row[2])]

            h5f.close()

        # x = x.reshape(x.shape[0], 1, 100, 100)

        return x, y


class DataGeneratorCW(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, h5files, batch_size=32, arrival_time=False, weights=None, shuffle=True):
        self.batch_size = batch_size
        self.h5files = h5files
        self.indexes = np.array([], dtype=np.int64).reshape(0, 4)
        self.shuffle = shuffle
        self.generate_indexes()
        self.arrival_time = arrival_time
        self.weighted = False
        if weights is not None:
            self.weighted = True
            data = np.load(weights[0])
            self.gamm_edges = data['edges']
            self.gamm_w = data['class_weights']
            data = np.load(weights[1])
            self.prot_edges = data['edges']
            self.prot_w = data['class_weights']
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        # total number of images in the dataset
        return int(np.floor(self.indexes.shape[0] / self.batch_size))

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
        x, y, w = self.__data_generation(indexes)

        # print("training idx: ", indexes)

        return x, y, w

    def get_indexes(self):
        return self.indexes[0:self.__len__() * self.batch_size]

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

    def chunkit(self, seq, num):

        avg = len(seq) / float(num)
        out = []
        last = 0.0

        while last < len(seq):
            out.append(seq[int(last):int(last + avg)])
            last += avg

        return out

    def worker(self, h5files, positions, i, return_dict):

        idx = np.array([], dtype=np.int64).reshape(0, 4)

        for l, f in enumerate(h5files):
            h5f = h5py.File(f, 'r')
            lst_idx = h5f['LST/LST_event_index'][1:]
            h5f.close()
            r = np.arange(len(lst_idx))

            fn_basename = os.path.basename(os.path.normpath(f))

            clas = np.zeros(len(r))  # class: proton by default

            if fn_basename.startswith('g'):
                clas = np.ones(len(r))

            # cp = np.dstack(np.meshgrid([positions[l]], r, clas, lst_idx)).reshape(-1, 4)  # cartesian product
            cp = np.dstack(([positions[l]] * len(r), r, clas, lst_idx)).reshape(-1, 4)

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
            np.random.shuffle(self.indexes)  # shuffle all the pairs (if, ii) - (index file, index image in the file)

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples'
        # Initialization
        x = np.empty([self.batch_size, self.arrival_time + 1, 100, 100])
        y = np.empty([self.batch_size], dtype=int)
        w = np.ones([self.batch_size], dtype=float)

        # print('__data_generation', indexes)

        # Generate data
        for i, row in enumerate(indexes):

            # print(row[0])

            filename = self.h5files[int(row[0])]

            h5f = h5py.File(filename, 'r')
            # Store image
            x[i, 0] = h5f['LST/LST_image_charge_interp'][int(row[1])]
            if self.arrival_time:
                x[i, 1] = h5f['LST/LST_image_peak_times_interp'][int(row[1])]
            # Store class
            y[i] = int(row[2])

            if self.weighted:
                e = np.log10(h5f['Event_Info/ei_mc_energy'][:][int(row[3])])
                if row[2] == 1:
                    j = np.digitize(e, self.gamm_edges)
                    w[i] = self.gamm_w[j - 1]
                else:
                    j = np.digitize(e, self.prot_edges)
                    w[i] = self.prot_w[j - 1]

            h5f.close()

        return x, y, w


class DataGeneratorRW(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, h5files, feature, batch_size=32, arrival_time=False, weights=None, shuffle=True):
        self.batch_size = batch_size
        self.h5files = h5files
        self.feature = feature
        self.indexes = np.array([], dtype=np.int64).reshape(0, 3)
        self.shuffle = shuffle
        self.generate_indexes()
        self.arrival_time = arrival_time
        self.weighted = False
        if weights is not None:
            self.weighted = True
            data = np.load(weights[0])
            self.gamm_edges = data['edges']
            self.gamm_w = data['class_weights']
        self.outcomes = 1
        if feature == 'xy': self.outcomes += 1
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        # total number of images in the dataset
        return int(np.floor(self.indexes.shape[0] / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # index goes from 0 to the number of batches
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        # list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        x, y, w = self.__data_generation(indexes)

        # print("training idx: ", indexes)

        # if self.test_mode:
        #    self.feature_array = np.append(self.feature_array, y)

        return x, y, w

    def get_indexes(self):
        return self.indexes[0:self.__len__() * self.batch_size]

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

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle:
            np.random.shuffle(self.indexes)  # shuffle all the pairs (if, ii) - (index file, index image in the file)

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples'
        # Initialization
        x = np.empty([self.batch_size, self.arrival_time + 1, 100, 100])
        y = np.empty([self.batch_size, self.outcomes], dtype=float)
        w = np.ones([self.batch_size], dtype=float)

        # Generate data
        for i, row in enumerate(indexes):

            filename = self.h5files[int(row[0])]

            h5f = h5py.File(filename, 'r')

            # Store image
            x[i, 0] = h5f['LST/LST_image_charge_interp'][int(row[1])]
            if self.arrival_time:
                x[i, 1] = h5f['LST/LST_image_peak_times_interp'][row[1]]

            # Store features
            if self.feature == 'energy':
                y[i] = np.log10(h5f['Event_Info/ei_mc_energy'][:][int(row[2])])
            elif self.feature == 'xy':
                y[i, 0] = h5f['Event_Info/ei_core_x'][:][int(row[2])]
                y[i, 1] = h5f['Event_Info/ei_core_y'][:][int(row[2])]

            # store weight
            if self.weighted:
                e = np.log10(h5f['Event_Info/ei_mc_energy'][:][int(row[3])])
                j = np.digitize(e, self.gamm_edges)
                w[i] = self.gamm_w[j - 1]

            h5f.close()

        # x = x.reshape(x.shape[0], 1, 100, 100)

        return x, y, w
