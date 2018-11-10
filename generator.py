import keras
import numpy as np
import h5py


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, h5files, batch_size=32, shuffle=True):
        self.batch_size = batch_size
        self.h5files = h5files
        # TODO: creare una lista che contiene la lunghezza di ogni file di protoni e una uguale per i gamma
        self.h5fprot = [s for s in h5files if s.startswith('p')]    # h5files - protons
        self.h5fgamm = [s for s in h5files if s.startswith('g')]    # h5files - gammas
        self.h5f_p = h5py.File(self.h5fprot[-1], 'r')               # object containing the actual protons file
        self.h5f_g = h5py.File(self.h5fgamm[-1], 'r')               # object containing the actual gammas file
        self.h5p_idx = 0                                            # index that iterates on the actual protons file
        self.h5g_idx = 0                                            # index that iterates on the actual gammas file
        self.h5f_pl = len(self.h5f_p['LST/LST_image_charge_interp'][:])  # number of images in the actual protons file
        self.h5f_gl = len(self.h5f_g['LST/LST_image_charge_interp'][:])  # number of images in the actual gammas file
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        # TODO: check whether this method is called just one time, otherwise it starts to be a bit expensive
        # total number of images in the dataset
        n_images = 0

        for f in self.h5files:
            h5f = h5py.File(f, 'r')
            n_images += len(h5f['LST/LST_image_charge_interp'][:])
            h5f.close()

        return int(np.floor(n_images/self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        # indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        if self.h5p_idx >

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def count_images(self):



    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = np.load('data/' + ID + '.npy')

            # Store class
            y[i] = self.labels[ID]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)