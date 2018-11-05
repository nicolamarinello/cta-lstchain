from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D
#from keras.callbacks import EarlyStopping, TensorBoard
#from sklearn.metrics import accuracy_score, f1_score
from ctapipe.instrument import CameraGeometry
from astropy import units as u
from sklearn.model_selection import train_test_split
from scipy.interpolate import griddata
from os import listdir
from os.path import isfile, join
import datetime
import matplotlib.pyplot as plt
import tables
import keras
import numpy as np
import pandas as pd


def get_array_data(data):

    df = pd.DataFrame()

    data_ainfo = data.root.Array_Info

    # array info data
    df['ai_run_array_direction'] = [x['run_array_direction'] for x in data_ainfo.iterrows()]
    df['ai_tel_id'] = [x['tel_id'] for x in data_ainfo.iterrows()]
    df['ai_tel_type'] = [x['tel_type'] for x in data_ainfo.iterrows()]
    df['ai_tel_x'] = [x['tel_x'] for x in data_ainfo.iterrows()]
    df['ai_tel_y'] = [x['tel_y'] for x in data_ainfo.iterrows()]
    df['ai_tel_z'] = [x['tel_z'] for x in data_ainfo.iterrows()]

    return df

def get_event_data(data):

    df = pd.DataFrame()

    data_einfo = data.root.Event_Info

    # event info data
    df['ei_alt'] = [x['alt'] for x in data_einfo.iterrows()]
    df['ei_az'] = [x['az'] for x in data_einfo.iterrows()]
    df['ei_core_x'] = [x['core_x'] for x in data_einfo.iterrows()]
    df['ei_core_y'] = [x['core_y'] for x in data_einfo.iterrows()]
    df['ei_event_number'] = [x['event_number'] for x in data_einfo.iterrows()]
    df['ei_h_first_int'] = [x['h_first_int'] for x in data_einfo.iterrows()]
    df['ei_mc_energy'] = [x['mc_energy'] for x in data_einfo.iterrows()]
    df['ei_particle_id'] = [x['particle_id'] for x in data_einfo.iterrows()]
    df['ei_run_number'] = [x['run_number'] for x in data_einfo.iterrows()]
    df['ei_LST_indices'] = [x['LST_indices'] for x in data_einfo.iterrows()]

    return df

def get_LST_data(data):

    data_LST = data.root.LST

    # LST data
    LST_event_index = [x['event_index'] for x in data_LST.iterrows()]
    LST_image_charge = [x['image_charge'] for x in data_LST.iterrows()]
    LST_image_peak_times = [x['image_peak_times'] for x in data_LST.iterrows()]

    return LST_event_index, LST_image_charge, LST_image_peak_times


if __name__ == "__main__":

    epochs = 10
    num_classes = 2
    batch_size = 128
    batch_size = 128
    img_rows, img_cols = 100, 100

    # define paths containing protons & gammas
    data_path_p = '../../simulations/Paranal_proton_North_20deg_3HB9_DL1_ML1/'
    data_path_g = '../../simulations/Paranal_gamma-diffuse_North_20deg_3HB9_DL1_ML1/'

    # retrieve file lists for protons & gammas
    files_p = [f for f in listdir(data_path_p) if isfile(join(data_path_p, f))]
    files_g = [f for f in listdir(data_path_g) if isfile(join(data_path_g, f))]
    
    # define the array that will collect the simulated charge for protons
    LST_image_charge_pp = np.array([], dtype=np.int64).reshape(0,1855)
    
    # iterate on each proton file & concatenate charge arrays
    for f in files_p:
        data_p = tables.open_file(join(data_path_p, f))
        _, LST_image_charge_p, _ = get_LST_data(data_p)
        LST_image_charge_pp = np.concatenate((LST_image_charge_pp,LST_image_charge_p[1:]), axis=0)

    # define the array that will collect the simulated charge for gammas
    LST_image_charge_gg = np.array([], dtype=np.int64).reshape(0,1855)

    # iterate on each gamma file & concatenate charge arrays
    for f in files_g:
        data_g = tables.open_file(join(data_path_g, f))
        _, LST_image_charge_g, _ = get_LST_data(data_p)
        LST_image_charge_gg = np.concatenate((LST_image_charge_gg,LST_image_charge_g[1:]), axis=0)
    
    # concatenate protons & gammas values
    LST_image_charge = np.concatenate((LST_image_charge_pp,LST_image_charge_gg), axis=0)
    
    # set a label for gammas & protons
    n_g = len(LST_image_charge_gg)
    n_p = len(LST_image_charge_pp)

    y_g = np.ones((n_g, 1), dtype=np.int8)
    y_p = np.zeros((n_p, 1), dtype=np.int8)
    
    y_ = np.concatenate((y_g, y_p), axis=0)

    print(LST_image_charge)
    print(y_)

    # get camera geometry & camera pixels coordinates
    geom = CameraGeometry.from_name("LSTCam")
    points = np.array([np.array(geom.pix_x / u.m), np.array(geom.pix_y / u.m)]).T

    # define the final array that will contain the interpolated images
    LST_image_charge_interp = np.zeros((len(LST_image_charge), 1, img_rows, img_cols))

    # slow operation <-------------------------------------
    grid_x, grid_y = np.mgrid[-1.25:1.25:100j, -1.25:1.25:100j]
    for i in range(0, len(LST_image_charge)):
        values = LST_image_charge[i]       
        grid_z = griddata(points, values, (grid_x, grid_y), method='cubic')
        grid_z = np.nan_to_num(grid_z)
        LST_image_charge_interp[i,:,:,:] = grid_z.reshape(1, img_rows, img_cols)

    # splitting entire dataset in train & test sets
    x_train, x_test, y_train, y_test = train_test_split(LST_image_charge_interp, y_, test_size=0.2, random_state=42)

    # define the network model
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(1, img_rows, img_cols), data_format="channels_first"))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units = 1, activation='sigmoid'))
    
    model.summary()

    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    history = model.fit(x=x_train, y=y_train, epochs=10, verbose=1, validation_split=0.2, shuffle=True)
    score = model.evaluate(x=x_test, y=y_test, batch_size=None, verbose=1, sample_weight=None, steps=None)

    now = datetime.datetime.now()

    # save the model
    model.save('LST_classifier_' + str(now.strftime("%Y-%m-%d %H:%M")) + '.h5') 
    
    print('Test loss:' + str(score[0]))
    print('Test accuracy:' + str(score[1]))