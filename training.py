from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D
#from keras.callbacks import EarlyStopping, TensorBoard
#from sklearn.metrics import accuracy_score, f1_score
#from datetime import datetime
from ctapipe.instrument import CameraGeometry
from astropy import units as u
from sklearn.model_selection import train_test_split
from scipy.interpolate import griddata
import tables
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

    df = pd.DataFrame()

    data_LST = data.root.LST

    # LST data
    df['LST_event_index'] = [x['event_index'] for x in data_LST.iterrows()]
    df['LST_image_charge'] = [x['image_charge'] for x in data_LST.iterrows()]
    df['LST_image_peak_times'] = [x['image_peak_times'] for x in data_LST.iterrows()]

    return df


if __name__ == "__main__":

    data_path_p = '/Users/nicolamarinello/ctasoft/simulations/Paranal_proton_North_20deg_3HB9_DL1_ML1/proton_20deg_0deg_srun13316-33715___cta-prod3_desert-2150m-Paranal-HB9.h5'
    data_path_g = '/Users/nicolamarinello/ctasoft/simulations/Paranal_gamma-diffuse_North_20deg_3HB9_DL1_ML1/gamma_20deg_0deg_srun5865-23126___cta-prod3_desert-2150m-Paranal-HB9_cone10.h5'
    
    data_p = tables.open_file(data_path_p)
    data_g = tables.open_file(data_path_g)
    
    epochs = 10
    num_classes = 2
    batch_size = 128
    batch_size = 128
    img_rows, img_cols = 100, 100

    df_g = get_LST_data(data_g)
    df_p = get_LST_data(data_p)
    df_g['y'] = 1
    df_p['y'] = 0

    df = pd.DataFrame()

    df = df.append(df_g)
    df = df.append(df_p)

    geom = CameraGeometry.from_name("LSTCam")
    points = np.array([np.array(geom.pix_x / u.m), np.array(geom.pix_y / u.m)]).T

    df = df.drop(columns=['LST_event_index','LST_image_peak_times'])

    # slow operation <-------------------------------------
    for index, row in df.iterrows():
        #print(type(row['LST_image_charge']))
        values = np.array(row['LST_image_charge'])
        grid_x, grid_y = np.mgrid[-1.25:1.25:100j, -1.25:1.25:100j]
        grid_z = griddata(points, values, (grid_x, grid_y), method='cubic')
        grid_z = np.nan_to_num(grid_z)
        row['LST_image_charge'] = grid_z
        print(row['LST_image_charge'].shape)

    #print(df)

    x_train, x_test, y_train, y_test = train_test_split(df.loc[:, df.columns != 'y'], df.loc[:, df.columns == 'y'], test_size=0.2, random_state=42)

    print(x_train['LST_image_charge'].tolist()[0].shape)
    #print(y_train['y'])

    # remove 

    # classifier

    classifier = Sequential()


    classifier.add(Conv2D(32, (3, 3), input_shape = (100, 100, 1), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    classifier.add(Flatten())
    classifier.add(Dense(units = 128, activation = 'relu'))
    classifier.add(Dense(units = 1, activation = 'sigmoid'))

    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    #classifier.fit_generator(training_set, steps_per_epoch = 8000, epochs = 25, validation_data = test_set, validation_steps = 2000)

    classifier.fit(x=x_train['LST_image_charge'], y=y_train['y'], epochs=1, verbose=1, validation_split=0.15, shuffle=True)



