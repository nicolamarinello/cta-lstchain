from multiprocessing import Process
from os import listdir
from os.path import isfile, join
from ctapipe.instrument import CameraGeometry
from astropy import units as u
from scipy.interpolate import griddata
import multiprocessing as mp
import numpy as np
import sys
import tables

'''
usage: python lst_interpolate path/to/folder1 path/to/folder2 path/to/folder3 ...
'''

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


def func(paths):

    print(paths)

    img_rows, img_cols = 100, 100
    
    # iterate on each proton file & concatenate charge arrays
    for f in paths:

        # get the data from the file
        data_p = tables.open_file(f)
        _, LST_image_charge, _ = get_LST_data(data_p)

        LST_image_charge = LST_image_charge[1:]
        
        # get camera geometry & camera pixels coordinates
        geom = CameraGeometry.from_name("LSTCam")
        points = np.array([np.array(geom.pix_x / u.m), np.array(geom.pix_y / u.m)]).T
        
        grid_x, grid_y = np.mgrid[-1.25:1.25:100j, -1.25:1.25:100j]

        # define the final array that will contain the interpolated images
        LST_image_charge_interp = np.zeros((len(LST_image_charge), img_rows, img_cols))

        for i in range(0, len(LST_image_charge)):
            LST_image_charge_interp[i] = griddata(points, LST_image_charge[i], (grid_x, grid_y), fill_value=0, method='cubic')

        print("Interpolated data: ")
        print(LST_image_charge_interp)

if __name__ == '__main__':

    ncpus = mp.cpu_count()
    print("Number of CPUs: " + str(ncpus))
    
    # get all the parameters given by the command line
    folders = sys.argv[1:]

    print('Folders: ' + str(folders))

    # create a single big list containing the paths of all the files
    all_files = []
    
    for path in folders:
        files = [join(path, f) for f in listdir(path) if (isfile(join(path, f)) and f.endswith(".h5"))]
        all_files = all_files + files

    print('All files: ' + str(all_files))

    num_files = len(all_files)

    if ncpus >= num_files:
        print('ncpus >= num_files')
        for f in all_files:
            Process(target=func, args=([f],)).start()


    #for i in all_files:
    #    Process(target=func1, args=(args.a,)).start()

    #Process(target=func, args=([all_files[0]],)).start()
    #Process(target=func, args=([all_files[0]],)).start()

    '''
    p1 = 
    p1.start()
    p2 = Process(target=func2, args=(args.b,))
    p2.start()
    # p1.join()
    # p2.join()

    '''