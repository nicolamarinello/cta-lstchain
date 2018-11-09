from multiprocessing import Process
from os import listdir, remove
from os.path import isfile, join
from ctapipe.instrument import CameraGeometry
from astropy import units as u
from scipy.interpolate import griddata
from tables.exceptions import HDF5ExtError
import multiprocessing as mp
import numpy as np
import h5py
import sys
import tables

'''
usage: python lst_interpolate path/to/folder1 path/to/folder2 path/to/folder3 ...
'''

def get_array_data(data):

    data_ainfo = data.root.Array_Info

    # array info data
    ai_run_array_direction = [x['run_array_direction'] for x in data_ainfo.iterrows()]
    ai_tel_id = [x['tel_id'] for x in data_ainfo.iterrows()]
    ai_tel_type = [x['tel_type'] for x in data_ainfo.iterrows()]
    ai_tel_x = [x['tel_x'] for x in data_ainfo.iterrows()]
    ai_tel_y = [x['tel_y'] for x in data_ainfo.iterrows()]
    ai_tel_z = [x['tel_z'] for x in data_ainfo.iterrows()]

    return data_ainfo, ai_run_array_direction, ai_tel_id, ai_tel_type, ai_tel_x, ai_tel_y, ai_tel_z

def get_event_data(data):

    data_einfo = data.root.Event_Info

    # event info data
    ei_alt = [x['alt'] for x in data_einfo.iterrows()]
    ei_az = [x['az'] for x in data_einfo.iterrows()]
    ei_core_x = [x['core_x'] for x in data_einfo.iterrows()]
    ei_core_y = [x['core_y'] for x in data_einfo.iterrows()]
    ei_event_number = [x['event_number'] for x in data_einfo.iterrows()]
    ei_h_first_int = [x['h_first_int'] for x in data_einfo.iterrows()]
    ei_mc_energy = [x['mc_energy'] for x in data_einfo.iterrows()]
    ei_particle_id = [x['particle_id'] for x in data_einfo.iterrows()]
    ei_run_number = [x['run_number'] for x in data_einfo.iterrows()]
    ei_LST_indices = [x['LST_indices'] for x in data_einfo.iterrows()]

    return data_einfo, ei_alt, ei_az, ei_core_x, ei_core_y, ei_event_number, ei_h_first_int, ei_mc_energy, ei_particle_id, ei_run_number, ei_LST_indices

def get_LST_data(data):

    data_LST = data.root.LST

    # LST data
    LST_event_index = [x['event_index'] for x in data_LST.iterrows()]
    LST_image_charge = [x['image_charge'] for x in data_LST.iterrows()]
    LST_image_peak_times = [x['image_peak_times'] for x in data_LST.iterrows()]

    return data_LST, LST_event_index, LST_image_charge, LST_image_peak_times


def func(paths):

    print(paths)

    img_rows, img_cols = 100, 100
    
    # iterate on each proton file & concatenate charge arrays
    for f in paths:

        # get the data from the file
        try:
            data_p = tables.open_file(f)
            _, _, LST_image_charge, _ = get_LST_data(data_p)
            _, ei_alt, ei_az, ei_core_x, ei_core_y, ei_event_number, ei_h_first_int, ei_mc_energy, ei_particle_id, ei_run_number, ei_LST_indices = get_event_data(data_p)
            _, ai_run_array_direction, ai_tel_id, ai_tel_type, ai_tel_x, ai_tel_y, ai_tel_z = get_array_data(data_p)

            LST_image_charge = LST_image_charge[1:]
            
            # get camera geometry & camera pixels coordinates
            geom = CameraGeometry.from_name("LSTCam")
            points = np.array([np.array(geom.pix_x / u.m), np.array(geom.pix_y / u.m)]).T
            
            grid_x, grid_y = np.mgrid[-1.25:1.25:100j, -1.25:1.25:100j]

            # define the final array that will contain the interpolated images
            LST_image_charge_interp = np.zeros((len(LST_image_charge), img_rows, img_cols))

            for i in range(0, len(LST_image_charge)):
                LST_image_charge_interp[i] = griddata(points, LST_image_charge[i], (grid_x, grid_y), fill_value=0, method='cubic')

            data_p.close()

            print("Writing file: " + f[:-3] + '_interp.h5')

            data_file = h5py.File(f[:-3] + '_interp.h5', 'w')

            data_file.create_dataset('Array_Info/ai_run_array_direction', data=np.array(ai_run_array_direction))
            data_file.create_dataset('Array_Info/ai_tel_id', data=np.array(ai_tel_id))
            data_file.create_dataset('Array_Info/ai_tel_type', data=np.array(ai_tel_type))
            data_file.create_dataset('Array_Info/ai_tel_x', data=np.array(ai_tel_x))
            data_file.create_dataset('Array_Info/ai_tel_y', data=np.array(ai_tel_y))
            data_file.create_dataset('Array_Info/ai_tel_z', data=np.array(ai_tel_z))
            
            data_file.create_dataset('Event_Info/ei_alt', data=np.array(ei_alt))
            data_file.create_dataset('Event_Info/ei_az', data=np.array(ei_az))
            data_file.create_dataset('Event_Info/ei_core_x', data=np.array(ei_core_x))
            data_file.create_dataset('Event_Info/ei_core_y', data=np.array(ei_core_y))
            data_file.create_dataset('Event_Info/ei_event_number', data=np.array(ei_event_number))
            data_file.create_dataset('Event_Info/ei_h_first_int', data=np.array(ei_h_first_int))
            data_file.create_dataset('Event_Info/ei_mc_energy', data=np.array(ei_mc_energy))
            data_file.create_dataset('Event_Info/ei_particle_id', data=np.array(ei_particle_id))
            data_file.create_dataset('Event_Info/ei_run_number', data=np.array(ei_run_number))
            data_file.create_dataset('Event_Info/ei_LST_indices', data=np.array(ei_LST_indices))
           

            data_file.create_dataset('LST_image_charge_interp', data=np.array(LST_image_charge_interp))
            data_file.close()
        
        except HDF5ExtError:
            print('\nUnable to open file' + f)
            print('Removing it...')
            remove(f)

def chunkit(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out

if __name__ == '__main__':

    ncpus = mp.cpu_count()
    print("\nNumber of CPUs: " + str(ncpus))
    
    # get all the parameters given by the command line
    folders = sys.argv[1:]

    print('Folders: ' + str(folders) + '\n')

    # create a single big list containing the paths of all the files
    all_files = []
    
    for path in folders:
        files = [join(path, f) for f in listdir(path) if (isfile(join(path, f)) and f.endswith(".h5"))]
        all_files = all_files + files

    print('Files: ' + '\n' + str(all_files) + '\n')

    num_files = len(all_files)

    if ncpus >= num_files:
        print('ncpus >= num_files')
        for f in all_files:
            Process(target=func, args=([f],)).start()
    else:
        print('ncpus < num_files')
        c = chunkit(all_files, ncpus)
        for f in c:
            Process(target=func, args=(f,)).start()