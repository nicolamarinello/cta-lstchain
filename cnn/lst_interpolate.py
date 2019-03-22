import argparse
import multiprocessing as mp
from multiprocessing import Process
from os import listdir, remove
from os.path import isfile, join

import h5py
import numpy as np
import tables
from astropy import units as u
from ctapipe.coordinates import NominalFrame, AltAz
from ctapipe.image import hillas_parameters, leakage
from ctapipe.image.cleaning import tailcuts_clean
from ctapipe.instrument import CameraGeometry
from scipy.interpolate import griddata
from tables.exceptions import HDF5ExtError, NoSuchNodeError

'''
usage: python lst_interpolate.py --dirs path/to/folder1 path/to/folder2 path/to/folder3 ... --rem_org 0 --rem_corr 0 --rem_nsnerr 0
'''


def get_array_data(data):
    data_ainfo = data.root.Array_Info

    # array info data
    ai_run_array_direction = [x['run_array_direction']
                              for x in data_ainfo.iterrows()]
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


def func(paths, format, ro, rc, rn):
    img_rows, img_cols = 100, 100

    # iterate on each proton file & concatenate charge arrays
    for f in paths:

        # get the data from the file
        try:
            data_p = tables.open_file(f)
            _, LST_event_index, LST_image_charge, LST_image_peak_times = get_LST_data(
                data_p)
            _, ei_alt, ei_az, ei_core_x, ei_core_y, ei_event_number, ei_h_first_int, ei_mc_energy, ei_particle_id, ei_run_number, ei_LST_indices = get_event_data(
                data_p)
            _, ai_run_array_direction, ai_tel_id, ai_tel_type, ai_tel_x, ai_tel_y, ai_tel_z = get_array_data(
                data_p)

            LST_event_index = LST_event_index[1:]
            LST_image_charge = LST_image_charge[1:]
            LST_image_peak_times = LST_image_peak_times[1:]

            # get camera geometry & camera pixels coordinates
            camera = CameraGeometry.from_name("LSTCam")
            points = np.array([np.array(camera.pix_x / u.m),
                               np.array(camera.pix_y / u.m)]).T

            grid_x, grid_y = np.mgrid[-1.25:1.25:100j, -1.25:1.25:100j]

            # define the final array that will contain the interpolated images
            # LST_image_charge_interp = np.zeros(
            #    (len(LST_image_charge), img_rows, img_cols))

            # alt az of the array
            az_array = ai_run_array_direction[0][0]
            alt_array = ai_run_array_direction[0][1]

            if alt_array > 90:
                alt_array = 90

            point = AltAz(alt=alt_array * u.rad, az=az_array * u.rad)

            lst_image_charge_interp = []
            lst_image_peak_times_interp = []
            delta_az = []
            delta_alt = []
            acc_idxs = []  # accepted indexes

            cleaning_level = {'LSTCam': (3.5, 7.5, 2)}

            for i in range(0, len(LST_image_charge)):

                image = LST_image_charge[i]
                time = LST_image_peak_times[i]

                boundary, picture, min_neighbors = cleaning_level['LSTCam']
                clean = tailcuts_clean(
                    camera,
                    image,
                    boundary_thresh=boundary,
                    picture_thresh=picture,
                    min_number_picture_neighbors=min_neighbors
                )

                if len(np.where(clean > 0)[0]) != 0:
                    hillas = hillas_parameters(camera[clean], image[clean])
                    intensity = hillas['intensity']

                    l = leakage(camera, image, clean)
                    leakage1_intensity = l['leakage1_intensity']

                    if intensity > 50 and leakage1_intensity < 0.2:
                        # cubic interpolation
                        interp_img = griddata(points, image, (grid_x, grid_y), fill_value=0, method='cubic')
                        interp_time = griddata(points, time, (grid_x, grid_y), fill_value=0, method='cubic')

                        # delta az, delta alt computation
                        az = ei_az[LST_event_index[i]]
                        alt = ei_alt[LST_event_index[i]]
                        src = AltAz(alt=alt * u.rad, az=az * u.rad)
                        source_direction = src.transform_to(NominalFrame(origin=point))

                        # appending to arrays
                        lst_image_charge_interp.append(interp_img)
                        lst_image_peak_times_interp.append(interp_time)
                        delta_az.append(source_direction.delta_az.deg)
                        delta_alt.append(source_direction.delta_alt.deg)

                        acc_idxs += [i]

            lst_image_charge_interp = np.array(lst_image_charge_interp)

            data_p.close()

            if format == 'hdf5':

                filename = f[:-3] + '_interp.h5'

                print("Writing file: " + filename)

                data_file = h5py.File(filename, 'w')

                # data_file.create_dataset(
                #    'Array_Info/ai_run_array_direction', data=np.array(ai_run_array_direction))
                # data_file.create_dataset(
                #    'Array_Info/ai_tel_id', data=np.array(ai_tel_id))
                # data_file.create_dataset(
                #    'Array_Info/ai_tel_type', data=np.array(ai_tel_type))
                # data_file.create_dataset(
                #    'Array_Info/ai_tel_x', data=np.array(ai_tel_x))
                # data_file.create_dataset(
                #    'Array_Info/ai_tel_y', data=np.array(ai_tel_y))
                # data_file.create_dataset(
                #    'Array_Info/ai_tel_z', data=np.array(ai_tel_z))

                data_file.create_dataset(
                    'Event_Info/ei_alt', data=np.array(ei_alt))
                data_file.create_dataset('Event_Info/ei_az', data=np.array(ei_az))
                # data_file.create_dataset(
                #    'Event_Info/ei_core_x', data=np.array(ei_core_x))
                # data_file.create_dataset(
                #    'Event_Info/ei_core_y', data=np.array(ei_core_y))
                # data_file.create_dataset(
                #    'Event_Info/ei_event_number', data=np.array(ei_event_number))
                # data_file.create_dataset(
                #    'Event_Info/ei_h_first_int', data=np.array(ei_h_first_int))
                data_file.create_dataset(
                    'Event_Info/ei_mc_energy', data=np.array(ei_mc_energy))
                # data_file.create_dataset(
                #    'Event_Info/ei_particle_id', data=np.array(ei_particle_id))
                # data_file.create_dataset(
                #    'Event_Info/ei_run_number', data=np.array(ei_run_number))
                # data_file.create_dataset(
                #    'Event_Info/ei_LST_indices', data=np.array(ei_LST_indices))

                data_file.create_dataset(
                    'LST/LST_event_index', data=np.array(LST_event_index)[acc_idxs])
                data_file.create_dataset(
                    'LST/LST_image_charge', data=np.array(LST_image_charge)[acc_idxs])
                data_file.create_dataset(
                    'LST/LST_image_peak_times', data=np.array(LST_image_peak_times)[acc_idxs])
                data_file.create_dataset(
                    'LST/LST_image_charge_interp', data=np.array(lst_image_charge_interp))
                data_file.create_dataset(
                    'LST/LST_image_peak_times_interp', data=np.array(lst_image_peak_times_interp))
                data_file.create_dataset(
                    'LST/delta_alt', data=np.array(delta_alt))
                data_file.create_dataset(
                    'LST/delta_az', data=np.array(delta_az))
                data_file.close()

                # in the interpolated files there will be all the original events
                # but for the LST only the ones actually see at least from one LST (as in the original files)
                # and that are above thresholds cuts

            elif format == 'npz':

                filename = f[:-3] + '_interp.npz'

                print("Writing file: " + filename)

                np.savez(filename, ei_az=np.array(ei_az), ei_mc_energy=np.array(ei_mc_energy),
                         LST_event_index=np.array(LST_event_index)[acc_idxs],
                         LST_image_charge=np.array(LST_image_charge)[acc_idxs],
                         LST_image_peak_times=np.array(LST_image_peak_times)[acc_idxs],
                         LST_image_charge_interp=np.array(lst_image_charge_interp),
                         LST_image_peak_times_interp=np.array(lst_image_peak_times_interp))

            if ro == '1':
                remove(f)
                print('Removing original file')

        except HDF5ExtError:

            print('\nUnable to open file' + f)

            if rc == '1':
                print('Removing it...')
                remove(f)

        except NoSuchNodeError:

            print('This file has a problem with the data structure: ' + f)

            if rn == '1':
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

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dirs', type=str, default='', nargs='+', help='Folder that contain .h5 files.', required=True)
    parser.add_argument(
        '--format', type=str, default='hdf5', help='Choose format.', required=False)
    parser.add_argument(
        '--rem_org', type=str, default='0', help='Select 1 to remove the original files.', required=False)
    parser.add_argument(
        '--rem_corr', type=str, default='0', help='Select 1 to remove corrupted files.', required=False)
    parser.add_argument(
        '--rem_nsnerr', type=str, default='0', help='Select 1 to remove files that raise NoSuchNodeError exception.',
        required=False)

    FLAGS, unparsed = parser.parse_known_args()

    print(FLAGS.dirs)

    format = FLAGS.format

    ncpus = mp.cpu_count()
    print("\nNumber of CPUs: " + str(ncpus))

    # get all the parameters given by the command line
    folders = FLAGS.dirs

    print('Folders: ' + str(folders) + '\n')

    # create a single big list containing the paths of all the files
    all_files = []

    for path in folders:
        files = [join(path, f) for f in listdir(path) if (
                isfile(join(path, f)) and f.endswith(".h5"))]
        all_files = all_files + files

    # print('Files: ' + '\n' + str(all_files) + '\n')

    num_files = len(all_files)

    processes = []

    if ncpus >= num_files:
        print('ncpus >= num_files')
        for f in all_files:
            p = Process(target=func, args=([f], format, FLAGS.rem_org, FLAGS.rem_corr, FLAGS.rem_nsnerr))
            p.start()
            processes.append(p)
    else:
        print('ncpus < num_files')
        c = chunkit(all_files, ncpus)
        for f in c:
            p = Process(target=func, args=(f, format, FLAGS.rem_org, FLAGS.rem_corr, FLAGS.rem_nsnerr))
            p.start()
            processes.append(p)

    for p in processes:
        p.join()
