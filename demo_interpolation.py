from ctapipe.image import toymodel, hillas_parameters, tailcuts_clean
from ctapipe.instrument import CameraGeometry
from ctapipe.visualization import CameraDisplay
from astropy import units as u
from scipy.interpolate import griddata
import tables
import matplotlib.pyplot as plt
import numpy as np
import time


if __name__ == "__main__":

    # data path
    #data_path = '/Users/nicolamarinello/ctasoft/simulations/Paranal_proton_North_20deg_3HB9_DL1_ML1/proton_20deg_0deg_srun13316-33715___cta-prod3_desert-2150m-Paranal-HB9.h5'
    data_path = '/Users/nicolamarinello/ctasoft/simulations/Paranal_gamma-diffuse_North_20deg_3HB9_DL1_ML1/gamma_20deg_0deg_srun5865-23126___cta-prod3_desert-2150m-Paranal-HB9_cone10.h5'
    data = tables.open_file(data_path)

    # acquire the data
    data_ainfo = data.root.Array_Info
    data_einfo = data.root.Event_Info
    data_LST = data.root.LST

    print(repr(data_ainfo))
    print(repr(data_einfo))
    print(repr(data_LST))

    # array info data
    ai_run_array_direction = [x['run_array_direction']
                              for x in data_ainfo.iterrows()]
    ai_tel_id = [x['tel_id'] for x in data_ainfo.iterrows()]
    ai_tel_type = [x['tel_type'] for x in data_ainfo.iterrows()]
    ai_tel_x = [x['tel_x'] for x in data_ainfo.iterrows()]
    ai_tel_y = [x['tel_y'] for x in data_ainfo.iterrows()]
    ai_tel_z = [x['tel_z'] for x in data_ainfo.iterrows()]

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

    # LST data
    LST_event_index = [x['event_index'] for x in data_LST.iterrows()]
    LST_image_charge = [x['image_charge'] for x in data_LST.iterrows()]
    LST_image_peak_times = [x['image_peak_times'] for x in data_LST.iterrows()]

    ###### D I S P L A Y  A L L  T H E  E V E N T S ######

    # Load the camera
    geom = CameraGeometry.from_name("LSTCam")
    fig = plt.figure(figsize=(12, 8))

    plt.style.use("ggplot")
    plt.show(block=False)

    for e_idx in range(0, len(data_einfo) - 1):

        # select a spcific event
        my_event = data_einfo[e_idx]
        print('Event number: {}'.format(my_event['event_number']))
        print('Energy: {} TeV'.format(my_event['mc_energy']))
        print('Alt: {} rad'.format(my_event['alt']))
        print('Az: {} rad'.format(my_event['az']))
        my_indices = my_event['LST_indices']
        print('LST_indices: ' + str(my_indices))

        for img_index in my_indices:
            if img_index > 0:
                # print event information
                img_charge = LST_image_charge[img_index]
                img_time = LST_image_peak_times[img_index]
                print(img_charge)
                print(img_time)

                fig.clear()
                plt.suptitle('EVENT')
                disps = []

                ax = plt.subplot(1, 2, 1)

                disp = CameraDisplay(geom, ax=ax, title="Real Camera Event")
                disp.add_colorbar()

                # Apply image cleaning
                cleanmask = tailcuts_clean(
                    geom, img_charge, picture_thresh=10, boundary_thresh=5
                )
                clean = img_charge.copy()
                clean[~cleanmask] = 0.0

                # Calculate image parameters
                #hillas = hillas_parameters(geom, clean)
                # print(hillas)

                # Show the camera image and overlay Hillas ellipse and clean
                # pixels
                disp.image = img_charge
                disp.cmap = 'inferno'
                disp.highlight_pixels(cleanmask, color='crimson')
                #disp.overlay_moments(hillas, color='cyan', linewidth=1)

                disps.append(disp)

                # interpolation

                points = np.array(
                    [np.array(geom.pix_x / u.m), np.array(geom.pix_y / u.m)]).T
                print(points.shape)
                values = np.array(img_charge)
                print(values.shape)

                grid_x, grid_y = np.mgrid[-1.25:1.25:100j, -1.25:1.25:100j]
                grid_z = griddata(
                    points, values, (grid_x, grid_y), method='cubic')

                grid_z = np.nan_to_num(grid_z)
                print(grid_z)

                ax = plt.subplot(1, 2, 2)

                inferno = plt.get_cmap('inferno')

                inter = ax.imshow(grid_z.T, extent=(-1.25, 1.25, -
                                                    1.25, 1.25), origin='lower', cmap=inferno)
                plt.colorbar(inter, ax=ax)
                #plt.gcf().set_size_inches(10, 10)
                # plt.savefig('foo.png')
                # plt.show()
                plt.pause(0.1)
