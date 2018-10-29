import tables

if __name__ == "__main__":

    # data path
    data_path = '/Users/nicolamarinello/ctasoft/simulations/Paranal_proton_North_20deg_3HB9_DL1_ML1/proton_20deg_0deg_srun13316-33715___cta-prod3_desert-2150m-Paranal-HB9.h5'
    #data_path = '/Users/nicolamarinello/ctasoft/simulations/Paranal_gamma-diffuse_North_20deg_3HB9_DL1_ML1/gamma_20deg_0deg_srun5865-23126___cta-prod3_desert-2150m-Paranal-HB9_cone10.h5'
    data = tables.open_file(data_path)

    # acquire the data
    data_ainfo = data.root.Array_Info

    # array info data
    ai_run_array_direction = [x['run_array_direction']
                              for x in data_ainfo.iterrows()]
    ai_tel_id = [x['tel_id'] for x in data_ainfo.iterrows()]
    ai_tel_type = [x['tel_type'] for x in data_ainfo.iterrows()]
    ai_tel_x = [x['tel_x'] for x in data_ainfo.iterrows()]
    ai_tel_y = [x['tel_y'] for x in data_ainfo.iterrows()]
    ai_tel_z = [x['tel_z'] for x in data_ainfo.iterrows()]

    # plot the telescopes array
    tel_types = ['LST', 'MSTS', 'SSTC']

    arr_table = data_ainfo

    fig, ax = plt.subplots(figsize=(10, 10))

    for tel_type in tel_types:
        tel_x = [x['tel_x'] for x in arr_table.iterrows(
        ) if x['tel_type'] == tel_type.encode('ascii')]
        tel_y = [x['tel_y'] for x in arr_table.iterrows(
        ) if x['tel_type'] == tel_type.encode('ascii')]
        plt.scatter(tel_x, tel_y, label=tel_type)

    ax.legend()
    ax.grid()
    plt.show()

    # extract indexes of a specific telescope
    tel_type = 'LST'
    tel_ids = [x['tel_id'] for x in data_ainfo.iterrows(
    ) if x['tel_type'] == tel_type.encode('ascii')]
    print(tel_ids)
