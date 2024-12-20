# %%
import os
import numpy as np
import pandas as pd
import json
import xarray as xr


# 'netcdf4', 'h5netcdf', 'scipy'
# %%
def M9(information):
    """
    the default is to select sites from all M9 sites, but
    grid type (options: A, B, C, D, E, Y, and Z, can be empty)
    (ref: https://sites.uw.edu/pnet/m9-simulations/about-m9-simulations/extent-of-model/)
    """

    LocationFlag = information['LocationFlag']
    numSiteGM = information['number_of_realizations']
    grid_type = information[
        'grid_type'
    ]  # grid type (options: A, B, C, D, E, Y, and Z, can be "all")

    randomFLag = True  # if True, the realizations are selected randomly, otherwise, the first numSiteGM sites are selected
    maxnumSiteGM = 30
    numSiteGM = min(numSiteGM, maxnumSiteGM)  # number of realizations

    my_Directory = os.getcwd()
    print(f'Current Directory: {my_Directory}')

    files = [f for f in os.listdir() if os.path.isfile(f)]
    print(files)

    # changing realizations order
    # indicies = list(range(maxnumSiteGM));
    Realizations = ['032']
    indicies = [0]
    if randomFLag:
        np.random.shuffle(indicies)
    indicies = indicies[:numSiteGM]

    username = os.getlogin()
    print(f'Username: {username}')

    M9Path = f'/home/{username}/work/projects/PRJ-4603'
    # M9Path = "/home/parduino/work/projects/PRJ-4603"

    # M9Path = "/home/jovyan/work/projects/PRJ-4603"
    # M9Path = "tapis://project-4127798437512801810-242ac118-0001-012"
    # M9Path = f{"tapis://project-"+_UserProjects"
    # M9Path = "/data/projects/PRJ-4603"

    print('M9PATH defined')

    directory = './Events'  # directory to save the data
    # create the directory if it does not exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    print('Checking if the directory is created')
    all_entries = os.listdir()
    print(all_entries)

    gdf = pd.read_csv('selectedSites.csv', index_col=0)
    APIFLAG = information[
        'APIFLAG'
    ]  # if the APIFLAG is True, we use M9 API to get the motion data

    if not (APIFLAG):
        for i in indicies:
            for _, site in gdf.iterrows():
                # find the first Letter of the site name
                site_name = site['Station Name']
                lat = site['Latitude']
                lon = site['Longitude']
                firstLetter = site_name[0]
                filename = f'{M9Path}/csz{Realizations[i]}/{firstLetter}/Xarray.nc'

                # reading the nc file
                data = xr.open_dataset(filename)
                subset = data.sel(lat=lat, lon=lon, method='nearest')
                dt = data.coords['time'].values
                dt = dt[1] - dt[0]
                sitedata = {
                    'dT': dt,
                    'accel_x': subset['acc_x'].values.tolist(),
                    'accel_y': subset['acc_y'].values.tolist(),
                    'accel_z': subset['acc_z'].values.tolist(),
                }
                write_motion(site_name, directory, i, sitedata, APIFLAG)
                gdf['filename'] = f'{site_name}_{i}'
                if LocationFlag:
                    break

    if LocationFlag:
        gdf = gdf.loc[[0]]

    # save the gdf to a csv file in the directory just "Station Name", "Latitude", "Longitude"
    gdf[['filename', 'Latitude', 'Longitude']].to_csv(
        f'{directory}/sites.csv', index=False
    )

    print('Script is done')


def write_motion(site_name, directory, i, motiondict, APIFLAG):
    filename = f'{directory}/{site_name}_{i}.json'

    print(f'Writing {filename}')

    if APIFLAG:
        accel_x = 'AccelerationHistory-EW'
        accel_y = 'AccelerationHistory-NS'
        accel_z = 'AccelerationHistory-Vert'
        dt = 'TimeStep'
        datatowrite = {
            'Data': 'Time history generated using M9 simulations',
            'dT': motiondict[dt],
            'name': f'{site_name}_{i}',
            'numSteps': len(motiondict[accel_x]),
            'accel_x': motiondict[accel_x],
            'accel_y': motiondict[accel_y],
            'accel_z': motiondict[accel_z],
        }
    else:
        datatowrite = motiondict
        datatowrite['Data'] = 'Time history generated using M9 simulations'
        datatowrite['name'] = f'{site_name}_{i}'

    with open(filename, 'w') as f:
        json.dump(datatowrite, f, indent=2)


if __name__ == '__main__':
    # change the directory to the directory of the current file
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    with open('information.json', 'r') as file:
        information = json.load(file)
    M9(information)
