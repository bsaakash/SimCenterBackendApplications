#  # noqa: INP001, D100
# Copyright (c) 2018 Leland Stanford Junior University
# Copyright (c) 2018 The Regents of the University of California
#
# This file is part of the SimCenter Backend Applications
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# You should have received a copy of the BSD 3-Clause License along with
# this file. If not, see <http://www.opensource.org/licenses/>.
#
# Contributors:
# Kuanshi Zhong
# Ajay B Harish
# Frank Mckenna
# Sanjay Govindjee
#
# Special thanks to the original authors Snaiki and Wu for
# sharing the Matlab scripts
# Reference:
# 1. Snaiki, R. and Wu, T. (2017). Modeling tropical cyclone boundary layer: Height-
# resolving pressure and wind fields. Journal of Wind Engineering and Industrial
# Aerodynamics, 170, pp. 18-27.
# 2. Snaiki, R. and Wu, T. (2017). A linear height-resolving wind field model for
# tropical cyclone boundary layer. Journal of Wind Engineering and Industrial
# Aerodynamics, 171, pp. 248-260.

from itertools import starmap

import numpy as np
from shapely.geometry import Point, Polygon


class LinearAnalyticalModel_SnaikiWu_2017:  # noqa: D101
    def __init__(self, cyclone_param=[], storm_track=[]):  # noqa: B006
        """__init__: initializing the tropical cyclone
        cyclone_param: 6-dimensional array
        - cyclone_param[0]: landfall Latitude
        - cyclone_param[1]: landfall Longitude
        - cyclone_param[2]: landfall angle (degree)
        - cyclone_param[3]: central pressure different (hPa)
        - cyclone_param[4]: moving speed (km/h)
        - cyclone_param[5]: cyclone radius of the maximum winnds (km)
        storm_track:
        - storm_track['Latitude']: latitude values of the storm track
        - storm_track['Longittude']: longitude values of the storm track
        """  # noqa: D205, D400
        # constants
        self.R = 6371.0 * 1e3
        self.EDDY_VISCOCITY = 75.0
        self.AIR_DENSITY = 1.1
        #self.EDDY_VISCOCITY = 100.0 # To reproduce the paper
        #self.AIR_DENSITY = 1.2      # To reproduce the paper
        self.RA = 180.0 / np.pi
        self.EPS = np.spacing(1)

        # saving cyclone parameters
        try:
            self.landfall_lat = cyclone_param[0]
            self.landfall_lon = cyclone_param[1]
            self.landfall_ang = cyclone_param[2]
            self.cyclone_pres = cyclone_param[3] * 100.0
            self.cyclone_sped = cyclone_param[4] * 1000.0 / 3600.0
            self.cyclone_radi = cyclone_param[5]
            self.cyclone_radm = self.cyclone_radi * 1000.0
            self.Holland_B = (
                1.38
                + 0.00184 * self.cyclone_pres / 100.0
                - 0.00309 * self.cyclone_radi
            )
        except:  # noqa: E722
            print('WindFieldSimulaiton: please check the cyclone_param input.')  # noqa: T201

        # saving storm track data
        try:
            self.track_lat = storm_track['Latitude']
            self.track_lon = storm_track['Longitude']
            if len(self.track_lat) != len(self.track_lon):
                print(  # noqa: T201
                    'WindFieldSimulation: warning - storm track Latitude and Longitude sizes are different, data truncated.'
                )
                self.track_lat = self.track_lat[
                    0 : int(min(len(self.track_lat), len(self.track_lon)))
                ]
                self.track_lon = self.track_lon[
                    0 : int(min(len(self.track_lat), len(self.track_lon)))
                ]
        except:  # noqa: E722
            print('WindFieldSimulaiton: please check the strom_track input.')  # noqa: T201

        # initiation
        self.station_num = 0
        self.station = {
            'Latitude': [],
            'Longitude': [],
            'z0': [],
            'PWS': {'height': [], 'duration': 600.0, 'windspeed': []},
        }
        self.terrain_num = 0
        self.terrain_poly = []
        self.terrain_z0 = []
        self.delta_path = np.zeros(3)
        self.r = []
        self.theta = []
        self.zp = []
        self.mesh_info = []

    def set_delta_path(self, delta_path):
        """set_delta_path: perturbing the path coordinates and heading angle of the storm track"""  # noqa: D400
        if len(delta_path) == 3:  # noqa: PLR2004
            self.delta_path = delta_path
        else:
            print(  # noqa: T201
                'WindFieldSimulation: the delta_path should have a size of 3, default delta_path used.'
            )

    def set_delta_feat(self, delta_feat):
        """set_delta_feat: perturbing the central pressure difference, traslational speed, and max-wind-speed radius"""  # noqa: D400
        if len(delta_feat) == 3:  # noqa: PLR2004
            self.cyclone_pres = delta_feat[0] * 100.0
            self.cyclone_sped = delta_feat[1] * 1000.0 / 3600.0
            self.cyclone_radi = delta_feat[2]
            self.cyclone_radm = self.cyclone_radi * 1000.0
            self.Holland_B = (
                1.38
                + 0.00184 * self.cyclone_pres / 100.0
                - 0.00309 * self.cyclone_radi
            )
        else:
            print(  # noqa: T201
                'WindFieldSimulation: the delta_feat should have a size of 3, default delta_feat used.'
            )

    def __interp_z0(self, lat, lon):
        """__interp_z0: finding the z0 at (lat, lon) by interpolating reference terrain polygons"""  # noqa: D400
        z0 = []
        if not self.terrain_z0:
            # no reference terrain provided, using default reference z0 = 0.03
            z0 = 0.03
        else:
            # pt = Point(lat, lon)
            pt = Point(lon, lat)
            for p, z in zip(self.terrain_poly, self.terrain_z0):
                if pt.within(p):
                    z0 = z
            if not z0:
                z0 = 0.01
        # return
        return z0  # noqa: DOC201, RUF100

    def add_reference_terrain(self, terrain_info):
        """add_reference_terrainL specifying reference z0 values for a set of polygons
        terrain_info: geojson formatted polygon and z0 data
        """  # noqa: D205, D400
        for p in terrain_info['features']:
            if p['geometry']['type'] == 'Polygon':
                # creating a new polygon
                new_poly = Polygon(p['geometry']['coordinates'][0])
                self.terrain_poly.append(new_poly)
                self.terrain_z0.append(p['properties']['z0'])
                self.terrain_num += 1

    def set_cyclone_mesh(self, mesh_info):
        """set_cyclone_meesh: meshing the cyclone in radius and cycle
        mesh_info[0]: interval R
        mesh_info[1]: interval delta_R
        mesh_info[2]: external R
        mesh_info[3]: starting angle (usually 0)
        mesh_info[4]: interval angle
        mesh_info[5]: ending angle (usually 360)
        """  # noqa: D205, D400
        try:
            self.mesh_info = mesh_info
            self.r = np.arange(
                mesh_info[0], mesh_info[2] + mesh_info[1], mesh_info[1]
            )
            self.theta = np.arange(
                mesh_info[3], mesh_info[5] + mesh_info[4], mesh_info[4]
            )
            print('WindFieldSimulation: cyclone meshed.')  # noqa: T201
        except:  # noqa: E722
            print('WindFieldSimulation: input format error in set_cyclone_mesh.')  # noqa: T201

    def set_track_mesh(self, mesh_lat):
        """set_track_meesh: meshing the storm track
        mesh_lat[0]: starting latitude value of the meshed track
        mesh_lat[1]: interval latitude value
        mesh_lat[2]: ending latitude value of the meshed track
        """  # noqa: D205, D400
        try:
            lat0 = mesh_lat[0]
            dlat = mesh_lat[1]
            lat1 = mesh_lat[2]
        except:  # noqa: E722
            print('WindFieldSimulation: input format error in set_track_mesh.')  # noqa: T201

        # boundary checks
        if (max(lat0, lat1) > max(self.track_lat)) or (
            min(lat0, lat1) < min(self.track_lat)
        ):
            print(  # noqa: T201
                'WindFieldSimulation: warning - forcing the track mesh consistent with the original track boundary.'
            )
            lat0 = min(lat0, max(self.track_lat))
            lat1 = min(lat1, max(self.track_lat))
            lat0 = max(lat0, min(self.track_lat))
            lat1 = max(lat1, min(self.track_lat))

        # computing meshed track's Latitude and Longitude values
        self.track_lat_m = np.arange(lat0, lat1, dlat).tolist()
        self.track_lon_m = np.abs(
            np.interp(self.track_lat_m, self.track_lat, self.track_lon)
        )
        print('WindFieldSimulation: track meshed.')  # noqa: T201

    def define_track(self, track_lat):
        """set_track_meesh: meshing the storm track
        mesh_lat[0]: starting latitude value of the meshed track
        mesh_lat[1]: interval latitude value
        mesh_lat[2]: ending latitude value of the meshed track
        """  # noqa: D205, D400
        # computing meshed track's Latitude and Longitude values
        self.track_lat_m = track_lat
        self.track_lon_m = np.abs(
            np.interp(self.track_lat_m, self.track_lat, self.track_lon)
        )
        print('WindFieldSimulation: track defined.')  # noqa: T201

    def set_measure_height(self, measure_info):
        """set_measure_height: defining the height for calculating wind speed"""  # noqa: D400
        try:
            self.zp = np.arange(
                measure_info[0], measure_info[2] + measure_info[1], measure_info[1]
            ).tolist()
            print('WindFieldSimulation: measurement height defined.')  # noqa: T201
        except:  # noqa: E722
            print('WindFieldSimulation: input format error in set_measure_height.')  # noqa: T201

    def add_stations(self, station_list):
        """add_stations: adding stations to the model
        station_list:
        - station_list['Latitude']: latitude values of stations
        - station_list['Longitude']: longitude values of stations
        - station_list['z0']: surface roughness (optional)
        """  # noqa: D205, D400
        # z0 default
        if 'z0' not in station_list.keys():  # noqa: SIM118
            # default value = 0 (no specified z0)
            station_list['z0'] = np.zeros(len(station_list['Latitude']))

        # adding stations (without duplication)
        for lat, lon, z0 in zip(
            station_list['Latitude'], station_list['Longitude'], station_list['z0']
        ):
            self.station['Latitude'].append(lat)
            self.station['Longitude'].append(lon)
            if z0 == 0:
                # interpolating z0 from terrain feature
                self.station['z0'].append(self.__interp_z0(lat, lon))
            else:
                self.station['z0'].append(z0)
            # updating station number
            self.station_num += 1

    def __calculate_heading(self):
        """__calculate_heading: computing the heading path"""  # noqa: D400
        self.beta_c = np.zeros(len(self.track_lat_m))
        for i in range(len(self.track_lat_m) - 1):
            Delta = self.track_lon_m[i + 1] - self.track_lon_m[i] + self.EPS**2  # noqa: N806
            self.beta_c[i] = (
                -self.delta_path[2]
                + 90.0
                + self.RA
                * np.arctan2(
                    np.sin(Delta / self.RA)
                    * np.cos(self.track_lat_m[i + 1] / self.RA),
                    np.cos(self.track_lat_m[i] / self.RA)
                    * np.sin(self.track_lat_m[i + 1] / self.RA)
                    - np.sin(self.track_lat_m[i] / self.RA)
                    * np.cos(self.track_lat_m[i + 1] / self.RA)
                    * np.cos(Delta / self.RA),
                )
            )
        # positive angle values for beta_c
        self.beta_c = [x if x >= 0 else x + 360.0 for x in self.beta_c]
        # fixing the last value
        self.beta_c[-1] = self.beta_c[-2]

    def compute_wind_field(self):
        """compute_wind_field: computing the peak wind speed (10-min gust duraiton)"""  # noqa: D400
        print('WindFieldSimulation: running linear analytical model.')  # noqa: T201
        # checking if all parameters are defined

        # calculating heading
        self.__calculate_heading()

        # initializing matrices
        station_lat = self.station['Latitude']
        station_lon = self.station['Longitude']
        station_umax = np.zeros((len(station_lat), len(self.zp)))
        u = np.zeros((len(self.theta), len(self.r), len(self.zp)))
        v = np.zeros((len(self.theta), len(self.r), len(self.zp)))
        vg1 = np.zeros((len(self.theta), len(self.r)))
        z0 = np.zeros(len(self.r))
        # looping over different storm cyclone locations
        for i in range(len(self.track_lat_m)):
            # location and heading
            lat = self.track_lat_m[i] + self.delta_path[0]
            lon = self.track_lon_m[i] - 0.3 * self.delta_path[1]
            beta = self.beta_c[i]
            # coriolis
            omega = 0.7292 * 1e-4
            f = 2.0 * omega * np.sin(lat * np.pi / 180.0)
            # looping over different polar coordinates theta
            for j in range(len(self.theta)):
                Ctheta = -self.cyclone_sped * np.sin(  # noqa: N806
                    (self.theta[j] - beta) / self.RA
                )
                if (self.theta[j] >= 0) and (self.theta[j] <= 90):  # noqa: PLR2004
                    THETA = 90.0 - self.theta[j]  # noqa: N806
                else:
                    THETA = 450 - self.theta[j]  # noqa: N806

                lat_t = self.RA * np.arcsin(
                    np.sin(lat / self.RA) * np.cos(self.r / self.R)
                    + np.cos(lat / self.RA)
                    * np.sin(self.r / self.R)
                    * np.cos(THETA / self.RA)
                )
                lon_t = lon + self.RA * np.arctan2(
                    np.sin(THETA / self.RA)
                    * np.sin(self.r / self.R)
                    * np.cos(lat / self.RA),
                    np.cos(self.r / self.R) - np.sin(lat / self.RA) * np.sin(lat_t),
                )
                # looping over different polar coordinates r
                for k in range(len(self.r)):
                    z0[k] = self.__interp_z0(lat_t[k], lon_t[k])
                # configuring coefficients
                z10 = 10.0
                A = 11.4  # noqa: N806
                h = A * z0**0.86
                d = 0.75 * h
                kappa = 0.40
                Cd = kappa**2 / (np.log((z10 + h - d) / z0)) ** 2  # noqa: N806
                der_p = (
                    self.Holland_B
                    * self.cyclone_radm**self.Holland_B
                    * self.cyclone_pres
                    * (self.r ** (-self.Holland_B - 1))
                    * np.exp(
                        -((self.cyclone_radm * self.r ** (-1.0)) ** self.Holland_B)
                    )
                )
                der_p_2 = (
                    -(self.Holland_B + 1) * (self.r ** (-1.0))
                    + self.Holland_B
                    * self.cyclone_radm**self.Holland_B
                    * (self.r ** (-self.Holland_B - 1))
                ) * der_p
                vg1[j, :] = (
                    0.5 * (Ctheta - f * self.r)
                    + (
                        (0.5 * (Ctheta - f * self.r)) ** 2.0
                        + (self.r / self.AIR_DENSITY) * der_p
                    )
                    ** 0.5
                )
                der_vg1_r = -0.5 * f + 0.5 * (
                    (
                        ((Ctheta - f * self.r) / 2.0) ** 2.0
                        + self.r / self.AIR_DENSITY * der_p
                    )
                    ** (-0.5)
                ) * (
                    -(Ctheta - f * self.r) * f / 2.0
                    + 1.0 / self.AIR_DENSITY * der_p
                    + 1.0 / self.AIR_DENSITY * self.r * der_p_2
                )
                der_vg1_theta = -self.cyclone_sped * np.cos(
                    (self.theta[j] - beta) / self.RA
                ) / 2.0 + 0.25 * self.cyclone_sped * np.cos(
                    (self.theta[j] - beta) / self.RA
                ) * (-Ctheta + f * self.r) * (
                    (0.5 * (Ctheta - f * self.r)) ** 2.0
                    + (self.r / self.AIR_DENSITY) * der_p
                ) ** (-0.5)
                BB = 1.0 / (2.0 * self.EDDY_VISCOCITY * self.r) * der_vg1_theta  # noqa: N806
                Eta = (  # noqa: N806
                    (0.5 * (Ctheta - f * self.r)) ** 2.0
                    + (self.r / self.AIR_DENSITY) * der_p
                ) ** 0.5
                ALPHA = (  # noqa: N806
                    1.0
                    / (2.0 * self.EDDY_VISCOCITY)
                    * (f + 2.0 * vg1[j, :] / self.r)
                )
                BETA = (  # noqa: N806
                    1.0
                    / (2.0 * self.EDDY_VISCOCITY)
                    * (f + vg1[j, :] / self.r + der_vg1_r)
                )
                GAMMA = 1.0 / (2.0 * self.EDDY_VISCOCITY) * vg1[j, :] / self.r  # noqa: N806
                ALPHA = np.array(  # noqa: N806
                    [complex(x, y) for x, y in zip(np.real(ALPHA), np.imag(ALPHA))]
                )
                BETA = np.array(  # noqa: N806
                    [complex(x, y) for x, y in zip(np.real(BETA), np.imag(BETA))]
                )
                XXX = -((ALPHA * BETA) ** 0.25)  # noqa: N806
                YYY = -((ALPHA * BETA) ** 0.25)  # noqa: N806
                PP_zero = np.array([complex(x, y) for x, y in zip(XXX, YYY)])  # noqa: N806
                PP_one = -complex(1, 1) * (  # noqa: N806
                    (GAMMA + np.sqrt(ALPHA * BETA) - BB) ** 0.5
                )
                PP_minus_one = -complex(1, 1) * (  # noqa: N806
                    (-GAMMA + np.sqrt(ALPHA * BETA) - BB) ** 0.5
                )
                X1 = (  # noqa: N806
                    PP_zero
                    + f * self.r * Cd / self.EDDY_VISCOCITY
                    - 2.0 * Eta * Cd / self.EDDY_VISCOCITY
                    - self.cyclone_sped**2.0
                    * Cd**2.0
                    / (
                        4.0
                        * self.EDDY_VISCOCITY**2.0
                        * (PP_one - np.conj(PP_minus_one))
                    )
                    + self.cyclone_sped**2.0
                    * Cd**2.0
                    / (
                        4.0
                        * self.EDDY_VISCOCITY**2.0
                        * (np.conj(PP_one) - PP_minus_one)
                    )
                )

                X2 = (  # noqa: N806
                    -np.conj(PP_zero)
                    - f * self.r * Cd / self.EDDY_VISCOCITY
                    + 2.0 * Eta * Cd / self.EDDY_VISCOCITY
                    - self.cyclone_sped**2.0
                    * Cd**2.0
                    / (
                        4.0
                        * self.EDDY_VISCOCITY**2.0
                        * (PP_one - np.conj(PP_minus_one))
                    )
                    + self.cyclone_sped**2.0
                    * Cd**2.0
                    / (
                        4.0
                        * self.EDDY_VISCOCITY**2.0
                        * (np.conj(PP_one) - PP_minus_one)
                    )
                )

                X3 = (  # noqa: N806
                    complex(0, -2)
                    * Cd
                    / self.EDDY_VISCOCITY
                    * (Eta - f * self.r / 2.0) ** 2.0
                )

                X4 = -(  # noqa: N806
                    -PP_zero
                    - f * self.r * Cd / (2.0 * self.EDDY_VISCOCITY)
                    + Eta * Cd / self.EDDY_VISCOCITY
                ) / (
                    -np.conj(PP_zero)
                    - f * self.r * Cd / (2.0 * self.EDDY_VISCOCITY)
                    + Eta * Cd / self.EDDY_VISCOCITY
                )

                A_zero = -X3 / (X1 + X2 * X4)  # noqa: N806
                A_one = (  # noqa: N806
                    complex(0, 1)
                    * self.cyclone_sped
                    * Cd
                    * np.exp(complex(0, -1) * beta)
                    / (4.0 * self.EDDY_VISCOCITY * (PP_one - np.conj(PP_minus_one)))
                    * (A_zero + np.conj(A_zero))
                )
                A_minus_one = -np.conj(A_one)  # noqa: N806
                # looping over different heights zp
                for ii in range(len(self.zp)):
                    u_zero = np.sqrt(ALPHA / BETA) * np.real(
                        A_zero * np.exp(PP_zero * self.zp[ii])
                    )
                    v_zero = np.imag(A_zero * np.exp(PP_zero * self.zp[ii]))
                    u_one = np.sqrt(ALPHA / BETA) * np.real(
                        A_one
                        * np.exp(
                            PP_one * self.zp[ii]
                            + complex(0, 1) * self.theta[j] / self.RA
                        )
                    )
                    u_minus_one = np.sqrt(ALPHA / BETA) * np.real(
                        A_minus_one
                        * np.exp(
                            PP_minus_one * self.zp[ii]
                            - complex(0, 1) * self.theta[j] / self.RA
                        )
                    )
                    v_one = np.imag(
                        A_one
                        * np.exp(
                            PP_one * self.zp[ii]
                            + complex(0, 1) * self.theta[j] / self.RA
                        )
                    )
                    v_minus_one = np.imag(
                        A_minus_one
                        * np.exp(
                            PP_minus_one * self.zp[ii]
                            - complex(0, 1) * self.theta[j] / self.RA
                        )
                    )
                    for tmptag in range(u.shape[1]):
                        u[j, tmptag, ii] = (
                            np.real(u_zero)[tmptag]
                            + np.real(u_one)[tmptag]
                            + np.real(u_minus_one)[tmptag]
                        )
                        v[j, tmptag, ii] = (
                            v_zero[tmptag] + v_one[tmptag] + v_minus_one[tmptag]
                        )

            # wind speed components
            v1 = v
            for m in range(v.shape[2]):
                v1[:, :, m] = v1[:, :, m] + vg1
            U = (v1**2.0 + u**2.0) ** 0.5  # noqa: N806

            # mapping to staitons
            dd = (
                np.arccos(
                    np.cos(np.array(station_lat) / self.RA)
                    * np.cos(lat / self.RA)
                    * np.cos((np.abs(np.array(station_lon)) - lon) / self.RA)
                    + np.sin(np.array(station_lat) / self.RA) * np.sin(lat / self.RA)
                )
                * 6371.0
                * 180.0
                / np.pi
                / self.RA
                * 1000.0
            )
            Delta = np.abs(np.array(station_lon)) - lon + self.EPS**2.0  # noqa: N806
            bearing = 90.0 + self.RA * np.arctan2(
                np.sin(Delta / self.RA) * np.cos(np.array(station_lat) / self.RA),
                np.cos(lat / self.RA) * np.sin(np.array(station_lat) / self.RA)
                - np.sin(lat / self.RA)
                * np.cos(np.array(station_lat) / self.RA)
                * np.cos(Delta / self.RA),
            )
            bearing = [x if x >= 0 else x + 360.0 for x in bearing]
            jj = [int(x / self.mesh_info[4]) for x in bearing]
            kk = [min(int(x / self.mesh_info[1]), len(self.r) - 1) for x in dd]
            for ii in range(len(self.zp)):
                tmp = U[:, :, ii].tolist()
                wind_speed = [tmp[jtag][ktag] for jtag, ktag in zip(jj, kk)]
                station_umax[:, ii] = [
                    max(x, y) for x, y in zip(wind_speed, station_umax[:, ii])
                ]

        # copying results
        self.station['PWS']['height'] = self.zp
        self.station['PWS']['windspeed'] = station_umax.tolist()
        print('WindFieldSimulation: linear analytical simulation completed.')  # noqa: T201

    def get_station_data(self):
        """get_station_data: returning station data"""  # noqa: D400
        # return station dictionary
        return self.station  # noqa: DOC201, RUF100
