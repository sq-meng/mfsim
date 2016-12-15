import numpy as np
from scipy import interpolate
import configparser
import json
import csv
import re
import pandas as pd
from collections import defaultdict
import time

#
# The core of MultiFLEXX tools, handles coordinate system conversions.
# Please use right-hand convention when defining scattering plane with hkl1 and hkl2. The opposite case can lead to
# unexpected behaviour.
#

np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
_fmt_8_3 = '{: >8.3f}'.format
_fmt_2 = '{:.2f}'.format


def elegant_float(num):
    try:
        return float(num)
    except ValueError:
        return np.NaN


def e_to_k(e):
    return np.sqrt(e) * 0.6947


def k_to_e(k):
    return (k / 0.6947) ** 2


class UBMatrix(object):
    def __init__(self, *args, config=None):
        if config is None:
            self._latparam, self._hkl1, self._hkl2, self._plot_x, self._plot_y = args
        else:
            self._latparam = config.sample['latparam']
            self._hkl1 = config.alignment['hkl1']
            self._hkl2 = config.alignment['hkl2']
            self._plot_x = config.plot['x']
            self._plot_y = config.plot['y']
        self.conversion_matrices = {}
        self.conversion_matrices['bl'] = self._calculate_bl()
        self.conversion_matrices['rl'] = self._calculate_rl()
        self.conversion_matrices['sl'] = self._calculate_sl()
        self.conversion_matrices['pl'] = self._calculate_pl()
        self.conversion_matrices['ll'] = np.diag([1, 1, 1])
        self.figure_aspect = self._calculate_aspect()

    def _calculate_bl(self):
        lattice_parameters = self._latparam
        a, b, c, alpha_deg, beta_deg, gamma_deg = lattice_parameters

        cos_alpha = np.cos(np.radians(alpha_deg))
        cos_beta = np.cos(np.radians(beta_deg))
        cos_gamma = np.cos(np.radians(gamma_deg))
        sin_gamma = np.sin(np.radians(gamma_deg))
        tan_gamma = np.tan(np.radians(gamma_deg))

        ax = a
        a_l = np.array([[ax], [0], [0]])

        bx = b * cos_gamma
        by = b * sin_gamma
        b_l = np.array([[bx], [by], [0]])

        cx = c * cos_beta
        cy = c * (cos_alpha / sin_gamma - cos_beta / tan_gamma)
        cz = c * np.sqrt(1 - (cx ** 2 + cy ** 2) / c ** 2)
        c_l = np.array([[cx], [cy], [cz]])

        b_in_l = np.concatenate((a_l, b_l, c_l), axis=1)

        return b_in_l

    def _calculate_rl(self):
        r_in_l = (2 * np.pi * np.linalg.inv(self.conversion_matrices['bl'])).T
        return r_in_l

    def _calculate_sl(self):
        hkl1_l = np.dot(self.conversion_matrices['rl'], self._hkl1)
        hkl2_l = np.dot(self.conversion_matrices['rl'], self._hkl2)

        hkl1_cross_hkl2 = np.cross(hkl1_l, hkl2_l)
        sz_in_l = hkl1_cross_hkl2 / np.linalg.norm(hkl1_cross_hkl2)
        sx_in_l = hkl1_l / np.linalg.norm(hkl1_l)
        sy_in_l = np.cross(sz_in_l, sx_in_l)

        s_in_l = np.array([sx_in_l, sy_in_l, sz_in_l]).T
        return s_in_l

    def _calculate_pl(self):
        px_in_l = np.dot(self.conversion_matrices['rl'], self._plot_x)
        py_in_l = np.dot(self.conversion_matrices['rl'], self._plot_y)
        pz_in_l = np.cross(px_in_l, py_in_l)  # for completeness

        p_in_l = np.array([px_in_l, py_in_l, pz_in_l]).T
        return p_in_l

    def _calculate_aspect(self):
        plot_x_r = self._plot_x
        plot_y_r = self._plot_y
        plot_x_unit_len = np.linalg.norm(self.convert(plot_x_r, 'rs'))
        plot_y_unit_len = np.linalg.norm(self.convert(plot_y_r, 'rs'))
        return plot_y_unit_len / plot_x_unit_len

    def convert(self, coord, sys):
        try:
            conversion_matrix = self.conversion_matrices['sys']
        except KeyError:
            source_system = sys[0]
            target_system = sys[1]
            source_to_l_matrix = self.conversion_matrices[source_system + 'l']
            target_to_l_matrix = self.conversion_matrices[target_system + 'l']
            conversion_matrix = np.dot(np.linalg.inv(target_to_l_matrix), source_to_l_matrix)
            self.conversion_matrices[sys] = conversion_matrix
            self.conversion_matrices[sys[::-1]] = np.linalg.inv(conversion_matrix)
        finally:
            pass

        return np.dot(conversion_matrix, coord)

    def __repr__(self):
        lattice_parameters = self._latparam
        a, b, c, alpha_deg, beta_deg, gamma_deg = lattice_parameters
        f = _fmt_8_3
        lattice_parameters = 'Lattice parameters: ' + f(a) + f(b) + f(c) + f(alpha_deg) + f(beta_deg) + f(gamma_deg) \
                             + '\n'
        b_to_l = 'lattice b to l: \n' + str(self.conversion_matrices['bl'])
        return lattice_parameters + b_to_l


def find_triangle_angles(a, b, c):
    #
    # Receives side lengths of triangle and calculate corresponding angles.
    #
    aa = (b ** 2 + c ** 2 - a ** 2) / (2 * b * c)
    bb = (a ** 2 + c ** 2 - b ** 2) / (2 * a * c)
    cc = (a ** 2 + b ** 2 - c ** 2) / (2 * a * b)
    if abs(aa) > 1 or abs(bb) > 1 or abs(cc) > 1:
        raise ValueError('Scattering triangle cannot close.')
    alpha = np.arccos(aa)
    beta = np.arccos(bb)
    gamma = np.arccos(cc)

    return alpha, beta, gamma


def load_config(config_parser):
    sample = {}
    sample['latparam'] = json.loads(config_parser.get('Sample', 'latparam'))

    alignment = {}
    alignment['hkl1'] = json.loads(config_parser.get('Alignment', 'hkl1'))
    alignment['hkl2'] = json.loads(config_parser.get('Alignment', 'hkl2'))

    instrument = {}
    instrument['ki'] = config_parser.getfloat('Instrument', 'ki')
    instrument['ei'] = (instrument['ki'] / 0.6947) ** 2

    instrument['ef'] = json.loads(config_parser.get('Instrument', 'ef'))
    instrument['kf'] = [0.6947 * np.sqrt(x) for x in instrument['ef']]
    instrument['channels'] = config_parser.getint('Instrument', 'channels')
    instrument['channel_offset'] = config_parser.getfloat('Instrument', 'channel_offset')
    coeff = config_parser.getfloat('Instrument', 'angular_coverage_coefficient')
    instrument['angular_coverage'] = [x * coeff for x in
                                      json.loads(config_parser.get('Instrument', 'angular_coverage'))]
    instrument['intensity_coeff'] = config_parser.get('Instrument', 'intensity_coeff')
    instrument['channel_weights'] = config_parser.get('Instrument', 'channel_weights')
    instrument['detector_enable'] = config_parser.get('Instrument', 'detector_enable')

    scan = {}
    scan['A3start'] = config_parser.getfloat('Scan', 'A3start')
    scan['A3end'] = config_parser.getfloat('Scan', 'A3end')
    scan['A4start'] = config_parser.getfloat('Scan', 'A4start')
    scan['A4end'] = config_parser.getfloat('Scan', 'A4end')

    plot = {}
    plot['x'] = json.loads(config_parser.get('Plot', 'x'))
    plot['y'] = json.loads(config_parser.get('Plot', 'y'))
    plot['xlabel'] = config_parser.get('Plot', 'xlabel')
    plot['ylabel'] = config_parser.get('Plot', 'ylabel')

    horizontal_magnet = {}
    horizontal_magnet['magnet_ident'] = config_parser.get('Horizontal_magnet', 'magnet_ident')
    horizontal_magnet['north_along'] = json.loads(config_parser.get('Horizontal_magnet', 'north_along'))
    horizontal_magnet['sample_stick_rotation'] = \
        config_parser.getfloat('Horizontal_magnet', 'sample_stick_rotation')
    return sample, alignment, instrument, scan, plot, horizontal_magnet


class Config(object):
    def __init__(self, config_file='MF.conf'):
        config_parser = configparser.ConfigParser()
        config_parser.read(config_file)
        sample, alignment, instrument, scan, plot, horizontal_magnet = load_config(config_parser)
        self.sample = sample
        self.alignment = alignment
        self.instrument = instrument
        self.scan = scan
        self.plot = plot
        self.horizontal_magnet = horizontal_magnet
        self.transmission = []
        self._load_magnet_transmission()

    def _load_magnet_transmission(self):
        ident = self.horizontal_magnet['magnet_ident']
        if ident == 'none':
            self.horizontal_magnet['exist'] = False
            return
        else:
            try:
                file = open(ident + '.csv')
                reader = csv.DictReader(file)
                theta = []
                transmission = []
                for row in reader:
                    theta.append(elegant_float(row['theta']))
                    transmission.append(elegant_float(row['transmission']))
                self.horizontal_magnet['exist'] = True
                print('Loaded HM transmission profile with ' + str(len(theta)) + ' entries.')
                self.transmission = interpolate.interp1d(np.radians(np.array(theta)), np.array(transmission))
            except FileNotFoundError:
                print('Error loading magnet transmission profile, assuming NO horizontal magnet.')
                self.horizontal_magnet['exist'] = False
            finally:
                pass


def angle_to_qins(ki, kf, a3, a4):
    try:
        length = min(len(a3), len(a4))
    except TypeError:
        length = 1
    a3 = np.array(a3).reshape(1, -1)
    a4 = np.array(a4).reshape(1, -1)
    ones = np.ones([1, length])
    zeros = np.zeros([1, length])
    initial_vectors = np.vstack([-ones, zeros, zeros])
    try:
        len_ki = len(ki)
        if len_ki == length:
            ki_in_s = np.array(ki) * initial_vectors
        else:
            ki_in_s = ki[0] * initial_vectors
    except TypeError:
        ki_in_s = ki * initial_vectors
    kf_in_s = kf * rotate_around_z(initial_vectors, np.radians(a4))
    q_in_s = rotate_around_z(kf_in_s - ki_in_s, - np.radians(a3))
    return q_in_s


def rotate_around_z(vectors, angles):
    #
    # Rotates provided vector around [0, 0, 1] for given degrees counterclockwise.
    #
    try:
        x, y, z = vectors[0, :], vectors[1, :], vectors[2, :]
    except IndexError:
        x, y, z = vectors[0],\
                  vectors[1], vectors[2]
    cosines = np.cos(angles)
    sines = np.sin(angles)
    return np.vstack([x * cosines - y * sines, x * sines + y * cosines, z])


def _parse_flatcone_line(line):
    data = np.array([int(x) for x in line.split()])
    array = np.reshape(data, (-1, 5))[0: -1, :]
    ang_channels = np.array([np.arange(1, 32)]).T
    array_with_ch_no = np.hstack([ang_channels, array])
    dataframe_flatcone = pd.DataFrame(data=array_with_ch_no, columns=['aCh', 'e1', 'e2', 'e3', 'e4', 'e5'])
    dataframe_flatcone.set_index('aCh', inplace=True)
    return dataframe_flatcone


def _parse_param_line(line):
    line_name = line[0:5]
    line_body = line[6:].strip()
    if line_name == 'COMND':
        no_points = int(re.findall('(?<=NP)[\s\t0-9]*', line_body)[0].strip())
        return line_name, {'value': line_body, 'NP': no_points}
    elif '=' not in line_body:
        return line_name, line_body
    else:
        equations = line_body.split(',')
        line_dict = {}
        for eq in equations:
            param_name, value_raw = [x.strip() for x in eq.split('=')]
            try:
                value = elegant_float(value_raw)
            except ValueError:
                value = value_raw
            line_dict[param_name] = value
        return line_name, line_dict


def parse_ill_data(f, start_flag='DATA_:\n'):
    # first parse headers
    f.seek(0, 0)
    text_data = f.read()
    headers = re.findall('^[A-Z_]{5}:.*', text_data, re.MULTILINE)
    header_dict = defaultdict(dict)
    for line in headers:
        line_name, line_body = _parse_param_line(line)
        if type(line_body) is dict:
            header_dict[line_name].update(line_body)
        else:
            header_dict[line_name].update({'value': line_body})
    # then parse scan parameters and counts
    data_section = text_data[text_data.find(start_flag) + len(start_flag) + 1:]
    column_names = data_section.splitlines()[0].split()
    parameters_text_lines = re.findall('^[0-9\*\-\s\t.]+?$', data_section, re.MULTILINE)  # line only w 0-9, . -, spc, tab
    parameters_value_array = np.array([[elegant_float(num) for num in line.split()] for line in parameters_text_lines])
    data_frame = pd.DataFrame(data=parameters_value_array, columns=column_names)
    data_frame['PNT'] = data_frame['PNT'].astype('int16')
    # data_frame.set_index('PNT', inplace='True')
    df_clean = data_frame.T.drop_duplicates().T
    # parse flatcone data if present
    flat_all = re.findall('(?<=flat: )[0-9w\s\t\n]+(?=endflat)', text_data, re.MULTILINE)
    flat_number_lines = len(flat_all)
    if len(df_clean) - flat_number_lines <= 1:
        flat_frames = [_parse_flatcone_line(line) for line in flat_all]
        if len(df_clean) - flat_number_lines == 1:
            df_clean.drop(df_clean.index[-1], inplace=True)
        df_clean = df_clean.assign(flat=flat_frames)
    else:
        pass
    data_dict = {'data_frame': df_clean, 'header': dict(header_dict)}  # Strip default factory
    return data_dict


class BackgroundInterpolator(object):
    def __init__(self, filenames):
        self.datasets = {}
        for filename in filenames:
            with open(filename) as file:
                data = parse_ill_data(file)
                for ind in range(len(data['data_frame'])):
                    self.datasets[round(data['data_frame'].loc[ind, 'EI'], 2)] = {
                        'flat': np.array(data['data_frame'].loc[ind, 'flat']), 'mon': data['data_frame'].loc[ind, 'M1']}

    def get_background(self, ei=None, ki=None):
        if ei is not None:
            ei = round(ei, 2)
        elif ki is not None:
            pass
        keys = sorted(self.datasets.keys())
        if ei in keys:
            return self.datasets[ei]['flat'] / self.datasets[ei]['mon']
        elif ei < keys[0]:
            return self.datasets[keys[0]]['flat'] / self.datasets[keys[0]]['mon']
        elif ei > keys[-1]:
            return self.datasets[keys[-1]]['flat'] / self.datasets[keys[-1]]['mon']
        else:

            left_ei = max([x for x in keys if x < ei])
            right_ei = min([x for x in keys if x > ei])
            left_ratio = (ei - left_ei) / (right_ei - left_ei)
            right_ratio = 1 - left_ratio
            return self.datasets[left_ei]['flat'] / self.datasets[left_ei]['mon'] * left_ratio + \
                   self.datasets[right_ei]['flat'] / self.datasets[right_ei]['mon'] * right_ratio


class MultiFlexxScan(object):
    def __init__(self, file_like, config: Config, ub_matrix, background=None):
        try:
            self.filename = file_like.name
        except AttributeError:
            self.filename = 'unknown source'
        self._data, self._config, self._ub_matrix = parse_ill_data(file_like), config, ub_matrix
        try:
            a3_offset = self._data['header']['PARAM']['A3O']
            self._data['data_frame'].loc[:, 'A3'] += a3_offset
            self._data['header']['VARIA']['A3'] += a3_offset
        except KeyError:
            pass
        self._intensity_coeff = np.loadtxt(self._config.instrument['intensity_coeff'], delimiter=',')
        self._weights = np.loadtxt(config.instrument['channel_weights'], delimiter=',')
        self.ki = self._data['data_frame'].loc[1, 'KI']
        self._background = self._load_background(background)
        self._alive = np.loadtxt(config.instrument['detector_enable'])
        self.ef_list = self._config.instrument['ef']
        self.kf_list = self._config.instrument['kf']
        self.de_list = [round(self.ei - ef, 2) for ef in self.ef_list]
        self.a3_start, self.a3_end_actual, self.a3_end_planned = None, None, None
        self.a4_start, self.a4_end_actual, self.a4_end_planned = None, None, None
        self._update_scan_range_()
        self.planned_locus_by_channel, self.actual_locus_by_channel = self._determine_locus()
        self.figure_aspect = self._calculate_aspect()
        self.channel_dataframe = self._process_data()
        self.scanned_parameters = self._data['header']['STEPS'].keys()
        self.no_frames_actual = len(self._data['data_frame'])
        self.no_frames_planned = self._data['header']['COMND']['NP']

        if 'flat' in self._data['data_frame']:
            self.is_flatcone = True
        else:
            self.is_flatcone = False

    @property
    def ei(self):
        return round((self.ki / 0.6947) ** 2, 2)

    def _load_background(self, background: BackgroundInterpolator):
        if background is None:
            return np.zeros([31,5])
        else:
            return background.get_background(self.ei)

    def _update_scan_range_(self):
        # Deal with A4
        self.a3_start = self._data['header']['VARIA']['A3']
        self.a3_end_actual = self._data['data_frame'].iloc[-1]['A3']
        try:
            self.a3_end_planned = self._data['header']['VARIA']['A3'] + \
                                  self._data['header']['STEPS']['A3'] * (self._data['header']['COMND']['NP'] - 1)
        except KeyError:
            self.a3_end_planned = self.a3_end_actual
        # Now deal with A4
        self.a4_start = self._data['header']['VARIA']['A4']
        if 'A4' in self._data['header']['STEPS']:
            self.a4_end_planned = self._data['header']['VARIA']['A4'] + \
                                  self._data['header']['STEPS']['A4'] * (self._data['header']['COMND']['NP'] - 1)
            self.a4_end_actual = self._data['data_frame'].iloc[-1]['A4']
        else:
            self.a4_end_planned = self.a4_start
            self.a4_end_actual = self.a4_start

    def _determine_locus(self):
        num_ch = self._config.instrument['channels']
        channel_offset = self._config.instrument['channel_offset']
        if self.a4_start > 0:
            a4_span = channel_offset * (num_ch - 1)
        else:
            a4_span = channel_offset * (num_ch - 1) * (-1)
        kf_list = self.kf_list
        planned_locus_by_channel = [calculate_locus(self.ki, kf, self.a3_start, self.a3_end_planned,
                                                    self.a4_start, self.a4_end_planned, a4_span, self._ub_matrix) for kf in
                                    kf_list]
        actual_locus_by_channel = [calculate_locus(self.ki, kf, self.a3_start, self.a3_end_actual,
                                                   self.a4_start, self.a4_end_actual, a4_span, self._ub_matrix) for kf in kf_list]

        return planned_locus_by_channel, actual_locus_by_channel

    def _process_data(self):
        num_ch = self._config.instrument['channels']
        channel_offset = self._config.instrument['channel_offset']
        num_flat_frames = len(self._data['data_frame'])
        weights = self._weights
        a4_channel_mask = np.linspace(-channel_offset * (num_ch - 1) / 2, channel_offset * (num_ch - 1) / 2,
                                      num_ch)
        a3_a4_mon_array = np.zeros([num_flat_frames * num_ch, 3])
        detector_alive = np.zeros([31, 5])
        flat_arrays = []
        for ind in range(num_flat_frames):
            # Pandas .loc indexer INCLUDES endpoint!!! ouch.
            a3_a4_mon_array[ind * num_ch: (ind + 1) * num_ch, 0] = self._data['data_frame'].loc[ind, 'A3']
            a3_a4_mon_array[ind * num_ch: (ind + 1) * num_ch, 1] = self._data['data_frame'].loc[ind, 'A4'] + \
                                                                   a4_channel_mask
            a3_a4_mon_array[ind * num_ch: (ind + 1) * num_ch, 2] = max(self._data['data_frame'].loc[ind, 'M1'], 1) * \
                                                                   weights

            detector_alive = np.array(np.logical_or(detector_alive, self._data['data_frame'].loc[ind, 'flat'] > 0))
            flat_frame = np.array(self._data['data_frame'].loc[ind, 'flat'])
            flat_arrays.append(flat_frame)
        detector_alive = self._alive
        channel_dataframe_template = pd.DataFrame(index=range(num_flat_frames * num_ch),
                                                  columns=['A3', 'A4', 'MON', 'px', 'py', 'pz', 'CNTS',
                                                           'valid'], dtype='float')
        channel_dataframe_template.loc[:, ['A3', 'A4', 'MON']] = a3_a4_mon_array
        channel_dataframe = [channel_dataframe_template.copy() for _ in range(len(self.kf_list))]

        a3_a4_array_row = a3_a4_mon_array[:, 0:2].T
        for ind, kf in enumerate(self.kf_list):
            channel_dataframe[ind].loc[:, ['px', 'py', 'pz']] = self._ub_matrix.convert(
                angle_to_qins(self.ki, kf, a3_a4_array_row[0, :], a3_a4_array_row[1, :]), 'sp').T

        channel_count_valid_array = [np.zeros([num_flat_frames * num_ch, 2]) for _ in range(len(self.kf_list))]
        for ind in range(num_flat_frames):
            flat_array = flat_arrays[ind]
            bg = self._background * self._data['data_frame'].loc[ind, 'M1']
            flat_array_min_background = (flat_array - bg * 0.6) / self._intensity_coeff
            for kf_num in range(len(self.kf_list)):
                channel_count_valid_array[kf_num][ind * num_ch: (ind + 1) * num_ch, 0] = flat_array_min_background[:, kf_num] * weights
                channel_count_valid_array[kf_num][ind * num_ch: (ind + 1) * num_ch, 1] = detector_alive[:, kf_num]

        for kf_num in range(len(self.kf_list)):
            channel_dataframe[kf_num].loc[:, ['CNTS', 'valid']] = channel_count_valid_array[kf_num]
        return channel_dataframe

    def _calculate_aspect(self):
        plot_x_r = self._config.plot['x']
        plot_y_r = self._config.plot['y']
        plot_x_unit_len = np.linalg.norm(self._ub_matrix.convert(plot_x_r, 'rs'))
        plot_y_unit_len = np.linalg.norm(self._ub_matrix.convert(plot_y_r, 'rs'))
        return plot_y_unit_len / plot_x_unit_len

    def __str__(self):
        file_name = self.filename
        ei = 'Ei= ' + _fmt_2(self.ei)
        a3_range = ', A3= ' + _fmt_2(self.a3_start) + ' to ' + _fmt_2(self.a3_end_actual) + ', '
        a4_range = 'A4= ' + _fmt_2(self.a4_start) + ' to ' + _fmt_2(self.a4_end_actual) + ', '
        scanned_steps = 'NP ' + str(self.no_frames_actual) + '/' + str(self.no_frames_planned)
        return 'scan ' + file_name + ', ' + ei + a3_range + a4_range + scanned_steps


def calculate_locus(ki, kf, a3_start, a3_end, a4_start, a4_end, a4_span, ub_matrix):
    a3_range = np.linspace(a3_start, a3_end, max(int(a3_end - a3_start), 2))
    a4_range_low = np.linspace(a4_start - a4_span / 2, a4_end - a4_span / 2, max(int(a3_end - a3_start), 2))
    a4_range_high = np.linspace(a4_end + a4_span / 2, a4_start + a4_span / 2, max(int(a3_end - a3_start), 2))
    a4_span_range_low = np.linspace(a4_start + a4_span / 2, a4_start - a4_span / 2, 31)
    a4_span_range_high = np.linspace(a4_end - a4_span / 2, a4_end + a4_span / 2, 31)

    a3_list = np.hstack((a3_range, a3_range[-1] * np.ones(len(a4_span_range_high)),
                         a3_range[::-1], a3_range[0] * np.ones(len(a4_span_range_low))))
    a4_list = np.hstack((a4_range_low, a4_span_range_high, a4_range_high, a4_span_range_low))
    s_locus = angle_to_qins(ki, kf, a3_list, a4_list)
    p_locus = ub_matrix.convert(s_locus, 'sp')

    return np.ndarray.tolist(p_locus[0:2, :].T)


