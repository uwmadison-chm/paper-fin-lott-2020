#!/usr/bin/env python3

import os
import sys
import glob
import argparse
import logging
import coloredlogs
import datetime
import numpy as np
from scipy import stats
from scipy import integrate
import mne
import csv
from statsmodels.stats.weightstats import ttest_ind

# Mutated from mmn_analysis.py and abr_grand_average.py to do ABR t-tests

# Baseline to the start of the section
BASELINE = (None, 0)

timestamp = datetime.datetime.now().isoformat()

parser = argparse.ArgumentParser(description='Automate FMed study statistical analysis of MMN.')
parser.add_argument('-v', '--verbose', action='count', default=0)
parser.add_argument('--debug', action="store_true")
# parser.add_argument('subject', nargs='+')

args = parser.parse_args()

if args.verbose > 0:
    coloredlogs.install(level='DEBUG')
else:                       
    coloredlogs.install(level='INFO')


group1 = ['FM1192', 'FM1618', 'FM7780', 'FM2004_0717', 'FM7779']
group1_name = "living"
group2 = ['FM0505_0115', 'FM1001_0313', 'FM1002_1018', 'FM2000_1117', 'FM2001_0413', 'FM2001_0518', 'FM5001_0518']
group2_name = "postmortem"


INPUT_DIR = "/study/thukdam/analyses/eeg_statistics/abr"

logging.info(f"Reading group 1 and group 2 from {INPUT_DIR}")

def load_group(group):
    total = []
    weights = []
    for sid in group:
        # Find the statistics files for this subject
        path = f"{INPUT_DIR}/{sid}/*-ave.fif"
        find = glob.glob(path)
        if len(find) == 0:
            logging.fatal(f"No {kind} summary file found for {sid}")
            sys.exit(1)
        elif len(find) > 1:
            logging.warn(f"Multiple summary files found for {sid}, picking first")

        total_file = find[0]

        total += mne.read_evokeds(total_file, baseline=BASELINE)

    # Calculate weights by # of trials not rejected
    nave = [ x.nave for x in total ]
    total_weight = sum(nave)
    weights = [ (x / total_weight) * len(nave) for x in nave ]

    return {
        'total': total,
        'weights': weights,
    }

data1 = load_group(group1)
data2 = load_group(group2)


def crop(electrode, evoked, window_start_ms, window_end_ms):
    pick = evoked.ch_names.index(electrode)

    times = evoked.times
    data = evoked.data[pick]

    # We have to crop to the window
    window_start_s = window_start_ms / 1000
    window_end_s = window_end_ms / 1000

    # NOTE: There has to be a more idiomatic way to find the first index
    # matching a filter but ... here we are
    start_index = np.where(times>=window_start_s)[0][0]
    end_index = np.where(times>=window_end_s)[0][0]

    data_window = data[start_index:end_index]
    times_window = times[start_index:end_index]

    return (data_window, times_window)


def amplitude(electrode, evoked, window_start_ms, window_end_ms):
    data_window, times_window = crop(electrode, evoked, window_start_ms, window_end_ms)

    # Now, instead of combining the evoked data using an average,
    # we calculate area under the curve / s
    # NOTE: Pretty sure this is resulting in seconds as the unit, not ms,
    # but since that's what the MNE Evoked objects think in, seems fine
    area = integrate.simps(data_window, times_window)
    return area


def get_amplitudes(electrode, data):
    ABR_START = 4
    ABR_END = 8
    return [ amplitude(electrode, x, ABR_START, ABR_END) for x in data ]



def peak_distance(electrode, evoked,
        min_window_start_ms, min_window_end_ms,
        max_window_start_ms, max_window_end_ms):

    min_window, _ = crop(electrode, evoked, min_window_start_ms, min_window_end_ms)
    max_window, _ = crop(electrode, evoked, max_window_start_ms, max_window_end_ms)

    # Instead of the default threshold taking (max - min) / 4, we have more data, so use it all
    # to find the threshold for both windows
    data = np.concatenate((min_window, max_window))
    thresh = (max(data) - min(data))/4

    low_locs, low_mags = mne.preprocessing.peak_finder(min_window, thresh=thresh, extrema=-1)
    high_locs, high_mags = mne.preprocessing.peak_finder(max_window, thresh=thresh, extrema=1)

    if len(high_mags) == 0:
        high_mags = [0]
    if len(low_mags) == 0:
        low_mags = [0]

    minimum = min(low_mags)
    maximum = max(high_mags)

    return maximum - minimum


def get_peaks(electrode, data):
    # ABR_START = 2
    # ABR_MID = 6
    # ABR_END = 10
    ABR_START = 4
    ABR_MID = 7
    ABR_END = 10
    return [ peak_distance(electrode, x, ABR_START, ABR_MID, ABR_MID, ABR_END) for x in data ]


group1_peak_fz = get_peaks('Fz', data1['total'])
group1_peak_cz = get_peaks('Cz', data1['total'])

group2_peak_fz = get_peaks('Fz', data2['total'])
group2_peak_cz = get_peaks('Cz', data2['total'])

group1_fz = get_amplitudes('Fz', data1['total'])
group1_cz = get_amplitudes('Cz', data1['total'])

group2_fz = get_amplitudes('Fz', data2['total'])
group2_cz = get_amplitudes('Cz', data2['total'])

group1_weights = data1['weights']
group2_weights = data2['weights']

if args.debug:
    from IPython import embed; embed()


# Dump details to csv files 

OUTPUT_DIR = "/study/thukdam/analyses/eeg_statistics/abr/stats"

def dump_csv(name, subjects, fz, cz, fzpp, czpp, w):
    with open(f"{OUTPUT_DIR}/{name}.csv", 'w', newline='') as csvfile:
        out = csv.writer(csvfile)
        out.writerow(['ID', 'Fz area amplitude', 'Cz area amplitude', 'Fz peak to peak', 'Cz peak to peak', 'Weight'])
        tuples = zip(subjects, fz, cz, fzpp, czpp, w)
        for x in tuples:
            out.writerow(list(x))

dump_csv(group1_name, group1, group1_fz, group1_cz, group1_peak_fz, group1_peak_cz, group1_weights)
dump_csv(group2_name, group2, group2_fz, group2_cz, group2_peak_fz, group2_peak_cz, group2_weights)


# And now, do a simple t test across those groups

def ttest(g1, g2, w1, w2):
    # output = ttest_ind(g1, g2, usevar='unequal')
    output = ttest_ind(g1, g2, usevar='unequal', weights=(w1, w2))
    return output

print(f"Welch's T test on fz area under difference curve: {ttest(group1_fz, group2_fz, group1_weights, group2_weights)}\n")
print(f"Welch's T test on cz area under difference curve: {ttest(group1_cz, group2_cz, group1_weights, group2_weights)}\n")

print(f"Welch's T test on fz peak-to-peak: {ttest(group1_peak_fz, group2_peak_fz, group1_weights, group2_weights)}\n")
print(f"Welch's T test on cz peak-to-peak: {ttest(group1_peak_cz, group2_peak_cz, group1_weights, group2_weights)}\n")

