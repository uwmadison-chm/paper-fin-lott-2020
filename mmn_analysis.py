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

# Mutated from mmn_grand_average.py to do statistics

# Baseline to the average of the section from the start of the epoch to the event
BASELINE = (None, 0.1)
# Expected number of samples in a decimated statistics file
EXPECTED_SAMPLES = 2731

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


INPUT_DIR = "/study/thukdam/analyses/eeg_statistics/mmn"
GOOD_TIMES = None

logging.info(f"Reading group 1 and group 2 from {INPUT_DIR}")

def read_evokeds(f):
    global GOOD_TIMES
    es = mne.read_evokeds(f, baseline=BASELINE)
    samples = es[0].data.shape[1]
    if samples != EXPECTED_SAMPLES:
        """
        Now, we're expecting a certain sample rate so that we end up with 2731 samples from these arrays.
        But we have old cruddy data that has been decimated differently.
        So we resample and force the timepoints to be identical (there's a little jitter)

        So far we only hit one file, so I am being a bad person and hard coding a resampling rate
        that will get files like that one to match. If this does NOT fix future files, we'll have
        to figure out how to get at the sample rate of the MNE Evoked lists, and do it dynamically.
        Couldn't find it in a few hours of poking.
        """
        logging.warning(f"Resampling on {f}, did not get expected decimated statistics length {EXPECTED_SAMPLES}, got {samples}...")

        es[0].resample(5441)
        es[0].times = GOOD_TIMES
    else:
        GOOD_TIMES = es[0].times
    return es

def load_group(group):
    total = []
    standard = []
    deviant = []
    weights = []
    for sid in group:
        # Find the statistics files for this subject
        def find_file(kind):
            path = f"{INPUT_DIR}/{sid}/*{kind}-ave.fif"
            find = glob.glob(path)
            if len(find) == 0:
                logging.fatal(f"No {kind} summary file found for {sid}")
                sys.exit(1)
            return find[0]

        total_file = find_file("all")
        standard_file = find_file("standard")
        deviant_file = find_file("deviant")

        total += read_evokeds(total_file)
        standard += read_evokeds(standard_file)
        deviant += read_evokeds(deviant_file)

    # Calculate difference waves separately
    pairs = zip(deviant, standard)
    difference = [ mne.combine_evoked([d, -s], weights='equal') for (d,s) in pairs ]

    nave = [ x.nave for x in total ]

    return {
        'total': total,
        'standard': standard,
        'deviant': deviant,
        'difference': difference,
        'nave': nave,
    }

data1 = load_group(group1)
data2 = load_group(group2)


def amplitude(electrode, evoked, window_start_ms, window_end_ms):
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

    # Now, instead of combining the evoked data using an average,
    # we calculate area under the curve / s
    # NOTE: this is resulting in uV * seconds as the unit, not ms
    area = integrate.simps(data_window, times_window)

    # Now, we multiply by 1000 to get ms and divide by the length of the window to get uV
    return area * 1000 / (window_end_ms - window_start_ms)


def get_amplitudes(electrode, data):
    MMN_START = 90
    MMN_END = 180
    return [ amplitude(electrode, x, MMN_START, MMN_END) for x in data ]


group1_difference_fz = get_amplitudes('Fz', data1['difference'])
group2_difference_fz = get_amplitudes('Fz', data2['difference'])

group1_difference_cz = get_amplitudes('Cz', data1['difference'])
group2_difference_cz = get_amplitudes('Cz', data2['difference'])

# Store "good" trial counts for each participant and electrode...
# We have to do this per-electrode to calculate weights when 
# rejecting specific electrodes.
group1_nave_fz = list(data1['nave'])
group1_nave_cz = list(data1['nave'])
group2_nave_fz = list(data2['nave'])
group2_nave_cz = list(data2['nave'])

# Remove Fz data from FM1618
remove = group1.index('FM1618')
del group1_difference_fz[remove]
del group1_nave_fz[remove]

def calc_weights(nave):
    # Calculate weights by # of trials not rejected
    total_weight = sum(nave)
    return [ (x / total_weight) * len(nave) for x in nave ]

group1_weights_cz = calc_weights(group1_nave_cz)
group1_weights_fz = calc_weights(group1_nave_fz)

group2_weights_cz = calc_weights(group2_nave_cz)
group2_weights_fz = calc_weights(group2_nave_fz)


if args.debug:
    from IPython import embed; embed()

# Dump details to csv files 

OUTPUT_DIR = "/study/thukdam/analyses/eeg_statistics/mmn/stats"

def dump_csv(name, subjects, fz, wfz, cz, wcz):
    with open(f"{OUTPUT_DIR}/{name}.csv", 'w', newline='') as csvfile:
        out = csv.writer(csvfile)
        out.writerow(['ID', 'Fz area amplitude', 'Fz weight', 'Cz area amplitude', 'Cz weight'])
        tuples = zip(subjects, fz, wfz, cz, wcz)
        for x in tuples:
            out.writerow(list(x))

dump_csv(group1_name, group1, group1_difference_fz, group1_weights_fz, group1_difference_cz, group1_weights_cz)
dump_csv(group2_name, group2, group2_difference_fz, group2_weights_fz, group2_difference_cz, group2_weights_cz)


# And now, do a simple t test across those groups

def ttest(g1, g2, w1, w2):
    # output = ttest_ind(g1, g2, usevar='unequal')
    output = ttest_ind(g1, g2, usevar='unequal', weights=(w1, w2))
    return output

print(f"Group difference T test on fz: {ttest(group1_difference_fz, group2_difference_fz, group1_weights_fz, group2_weights_fz)}\n")
print(f"Group difference T test on cz: {ttest(group1_difference_cz, group2_difference_cz, group1_weights_cz, group2_weights_cz)}\n")

# Weight the stats proportionally by the weights we calculated, as the T-test is doing above
wg1f = np.multiply(group1_difference_fz, group1_weights_fz)
wg2f = np.multiply(group2_difference_fz, group2_weights_fz)
wg1c = np.multiply(group1_difference_cz, group1_weights_cz)
wg2c = np.multiply(group2_difference_cz, group2_weights_cz)

print(f"Group 1 [{group1_name}] fz difference mean: {np.mean(wg1f)} std: {np.std(wg1f)}")
print(f"Group 1 [{group1_name}] cz difference mean: {np.mean(wg1c)} std: {np.std(wg1c)}")
print(f"Group 2 [{group2_name}] fz difference mean: {np.mean(wg2f)} std: {np.std(wg2f)}")
print(f"Group 2 [{group2_name}] cz difference mean: {np.mean(wg2c)} std: {np.std(wg2c)}")
