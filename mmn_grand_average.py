#!/usr/bin/env python3

import os
import sys
import glob
import argparse
import logging
import coloredlogs
import datetime
import numpy as np
from matplotlib import pyplot as plt
import mne

# Baseline to the average of the section from the start of the epoch to the event
BASELINE = (None, 0.1)
# Expected number of samples in a decimated statistics file
EXPECTED_SAMPLES = 2731

timestamp = datetime.datetime.now().isoformat()

parser = argparse.ArgumentParser(description='Automate FMed study grand averaging of MMN.')
parser.add_argument('-v', '--verbose', action='count', default=0)
parser.add_argument('-n', '--name', default=timestamp.replace(":","."))
parser.add_argument('--debug', action="store_true")
parser.add_argument('subject', nargs='+')

args = parser.parse_args()

if args.verbose > 0:
    coloredlogs.install(level='DEBUG')
else:                       
    coloredlogs.install(level='INFO')


INPUT_DIR = "/study/thukdam/analyses/eeg_statistics/mmn"
OUTPUT_DIR = f"/scratch/dfitch/plots/{args.name}"
os.makedirs(OUTPUT_DIR, exist_ok=True)
GOOD_TIMES = None

with open(f"{OUTPUT_DIR}/README.txt", 'w') as f:
    f.write(' '.join(sys.argv) + "\n\n")
    f.write(f"Generated on {timestamp} and written to {args.name} from the following subjects:\n")
    for item in args.subject:
        f.write("%s\n" % item)

logging.info(f"Reading {args.subject} from {INPUT_DIR} and writing to {OUTPUT_DIR}")

def read_evokeds(f):
    global GOOD_TIMES
    es = mne.read_evokeds(f, baseline=BASELINE)
    if es[0].data.shape[1] != EXPECTED_SAMPLES:
        """
        Now, we're expecting a certain sample rate so that we end up with 2731 samples from these arrays.
        But we have old cruddy data that has been decimated differently.
        So we resample and force the timepoints to be identical (there's a little jitter)

        So far we only hit one file, so I am being a bad person and hard coding a resampling rate
        that will get files like that one to match. If this does NOT fix future files, we'll have
        to figure out how to get at the sample rate of the MNE Evoked lists, and do it dynamically.
        Couldn't find it in a few hours of poking.
        """
        logging.warning(f"Resampling on {f}, did not get expected decimated statistics length {EXPECTED_SAMPLES}")

        es[0].resample(5441)
        es[0].times = GOOD_TIMES
    else:
        GOOD_TIMES = es[0].times
    return es

total = []
standard = []
deviant = []
for sid in args.subject:
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

if args.debug:
    from IPython import embed; embed()

all_average = mne.combine_evoked(total, weights='nave')
standard_average = mne.combine_evoked(standard, weights='nave')
deviant_average = mne.combine_evoked(deviant, weights='nave')
difference_average = mne.combine_evoked([deviant_average, -standard_average], weights='equal')


logging.info(f"Read {args.subject} from {INPUT_DIR}, creating plots in {OUTPUT_DIR}")

evoked = dict()
evoked["Standard"] = standard_average
evoked["Deviant"] = deviant_average
evoked["Difference"] = difference_average

colors = dict(Standard="Green", Deviant="Red", Difference="Black")

def plot_dms(electrode, scale=2.5, auto=False):
    if electrode is None:
        pick = "all"
        electrode = "all"
    else:
        pick = standard_average.ch_names.index(electrode)

    fig, ax = plt.subplots(figsize=(4, 8/3))

    kwargs = dict(axes=ax, picks=pick,
        truncate_yaxis=False,
        truncate_xaxis=False,
        colors=colors,
        split_legend=True,
        legend='lower right',
        show_sensors=False,
        ci=0.95,
        show=False)

    if pick == "all":
        # Default is gfp (global field power), let's use mean plz
        kwargs['combine'] = 'mean'

    if auto:
        name = "auto"
        mne.viz.plot_compare_evokeds(evoked,  **kwargs)
    else:
        name = str(scale)
        mne.viz.plot_compare_evokeds(evoked, ylim=dict(eeg=[-1 * scale, scale]), **kwargs)

    filename = f"{OUTPUT_DIR}/{args.name}_{name}_{electrode}.png"
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    logging.info(f"Plot for mmn grand average on {electrode} saved to {filename}")

plot_dms("Cz", 6.0)
plot_dms("Fz", 6.0)
plot_dms("Pz", 6.0)
plot_dms("T8", 6.0)

