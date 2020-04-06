#!/usr/bin/env python3

import os
import sys
import glob
import argparse
import logging
import coloredlogs
import numpy as np
from matplotlib import pyplot as plt
import mne

# Baseline to the average of the section from the start of the epoch to the event
BASELINE = (None, 0.1)

parser = argparse.ArgumentParser(description='Automate FMed study grand averaging of MMN.')
parser.add_argument('-v', '--verbose', action='count', default=0)
parser.add_argument('subject', nargs='+')

args = parser.parse_args()

if args.verbose > 0:
    coloredlogs.install(level='DEBUG')
else:                       
    coloredlogs.install(level='INFO')

standard = []
deviant = []
for sid in args.subject:
    # Find the statistics files for this subject
    def find_file(kind):
        path = f"/study/thukdam/analyses/eeg_statistics/mmn/{sid}/*{kind}*.fif"
        find = glob.glob(path)
        if len(find) == 0:
            logging.fatal(f"No {kind} summary file found for {sid}")
            sys.exit(1)
        return find[0]

    standard_file = find_file("standard")
    deviant_file = find_file("deviant")

    standard += mne.read_evokeds(standard_file, baseline=BASELINE)
    deviant += mne.read_evokeds(deviant_file, baseline=BASELINE)

standard_average = mne.combine_evoked(standard, weights='nave')
deviant_average = mne.combine_evoked(deviant, weights='nave')

difference_average = mne.combine_evoked([standard_average, -deviant_average], weights='equal')

evoked = dict()
evoked["Standard"] = standard_average
evoked["Deviant"] = deviant_average
evoked["Difference"] = difference_average

colors = dict(Standard="Green", Deviant="Red", Difference="Black")

def plot_sminusd(electrode, scale=2.5, auto=False):
    pick = standard_average.ch_names.index(electrode)
    fig, ax = plt.subplots(figsize=(6, 4))
    kwargs = dict(axes=ax, picks=pick,
        truncate_yaxis=False,
        truncate_xaxis=False,
        colors=colors,
        split_legend=True,
        legend='lower right',
        show_sensors=False,
        ci=0.95,
        show=False)
    if auto:
        name = "auto"
        mne.viz.plot_compare_evokeds(evoked,  **kwargs)
    else:
        name = str(scale)
        mne.viz.plot_compare_evokeds(evoked, ylim=dict(eeg=[-1 * scale, scale]), **kwargs)

    filename = f"/study3/thukdam/analyses/eeg_statistics/mmn/sminusd_grand_average_{name}_{electrode}.png"
    fig.savefig(filename, dpi=300)

plot_sminusd("Cz", auto=True)
plot_sminusd("Fz", auto=True)
plot_sminusd("Pz", auto=True)
plot_sminusd("T8", auto=True)
plot_sminusd("Cz", 2.5)
plot_sminusd("Fz", 2.5)
plot_sminusd("Pz", 2.5)
plot_sminusd("T8", 2.5)
plot_sminusd("Cz", 5.0)
plot_sminusd("Fz", 5.0)
plot_sminusd("Pz", 5.0)
plot_sminusd("T8", 5.0)


