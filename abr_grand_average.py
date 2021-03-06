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

# Baseline to the start of the section
BASELINE = (None, 0)

timestamp = datetime.datetime.now().isoformat()

parser = argparse.ArgumentParser(description='Automate FMed study grand averaging of ABR.')
parser.add_argument('-v', '--verbose', action='count', default=0)
parser.add_argument('-n', '--name', default=timestamp.replace(":","."))
parser.add_argument('subject', nargs='+')

args = parser.parse_args()

if args.verbose > 0:
    coloredlogs.install(level='DEBUG')
else:                       
    coloredlogs.install(level='INFO')


INPUT_DIR = "/study/thukdam/analyses/eeg_statistics/abr"
OUTPUT_DIR = f"/scratch/dfitch/plots/{args.name}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(f"{OUTPUT_DIR}/README.txt", 'w') as f:
    f.write(' '.join(sys.argv) + "\n\n")
    f.write(f"Generated on {timestamp} and written to {args.name} from the following subjects:\n")
    for item in args.subject:
        f.write("%s\n" % item)

logging.info(f"Reading {args.subject} from {INPUT_DIR} and writing to {OUTPUT_DIR}")

total = []
for sid in args.subject:
    # Find the statistics file for this subject
    path = f"{INPUT_DIR}/{sid}/*-ave.fif"
    find = glob.glob(path)
    if len(find) == 0:
        logging.fatal(f"No summary file found for {sid}")
        sys.exit(1)
    elif len(find) > 1:
        logging.warn(f"Multiple summary files found for {sid}, picking first")

    total_file = find[0]

    total += mne.read_evokeds(total_file, baseline=BASELINE)

all_average = mne.combine_evoked(total, weights='nave')

logging.info(f"Read {args.subject} from {INPUT_DIR}, creating plots in {OUTPUT_DIR}")



def plot(electrode, scale=2.5, auto=False):
    if electrode is None:
        pick = "all"
        electrode = "all"
    else:
        pick = all_average.ch_names.index(electrode)

    fig, ax = plt.subplots(figsize=(4, 8/3))
    ax.axvline(x=0, color='black')
    ax.axhline(y=0, color='black')

    kwargs = dict(axes=ax, picks=pick,
        titles=dict(eeg=electrode),
        window_title=electrode,
        time_unit="ms",
        show=False)

    if auto:
        name = "auto"
        fig = all_average.plot(**kwargs)
    else:
        name = str(scale)
        fig = all_average.plot(ylim=dict(eeg=[-1 * scale, scale]), **kwargs)

    # MNE's Evoked object doesn't let you pass linewidth. Thus, this horrendous hack:

    line = plt.gca().lines[-1]
    line.set_linewidth(2.)

    title = plt.gca().title
    title.set_text(electrode)

    filename = f"{OUTPUT_DIR}/{args.name}_{name}_{electrode}.png"
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    logging.info(f"Plot for abr grand average on {electrode} saved to {filename}")

plot("Cz", 1.0)
plot("Fz", 1.0)
plot("Pz", 1.0)
plot("T8", 1.0)


