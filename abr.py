#!/usr/bin/env python3

import sys
import argparse
import logging
import coloredlogs
import mne
from mne.preprocessing import ICA, create_ecg_epochs
import matplotlib.pyplot as plt

from eeg_shared import BDFWithMetadata

parser = argparse.ArgumentParser(description='Automate FMed study artifact rejection and analysis of MMN. By default loads the file for viewing')

parser.add_argument('input', help='Path to input file.')
parser.add_argument('-v', '--verbose', action='count', default=0)
parser.add_argument('--skip-view', action='store_true', help="Skip viewing file and editing artifact mask")
parser.add_argument('--topo', action='store_true', help="Topo map")
parser.add_argument('--shell', action='store_true', help="Drop into an interactive ipython environment")
parser.add_argument('--epoch-average', action='store_true', help="Plot epoch averages")
parser.add_argument('--epoch-image', action='store_true', help="Very slow colormap image of epochs")
parser.add_argument('--epoch-view', action='store_true', help="Simple linear view of epochs, default end view")
parser.add_argument('--psd', metavar='HZ', action='store', help="Plot power spectral density up to HZ")
parser.add_argument('--force', action='store_true', help="Force running outside of raw-data/subjects, saving masks to current directory")
parser.add_argument('--all', action='store_true', help="Generate all plots")
parser.add_argument('--bandpass-from', metavar='HZ', action='store', help="Lower frequency of bandpass (default is 100)")
parser.add_argument('--bandpass-to', metavar='HZ', action='store', help="Higher frequency of bandpass (default is 3000)")


args = parser.parse_args()

if args.verbose > 0:
    coloredlogs.install(level='DEBUG')
else:                       
    coloredlogs.install(level='INFO')
  

raw_file = args.input

f = BDFWithMetadata(raw_file, "abr", args.force)
f.load()
if args.bandpass_from:
    f.highpass = float(args.bandpass_from)
    logging.info(f"Overriding highpass frequency of band to {f.highpass}Hz")
if args.bandpass_to:
    f.lowpass = float(args.bandpass_to)
    logging.info(f"Overriding lowpass frequency of band to {f.lowpass}Hz")
if not args.skip_view:
    f.artifact_rejection()

if args.psd or args.all:
    f.psd(int(args.psd or 2000))


epochs = f.build_epochs()


if args.shell:
    logging.warning("Dropping into shell, epochs are in `epochs` and the raw file wrapper is in `f`")
    from IPython import embed
    embed() 
    sys.exit()


elif args.topo or args.all:
    f.topo()


elif args.epoch_image or args.all:
    f.epoch_images()


elif args.epoch_average or args.all:
    logging.info("Loading epoch average plot...")
    average = epochs.average()
    fig = average.plot(spatial_colors=True, show=False)
    f.save_figure(fig, "epoch_average")


elif args.epoch_view or args.all:
    f.epoch_view()

