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
parser.add_argument('--save-average', action='store_true', help="Save averaged evoked epochs in a standard MNE file")
parser.add_argument('--all', action='store_true', help="Generate all plots")
parser.add_argument('--bandpass-from', metavar='HZ', action='store', help="Lower frequency of bandpass (default is 100)")
parser.add_argument('--bandpass-to', metavar='HZ', action='store', help="Higher frequency of bandpass (default is 3000)")
parser.add_argument('--no-reference', action='store_true', help="Do not reference mastoids")
parser.add_argument('--reference-o1', action='store_true', help="Only reference o1 mastoid")
parser.add_argument('--reference-o2', action='store_true', help="Only reference o2 mastoid")
parser.add_argument('--display-huge', action='store_true', help="Zoom way out to display entire file")
parser.add_argument('--no-crop', action='store_true', help="Do not crop file")
parser.add_argument('--no-notch', action='store_true', help="Do not notch filter at 50Hz")


args = parser.parse_args()

if args.verbose > 0:
    coloredlogs.install(level='DEBUG')
else:                       
    coloredlogs.install(level='INFO')
  

raw_file = args.input


f = BDFWithMetadata(raw_file, "abr", args.force, no_reference=args.no_reference, reference_o1=args.reference_o1, reference_o2=args.reference_o2, no_notch=(args.no_notch or args.skip_view), no_crop=args.no_crop)
f.load()
if args.bandpass_from:
    f.highpass = float(args.bandpass_from)
    logging.info(f"Overriding highpass frequency of band to {f.highpass}Hz")
if args.bandpass_to:
    f.lowpass = float(args.bandpass_to)
    logging.info(f"Overriding lowpass frequency of band to {f.lowpass}Hz")
if not args.skip_view:
    f.artifact_rejection(args.display_huge)

if args.psd or args.all:
    f.psd(int(args.psd or 2000))


epochs = f.build_epochs()


if args.shell:
    logging.warning("Dropping into shell, epochs are in `epochs` and the raw file wrapper is in `f`")
    from IPython import embed
    embed() 
    sys.exit()


if args.topo or args.all:
    f.topo()



if args.epoch_average or args.all:
    logging.info("Building epoch average plots...")
    average = epochs.average()

    def plot_average(electrode, scale=2.5, auto=False):
        pick = average.ch_names.index(electrode)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.axvline(x=0, linewidth=0.5, color='black')
        ax.axhline(y=0, linewidth=0.5, color='black')
        kwargs = dict(axes=ax, picks=pick,
                titles=dict(eeg=electrode),
                show=False)
        if auto:
            name = "auto"
            fig = average.plot(**kwargs)
        else:
            name = str(scale)
            fig = average.plot(ylim=dict(eeg=[-1 * scale, scale]), **kwargs)
        f.save_figure(fig, f"epoch_average_{name}_{electrode}")

    logging.info("Saving epoch average with spatial colors")
    fig = average.plot(spatial_colors=True, show=False)
    f.save_figure(fig, "epoch_average_spatial")

    logging.info("Plotting individual electrode averages")
    plot_average("Cz", 0.25)
    plot_average("Fz", 0.25)
    plot_average("Pz", 0.25)
    plot_average("T8", 0.25)
    plot_average("Cz", 2.5)
    plot_average("Fz", 2.5)
    plot_average("Pz", 2.5)
    plot_average("T8", 2.5)
    plot_average("Cz", auto=True)
    plot_average("Fz", auto=True)
    plot_average("Pz", auto=True)
    plot_average("T8", auto=True)



if args.epoch_image:
    f.epoch_images()


if args.epoch_view:
    f.epoch_view()


if args.save_average or args.all:
    f.save_average()
