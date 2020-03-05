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
parser.add_argument('--sminusd', metavar='ELECTRODE', action='store', help="Standard minus deviant of a specified electrode (Cz, Fz, T8, Pz)")
parser.add_argument('--sminusd-mean', action='store_true', help="Mean standard minus deviant across all 4 electrodes")
parser.add_argument('--epoch-image', action='store_true', help="Very slow colormap image of epochs")
parser.add_argument('--epoch-view', action='store_true', help="Simple linear view of epochs, default end view")
parser.add_argument('--psd', metavar='HZ', action='store', help="Plot power spectral density up to HZ")
parser.add_argument('--force', action='store_true', help="Force running outside of raw-data/subjects, saving masks to current directory")
parser.add_argument('--all', action='store_true', help="Generate all plots")
parser.add_argument('--initial-laptop', action='store_true', help="Data is from 2013I (initial settings) north laptop after restore")
parser.add_argument('--bandpass-from', metavar='HZ', action='store', help="Lower frequency of bandpass (default is 1)")
parser.add_argument('--bandpass-to', metavar='HZ', action='store', help="Higher frequency of bandpass (default is 35)")
parser.add_argument('--no-reference', action='store_true', help="Do not reference mastoids")
parser.add_argument('--no-events', action='store_true', help="Do not show events")
parser.add_argument('--display-huge', action='store_true', help="Zoom way out to display entire file")
parser.add_argument('--no-crop', action='store_true', help="Do not crop file")
parser.add_argument('--no-notch', action='store_true', help="Do not notch filter at 50Hz")


args = parser.parse_args()

if args.verbose > 0:
    coloredlogs.install(level='DEBUG')
else:                       
    coloredlogs.install(level='INFO')
  

raw_file = args.input

f = BDFWithMetadata(raw_file, "mmn", args.force, is_2013I=args.initial_laptop, no_reference=args.no_reference, no_notch=(args.no_notch or args.skip_view), no_crop=args.no_crop)
f.load()
if args.bandpass_from:
    f.highpass = float(args.bandpass_from)
    logging.info(f"Overriding highpass frequency of band to {f.highpass}Hz")
if args.bandpass_to:
    f.lowpass = float(args.bandpass_to)
    logging.info(f"Overriding lowpass frequency of band to {f.lowpass}Hz")
if not args.skip_view:
    f.artifact_rejection(args.display_huge, args.no_events)


if args.psd or args.all:
    f.psd(int(args.psd or 120))


epochs = f.build_epochs()
# Only decimate if srate is high
if f.raw.info['sfreq'] > 16000:
    logging.info("Decimating epochs in memory")
    epochs.decimate(3)
else:
    logging.info("File already decimated, not decimating")


if args.sminusd or args.sminusd_mean or args.all:
    # Plot standard and deviant on one figure, plus plot the difference, like original matlab
    deviant = epochs["Deviant"].average()
    standard = epochs["Standard"].average()

    difference = mne.combine_evoked([deviant, -standard], weights='equal')

    evoked = dict()
    evoked["Standard"] = standard
    evoked["Deviant"] = deviant
    evoked["Difference"] = difference

    if args.shell:
        logging.warning("Dropping into shell, epochs are in `epochs`, evoked dict is `evoked`, and the raw file wrapper is in `f`")
        from IPython import embed
        embed() 
        sys.exit()

    colors = dict(Standard="Green", Deviant="Red", Difference="Black")

    # TODO: Figure out what we need to change about the evoked data so we get confidence intervals displayed
    # @agramfort in mne-tools/mne-python gitter said: "to have confidence intervals you need repetitions which I think is a list of evoked or not epochs you need to pass" 
    # May want to do something more like: https://mne.tools/stable/auto_examples/stats/plot_sensor_regression.html?highlight=plot_compare_evokeds

    def plot_sminusd(electrode, scale=2.5, auto=False):
        pick = standard.ch_names.index(electrode)
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

        f.save_figure(fig, f"sminusd_{name}_{electrode}")

    if args.all:
        plot_sminusd("Cz", 2.5)
        plot_sminusd("Fz", 2.5)
        plot_sminusd("Pz", 2.5)
        plot_sminusd("T8", 2.5)
        plot_sminusd("Cz", 5.0)
        plot_sminusd("Fz", 5.0)
        plot_sminusd("Pz", 5.0)
        plot_sminusd("T8", 5.0)
        plot_sminusd("Cz", auto=True)
        plot_sminusd("Fz", auto=True)
        plot_sminusd("Pz", auto=True)
        plot_sminusd("T8", auto=True)

    if args.sminusd:
        if args.sminusd in standard.ch_names:
            plot_sminusd(args.sminusd)
        else:
            logging.warning(f"Could not find electrode '{args.sminusd}'")

    if args.sminusd_mean:
        picks = ['Cz', 'Fz', 'Pz', 'T8']
        fig = mne.viz.plot_compare_evokeds(evoked, picks=picks,
                colors=colors, combine='mean', ci=0.95, show=False)
        f.save_figure(fig[0], f"sminusd_mean")


elif args.shell:
    logging.warning("Dropping into shell, epochs are in `epochs` and the raw file wrapper is in `f`")
    from IPython import embed
    embed() 

elif args.epoch_image or args.all:
    f.epoch_images()

elif args.topo or args.all:
    f.topo()

elif args.epoch_view or args.all:
    f.epoch_view()

