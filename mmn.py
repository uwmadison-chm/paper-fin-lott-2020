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
parser.add_argument('--dms', metavar='ELECTRODE', action='store', help="Deviant minus standard of a specified electrode (Cz, Fz, T8, Pz)")
parser.add_argument('--dms-mean', action='store_true', help="Mean deviant minus standard across all 4 electrodes")
parser.add_argument('--epoch-image', action='store_true', help="Very slow colormap image of epochs")
parser.add_argument('--epoch-view', action='store_true', help="Simple linear view of epochs, default end view")
parser.add_argument('--psd', metavar='HZ', action='store', help="Plot power spectral density up to HZ")
parser.add_argument('--force', action='store_true', help="Force running outside of raw-data/subjects, saving masks to current directory")
parser.add_argument('--save-average', action='store_true', help="Save averaged evoked epochs in a standard MNE file")
parser.add_argument('--all', action='store_true', help="Generate all plots and save average evoked epochs")
parser.add_argument('--initial-laptop', action='store_true', help="Data is from 2013I (initial settings) north laptop after restore")
parser.add_argument('--bandpass-from', metavar='HZ', action='store', help="Lower frequency of bandpass (default is 1)")
parser.add_argument('--bandpass-to', metavar='HZ', action='store', help="Higher frequency of bandpass (default is 35)")
parser.add_argument('--no-reference', action='store_true', help="Do not reference mastoids")
parser.add_argument('--reference-o1', action='store_true', help="Only reference o1 mastoid")
parser.add_argument('--reference-o2', action='store_true', help="Only reference o2 mastoid")
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

f = BDFWithMetadata(raw_file, "mmn", args.force, is_2013I=args.initial_laptop, no_reference=args.no_reference, reference_o1=args.reference_o1, reference_o2=args.reference_o2, no_notch=(args.no_notch or args.skip_view), no_crop=args.no_crop)
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
    # All the data was just reduced by a factor of 3 because that fits in memory better
    # In the future, we probably want to reduce down to 512hz as the manual process did
    # factor = f.raw.info['sfreq'] / 512
    factor = 3
    logging.info(f"Decimating epochs in memory by a factor of {factor}")
    epochs.decimate(factor)
else:
    logging.info("File already decimated, not decimating")


if args.dms or args.dms_mean or args.all:
    # Plot standard and deviant on one figure, plus plot the deviant minus standard difference, like original matlab
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

    def plot_dms(electrode, scale=2.5, auto=False):
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

        f.save_figure(fig, f"dms_{name}_{electrode}")

    if args.all:
        plot_dms("Cz", 2.5)
        plot_dms("Fz", 2.5)
        plot_dms("Pz", 2.5)
        plot_dms("T8", 2.5)
        plot_dms("Cz", 5.0)
        plot_dms("Fz", 5.0)
        plot_dms("Pz", 5.0)
        plot_dms("T8", 5.0)
        plot_dms("Cz", auto=True)
        plot_dms("Fz", auto=True)
        plot_dms("Pz", auto=True)
        plot_dms("T8", auto=True)

    if args.dms:
        if args.dms in standard.ch_names:
            plot_dms(args.dms)
        else:
            logging.warning(f"Could not find electrode '{args.dms}'")

    if args.dms_mean:
        picks = ['Cz', 'Fz', 'Pz', 'T8']
        fig = mne.viz.plot_compare_evokeds(evoked, picks=picks,
                colors=colors, combine='mean', ci=0.95, show=False)
        f.save_figure(fig[0], f"dms_mean")


elif args.shell:
    logging.warning("Dropping into shell, epochs are in `epochs` and the raw file wrapper is in `f`")
    from IPython import embed
    embed() 

if args.epoch_image:
    f.epoch_images()

if args.topo:
    f.topo()

if args.epoch_view:
    f.epoch_view()

if args.save_average or args.all:
    f.save_average()

