import sys
import argparse
import logging
import coloredlogs
import mne

from eeg_shared import BDFWithMetadata

parser = argparse.ArgumentParser(description='Automate FMed study artifact rejection and analysis of MMN.')

parser.add_argument('input', help='Path to input file.')
parser.add_argument('-v', '--verbose', action='count', default=0)
parser.add_argument('--view', action='store_true', help="Allow viewing file and editing artifact mask")
parser.add_argument('--sminusd', metavar='ELECTRODE', action='store', help="Standard minus deviant of a specified electrode (cz, fz, t8, pz)")
parser.add_argument('--sminusd-mean', action='store_true', help="Mean standard minus deviant across all 4 electrodes")
parser.add_argument('--epoch-image', action='store_true', help="Very slow colormap image of epochs")
parser.add_argument('--epoch-view', action='store_true', help="Simple linear view of epochs, default end view")

args = parser.parse_args()

if args.verbose > 0:
    coloredlogs.install(level='DEBUG')
else:                       
    coloredlogs.install(level='INFO')
  

raw_file = args.input

f = BDFWithMetadata(raw_file)
f.load()
if args.view:
    f.artifact_rejection()

epochs = f.build_epochs()


if args.sminusd or args.sminusd_mean:
    # Plot standard and deviant on one figure, plus plot the difference, like original matlab
    deviant = epochs["Deviant"].average()
    standard = epochs["Standard"].average()

    difference = mne.combine_evoked([deviant, -standard], weights='equal')

    evoked = dict()
    evoked["Standard"] = standard
    evoked["Deviant"] = deviant
    evoked["Difference"] = difference

    colors = dict(Standard="Green", Deviant="Red", Difference="Black")

    # TODO: Figure out what we need to change about the evoked data so we get confidence intervals displayed - possibly MNE problems?
    # @agramfort in mne-tools/mne-python gitter said: "to have confidence intervals you need repetitions which I think is a list of evoked or not epochs you need to pass" 

    if args.sminusd:
        pick = standard.ch_names.index(args.sminusd)
        if pick:
            fig = mne.viz.plot_compare_evokeds(evoked, picks=pick, colors=colors, split_legend=True, ci=0.95)
        else:
            logging.warning(f"Could not find electrode '{args.sminusd}'")

    if args.sminusd_mean:
        picks = ['cz', 'fz', 'pz', 't8']
        fig = mne.viz.plot_compare_evokeds(evoked, picks=picks, colors=colors, combine='mean', ci=0.95)

    sys.exit()


if args.epoch_image:
    # VERY SLOW
    logging.warning("Plotting epoch image, VERY SLOW")
    epochs.plot_image(cmap="YlGnBu_r")
    sys.exit()

# args.epoch_view is the default
logging.info("Loading epoch view...")
epochs.plot(block=True)
sys.exit()

