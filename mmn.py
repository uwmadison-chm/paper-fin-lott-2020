import os
import sys
import argparse
import logging

from eeg_shared import BDFWithMetadata

parser = argparse.ArgumentParser(description='Automate FMed study artifact rejection and analysis.')

parser.add_argument('input', help='Path to input file. If BDF, we assume artifact rejection. If not, we assume we should load previous artifacts.')

args = parser.parse_args()
raw_file = args.input

f = BDFWithMetadata(raw_file)
f.artifact_rejection()

sys.exit(0)


# Actually do the real filtering (happens in-place)
raw.filter(l_freq=FREQUENCY_HIGHPASS, h_freq=FREQUENCY_LOWPASS)

# Epoching...
picks = ['cz', 'fz', 'pz', 't8']
tmin, tmax = -0.1, 0.4

# TODO: Do automatic rejection of spikes? Can also pass a "too-flat" rejection if needed
epochs_params = dict(events=events, event_id=EVENT_ID,
                     tmin=tmin, tmax=tmax,
                     picks=picks, reject=None, flat=None)
                     #, proj=True, detrend=0)

epochs = mne.Epochs(raw, **epochs_params)

# Simple epoch plot
#epochs.plot()

# VERY SLOW
#epochs.plot_image(cmap="YlGnBu_r")


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

pick = standard.ch_names.index('cz')
fig = mne.viz.plot_compare_evokeds(evoked, picks=pick, colors=colors, split_legend=True, ci=0.95)

picks = ['cz', 'fz', 'pz', 't8']
fig = mne.viz.plot_compare_evokeds(evoked, picks=picks, colors=colors, combine='mean', ci=0.95)

