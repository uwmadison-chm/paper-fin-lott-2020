import os
import sys
import csv
import numpy as np
import logging
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
import mne

scriptDir = sys.path[0]

FREQUENCY_HIGHPASS_ARTIFACT = 0.5
FREQUENCY_HIGHPASS = 1
FREQUENCY_LOWPASS = 25

# Event IDs
UNKNOWN = 1
SAME = 2
DEVIANT = 3
EVENT_COLORS = {
    UNKNOWN: "blue",
    SAME: "green",
    DEVIANT: "red"
}
EVENT_ID = {
    "Unknown": UNKNOWN,
    "Same": SAME,
    "Deviant": DEVIANT
}


raw_file = "FM2004_HB1_R.bdf"
# How wide of a buffer around the crop do we want?
BUFFER_SECONDS = 2

raw = mne.io.read_raw_bdf(raw_file)

# Crop to the MMN section of the file
sfreq = raw.info['sfreq']
raw_events = mne.find_events(raw)


# TODO: Original script does weird event deletion, with this comment:
"""
On the South computer, it appears that there are many short (2 or 3
sample duration) events that are probably due to a problem in the MMN
.WAV files or noise / dropouts / flakiness in the cables connecting the
field laptop audio-out jack to the input on the BioSemi system. Whatever
the cause, they're bogus and they screw up analysis. So we'll just delete
them.
"""

# crop to first 2000 events
tstart = (raw_events[0,0] - (sfreq * BUFFER_SECONDS))
tstart_seconds = tstart / sfreq
tstop = (raw_events[1999,0] + (sfreq * BUFFER_SECONDS))
tstop_seconds = tstop / sfreq

# TODO: Ensure we got the expected length in seconds
raw.crop(tmin=tstart_seconds, tmax=tstop_seconds)

# Truncate the events list to the 2000 MMN events
events = raw_events[:2000].copy()
# Subtract out the amount we cropped by
#events[:,0] = events[:,0] - tstart

raw.load_data()
# Rename channels in raw based on actual electrode names
raw.rename_channels({'EXG1': 'cz', 'EXG2': 'mr', 'EXG3': 'ml', 'EXG4': 'fz', 'EXG5': 'pz', 'EXG6': 't8'})
# Reference electrodes
raw.set_eeg_reference(['mr', 'ml'])

# LAYOUT is the 2D display. Probably don't need this.
layout = mne.channels.read_layout(os.path.join(scriptDir, 'biosemi.lay'))

# TODO: Figure out how to apply montage to the raw data
montage = mne.channels.make_standard_montage('biosemi16')



# Now we need to load the right event tones and paste them into the event array
mmnToneDir = os.path.join(scriptDir, 'MMN_tone_sequences')
# TODO: Get from user input - assume false for now
is2013Initial = False

if is2013Initial:
    # 2 seconds to account for user accepting MMN .WAV file to be played
    # 303 seconds to play "silence" .WAV file.
    # 1 second to avoid interference between file playback routines
    # 12 seconds from start of MMN .WAV file to first tone being played
    secondsFromScriptStartToFirstTone = 2 + 303 + 1 + 12
    mmnToneFileStart = os.path.join(mmnToneDir, 'south', 'MMN_roving_with_trigger_dpdb02_seed_10')
    mmnToneFileEnd = '_31-May-2014_tone_sequence.txt'
else:
    # 2 seconds to account for user accepting MMN .WAV file to be played
    # 300 seconds for 300 second "silence" pause
    # 10 seconds from start of MMN .WAV file to first tone being played
    secondsFromScriptStartToFirstTone = 2 + 300 + 10;
    mmnToneFileStart = os.path.join(mmnToneDir, 'north', 'MMN_roving_with_trigger_dpdb01_seed_10')
    mmnToneFileEnd = '_21-Dec-2012_tone_sequence.txt'

# TODO: check if this matches Matlab script, may need to convert from GMT
recordedDate = datetime.fromtimestamp(raw.info['meas_date'][0])
actualStart = recordedDate + timedelta(seconds=tstart_seconds - secondsFromScriptStartToFirstTone)
doy = actualStart.timetuple().tm_yday
logging.info(f'Script start day of year = {doy}')
# TODO: Warn if close to noon, we may be using the wrong file
# We can FINALLY guess which tone sequence .TXT file to use for assigning
# tone IDs to events in the .BDF file.

# This determination is based on day of the year and time of day:
if (actualStart.hour >= 12):
    daySegment = 1
else:
    daySegment = 0
dayEven = doy % 2
whichSeq = 2 * dayEven + daySegment
mmnToneFileName = f"{mmnToneFileStart}{whichSeq}{mmnToneFileEnd}"
logging.info(f"Loading from {mmnToneFileName}")

with open(mmnToneFileName) as csvfile:
    reader = csv.reader(csvfile)
    tones = next(reader)
    # NOTE: there are way more than 2000 entries because of... legacy reasons. IGNORE

# Finally, we know enough to repair the events in the raw data
# and mark them same or deviant
numSameEvents = 0
numDeviantEvents = 0
for i in range(1, len(events)):
    if (tones[i] == tones[i - 1]):
        events[i,2] = SAME
        numSameEvents += 1
    else:
        events[i,2] = DEVIANT
        numDeviantEvents += 1

# ARTIFACT REJECTION! Press 'a' to start
fig = raw.plot(
    block=True,
    remove_dc=True,
    events=events,
    event_color=EVENT_COLORS,
    highpass=FREQUENCY_HIGHPASS_ARTIFACT,
    duration=5.0,
    order=[0, 3, 4, 5], # display only the 4 channels we care about
    scalings=dict(eeg=50e-6))

# Save the annotations somewhere
# TODO
raw.annotations.save('saved-annotations.csv')



# PHASE 2
# TODO
annot_from_file = mne.read_annotations('saved-annotations.csv')

# Actually filter
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
epochs.plot()

# HANGS
epochs.plot_image(evoked=True)


# TODO: how to plot these, plus the difference?

deviant = epochs["Deviant"].average()
same = epochs["Same"].average()

difference = mne.combine_evoked([same, -deviant], weights='equal')
difference.plot()

difference.plot_image()

same.plot(picks=['fz'], title='Fz Same')
deviant.plot(picks=['fz'], title='Fz Deviant')


# OLD STYLE, remove later

same = mne.Epochs(filtered_raw, event_id=SAME, **epochs_params)
same_avg = same.average()
same_avg.plot()

# This throws "no digitization points found"
same_avg.plot_topomap(times=[0.1], size=3., title="Topo", time_unit='s')
