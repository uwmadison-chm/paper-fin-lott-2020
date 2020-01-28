import os
import sys
import csv
import numpy as np
import logging
import yaml
from pathlib import Path
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
import mne

FREQUENCY_HIGHPASS_ARTIFACT = 0.5
FREQUENCY_HIGHPASS = 1
FREQUENCY_LOWPASS = 35

# How wide of a buffer around the crop do we want?
BUFFER_SECONDS = 2

# Event IDs
UNKNOWN = 1
STANDARD = 2
DEVIANT = 3
EVENT_COLORS = {
    UNKNOWN: "blue",
    STANDARD: "green",
    DEVIANT: "red"
}
EVENT_ID = {
    "Unknown": UNKNOWN,
    "Standard": STANDARD,
    "Deviant": DEVIANT
}


class BDFWithMetadata():
    def __init__(self, path):
        self.script_dir = sys.path[0]
        self.source_path = path

        # Determine if source path is in the standard /study/thukdam/raw-data location or not
        p = Path(path).resolve()
        parts = p.parts
        if "raw-data" in parts and "subjects" in parts:
            # If "raw-data", "subjects" is in path, replace "raw-data" with "intermediate_data", "eeg"
            dest = list(parts)
            raw_index = dest.index('raw-data')
            dest[raw_index] = "intermediate_data"
            dest.insert(raw_index+1, "eeg")
            # Create directory if it doesn't exist
            dest_path = Path(os.path.join(*dest))
        else:
            dest_path = p

        self.output_dir = p.parent
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_path = str(dest_path)

    def load(self):
        # Do we have existing metadata?
        metadata = self.artifact_metadata_file()
        self.load_existing_metadata()
        self.load_existing_events()

        self.load_mmn(self.source_path)

    def load_existing_metadata(self):
        metadata = self.artifact_metadata_file()
        if os.path.exists(metadata):
            with open(metadata) as file:
                data = yaml.load(file, Loader=yaml.FullLoader)
                self.tstart_seconds = data['tstart_seconds']
                self.tstop_seconds = data['tstop_seconds']
                logging.info(f"Loaded existing start {self.tstart_seconds} and end {self.tstop_seconds} from {metadata}")
        else:
            self.tstart_seconds = None
            self.tstop_seconds = None

    def load_existing_events(self):
        e = self.events_file()
        if os.path.exists(e):
            self.events = np.load(e)
            logging.info(f"Loaded {len(self.events)} from events numpy file {e}")
        else:
            self.events = []

    def load_mmn(self, raw_file):
        raw = mne.io.read_raw_bdf(raw_file)
        self.raw = raw

        # TODO: Original script does weird event deletion, with this comment:
        """
        On the South computer, it appears that there are many short (2 or 3
        sample duration) events that are probably due to a problem in the MMN
        .WAV files or noise / dropouts / flakiness in the cables connecting the
        field laptop audio-out jack to the input on the BioSemi system. Whatever
        the cause, they're bogus and they screw up analysis. So we'll just delete
        them.
        """

        first_run = True
        
        # Crop to the MMN section of the file
        if self.tstart_seconds and len(self.events) > 0:
            # If we already have information, use that
            raw.crop(tmin=self.tstart_seconds, tmax=self.tstop_seconds)
            first_run = False
        else:
            # NO existing info, time to get fancy!
            sfreq = raw.info['sfreq']
            raw_events = mne.find_events(raw)

            # Crop to first 2000 events
            tstart = (raw_events[0,0] - (sfreq * BUFFER_SECONDS))
            self.tstart_seconds = tstart / sfreq
            tstop = (raw_events[1999,0] + (sfreq * BUFFER_SECONDS))
            self.tstop_seconds = tstop / sfreq

            duration_seconds = self.tstop_seconds - self.tstart_seconds
            if duration_seconds < 1000 - BUFFER_SECONDS*2 or \
            duration_seconds > 1000 + BUFFER_SECONDS*2:
                # TODO: Yeah sorry if this sucks in actual use, bit of a rush to get this all working
                logging.warning(f"Could not find MMN events automatically. Please scroll and find start and stop time in seconds manually!")
                # Temporarily set our events to the full list for plotting
                self.events = raw_events.copy()
                self.plot(False)
                self.tstart_seconds = input("MMN start time (in seconds)")
                self.tstop_seconds = input("MMN stop time (in seconds)")

            raw.crop(tmin=self.tstart_seconds, tmax=self.tstop_seconds)

            # Truncate the events list to the 2000 MMN events
            self.events = raw_events[:2000].copy()

        raw.load_data()
        # Rename channels in raw based on actual electrode names
        # This is based on FMed_Chanlocs_6channels.ced
        raw.rename_channels({'EXG1': 'cz', 'EXG2': 'mr', 'EXG3': 'ml', 'EXG4': 'fz', 'EXG5': 'pz', 'EXG6': 't8'})
        # Reference electrodes
        raw.set_eeg_reference(['mr', 'ml'])

        # LAYOUT is the 2D display. Probably don't need this.
        #layout = mne.channels.read_layout(os.path.join(self.script_dir, 'biosemi.lay'))
        
        # It would be nice figure out how to apply montage to the raw data,
        # so we could get averaged displays in spatial orientation...
        # but since we only have 6 channels, very unclear how to apply
        # any of the standard montages
        #montage = mne.channels.make_standard_montage('biosemi16')

        self.raw = raw

        if first_run:
            # Figure out if tones are same or deviant
            self.load_event_tones()
        
            # Now we automatically save out the cropping and events metadata
            self.save_metadata()
            self.save_events()

        # If previous annotations exist, read them
        mask_path = self.artifact_mask_file()
        if os.path.exists(mask_path):
            logging.info(f"Loading existing artifact annotations from {mask_path}")
            a = mne.read_annotations(mask_path)
            self.raw.set_annotations(a)

    def save_metadata(self):
        data = {
            'tstart_seconds': int(self.tstart_seconds),
            'tstop_seconds': int(self.tstop_seconds),
            'source_path': int(self.tstop_seconds),
        }
        with open(self.artifact_metadata_file(), 'w') as file:
            yaml.dump(data, file)

    def save_events(self):
        np.save(self.events_file(), self.events)


    def load_event_tones(self):
        logging.info(f"Determining MMN event types")
        # Now we need to load the right event tones and paste them into the event array
        mmnToneDir = os.path.join(self.script_dir, 'MMN_tone_sequences')
        # TODO: Get from user input, or, better, from subject metadata - assume false for now
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
        recordedDate = datetime.fromtimestamp(self.raw.info['meas_date'][0])
        actualStart = recordedDate + timedelta(seconds=self.tstart_seconds - secondsFromScriptStartToFirstTone)
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
        logging.info(f"Loading tones from {mmnToneFileName}")

        with open(mmnToneFileName) as csvfile:
            reader = csv.reader(csvfile)
            tones = next(reader)
            # NOTE: there are way more than 2000 entries because of... legacy reasons. IGNORE

        # Finally, we know enough to repair the events in the raw data
        # and mark them same or deviant
        numSameEvents = 0
        numDeviantEvents = 0
        for i in range(1, len(self.events)):
            if (tones[i] == tones[i - 1]):
                self.events[i,2] = STANDARD
                numSameEvents += 1
            else:
                self.events[i,2] = DEVIANT
                numDeviantEvents += 1
        logging.info(f"Determined {numSameEvents} same events and {numDeviantEvents} deviant events")

    def artifact_mask_file(self):
        return self.output_path.replace(".bdf", ".artifact_mask.csv")

    def artifact_metadata_file(self):
        return self.output_path.replace(".bdf", ".artifact_metadata.yaml")
    
    def events_file(self):
        return self.output_path.replace(".bdf", ".events.npy")

    def plot(self, block=True):
        self.raw.plot(
            block=block,
            remove_dc=True,
            events=self.events,
            event_color=EVENT_COLORS,
            highpass=FREQUENCY_HIGHPASS_ARTIFACT,
            duration=5.0,
            order=[0, 3, 4, 5], # display only the 4 channels we care about
            scalings=dict(eeg=50e-6))

    def artifact_rejection(self):
        logging.info("View loaded. Ready for artifact rejection! Press 'a' to start, add a label, and then drag on the graph. Close the view window to continue.")
        self.plot()

        mask_path = self.artifact_mask_file()

        # Save the annotations
        if len(self.raw.annotations) > 0:
            self.raw.annotations.save(mask_path)

    def build_epochs(self):
        # Actually do the real final filtering (happens in-place)
        self.raw.filter(l_freq=FREQUENCY_HIGHPASS, h_freq=FREQUENCY_LOWPASS)

        # Epoching...
        picks = ['cz', 'fz', 'pz', 't8']
        tmin, tmax = -0.1, 0.4

        # TODO: Do automatic rejection of spikes? Can also pass a "too-flat" rejection if needed
        epochs_params = dict(events=self.events, event_id=EVENT_ID,
                             tmin=tmin, tmax=tmax,
                             picks=picks, reject=None, flat=None)
                             #, proj=True, detrend=0)

        self.epochs = mne.Epochs(self.raw, **epochs_params)
        return self.epochs
