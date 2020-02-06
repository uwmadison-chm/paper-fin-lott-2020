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

# How wide of a buffer around the crop do we want?
# 1 second is enough with .5s epochs
BUFFER_SECONDS = 1

HIGHPASS_ARTIFACT = 0.5

# Event IDs
UNKNOWN = 1
STANDARD = 2
DEVIANT = 3
EVENT_COLORS = {
    UNKNOWN: "blue",
    STANDARD: "green",
    DEVIANT: "red"
}

class BDFWithMetadata():
    def __init__(self, path, kind, force=False, is_2013I=False):
        self.script_dir = sys.path[0]
        self.source_path = path
        self.kind = kind
        self.is_2013I = is_2013I

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
            if not force:
                # DIE unless they forced to save in current dir with a flag
                logging.critical("Data file not stored in expected raw-data/subjects directory, please run with --force to save masks and stuff to current directory")
                sys.exit(1)

        self.highpass_artifact = HIGHPASS_ARTIFACT
        if self.is_mmn():
            self.highpass = 1
            self.lowpass = 35
            self.event_id = {
                "Unknown": UNKNOWN,
                "Standard": STANDARD,
                "Deviant": DEVIANT
            }
        else:
            self.highpass = 100
            self.lowpass = 3000
            self.event_id = {
                "Unknown": UNKNOWN,
            }

        output_dir = p.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        self.output_path = str(dest_path)

    def is_mmn(self):
        return self.kind == "mmn"

    def load(self):
        # Do we have existing metadata?
        metadata = self.artifact_metadata_file()
        self.load_existing_metadata()
        self.load_existing_events()

        self.load_file(self.source_path)

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

    def locate_events(self, expected_events, expected_duration, kind):
        """
        Locate event chunks in file that match the given duration.

        expected_events: Integer number of events we want to find
        expected_duration: Duration in seconds that we want to find them in
        kind: User-visible sort of events we're looking for
        """
        sfreq = self.raw.info['sfreq']
        raw_events = mne.find_events(self.raw)

        skipped_events = 0
        looking = True

        while looking and len(raw_events) - skipped_events >= expected_events:
            tstart = (raw_events[skipped_events,0] - (sfreq * BUFFER_SECONDS))
            self.tstart_seconds = tstart / sfreq
            tstop = (raw_events[skipped_events+expected_events-1,0] + (sfreq * BUFFER_SECONDS))
            self.tstop_seconds = tstop / sfreq
            duration_seconds = self.tstop_seconds - self.tstart_seconds

            if duration_seconds > expected_duration - BUFFER_SECONDS*2 and \
                duration_seconds < expected_duration + BUFFER_SECONDS*2:
                logging.info(f"Found events at {tstart} with duration {duration_seconds} after skipping {skipped_events}")
                looking = False
            else:
                logging.debug(f"Skipped {skipped_events}, at {tstart} with {duration_seconds} (looking for {expected_events})")
                skipped_events += 1

        if looking:
            # Sorry if this sucks in actual use, bit of a rush to get this all working
            logging.warning(f"Could not find {expected_events} {kind} events automatically, skipped {skipped_events} while trying. Please scroll and find start and stop time in seconds manually!")
            # Temporarily set our events to the full list for plotting
            self.events = raw_events.copy()
            self.plot(False)
            self.tstart_seconds = input(f"Enter {kind} start time (in seconds): ")
            self.tstop_seconds = input(f"Enter {kind} stop time (in seconds): ")

        # Crop to those seconds
        self.raw.crop(tmin=self.tstart_seconds, tmax=self.tstop_seconds)
        self.tstart = self.tstart_seconds * sfreq
        self.tstop = self.tstop_seconds * sfreq

        # Truncate the events list to the ones we wanted,
        # keeping in mind we have to start at the index after the start
        index = np.searchsorted(raw_events[:,0], self.tstart)
        self.events = raw_events[index:index+expected_events].copy()

    def load_file(self, raw_file):
        self.raw = mne.io.read_raw_bdf(raw_file)

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
        
        if self.tstart_seconds and len(self.events) > 0:
            # If we already have information, use that
            self.raw.crop(tmin=self.tstart_seconds, tmax=self.tstop_seconds)
            first_run = False
        else:
            if self.is_mmn():
                # Crop to the MMN section of the file
                self.locate_events(2000, 1000, self.kind)
            else:
                # Crop to the ABR section of the file
                self.locate_events(4000, 200, self.kind)

        self.raw.load_data()
        # Rename channels in raw based on actual electrode names
        # This is based on FMed_Chanlocs_6channels.ced
        # EXG2 used to be called mr and EXG3 was ml,
        # they are renamed to O1 and O2
        # so that when we load the electrode montage below it matches

        if self.raw.info['nchan'] == 17:
            # Unclear why, but in the original files, the 6 channels are duplicated
            # MNE appends a number to differentiate the dupes
            self.raw.rename_channels({'EXG1-0': 'Cz', 'EXG2-0': 'O1', 'EXG3-0': 'O2', 'EXG4-0': 'Fz', 'EXG5-0': 'Pz', 'EXG6-0': 'T8'})
        else:
            self.raw.rename_channels({'EXG1': 'Cz', 'EXG2': 'O1', 'EXG3': 'O2', 'EXG4': 'Fz', 'EXG5': 'Pz', 'EXG6': 'T8'})

        # Reference electrodes on mastoids
        self.raw.set_eeg_reference(['O1', 'O2'])

        # Try to hack in some electrode location information into the raw.info
        montage = mne.channels.make_standard_montage('biosemi16')
        self.raw.set_montage(montage, raise_if_subset=False)
        
        if first_run:
            if self.is_mmn():
                # Figure out if tones are same or deviant
                self.load_event_tones_for_mmn()
            # We don't need to do any hacking of events for ABR
        
            # Now we automatically save out the cropping and events metadata
            self.save_metadata()
            self.save_events()

        # If previous annotations exist, read them
        self.load_annotations()


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

    def load_annotations(self):
        mask_path = self.artifact_mask_file()
        if os.path.exists(mask_path):
            logging.info(f"Loading existing artifact annotations from {mask_path}")
            a = mne.read_annotations(mask_path)
            self.raw.set_annotations(a)


    def load_event_tones_for_mmn(self):
        logging.info(f"Determining MMN event types")
        # Now we need to load the right event tones and paste them into the event array
        mmnToneDir = os.path.join(self.script_dir, 'MMN_tone_sequences')
        is2013Initial = self.is_2013I

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
        noon = actualStart.replace(hour=12, minute=0, second=0)
        if abs(noon - actualStart).seconds < 600:
            logging.warning(f"WARNING: start time {actualStart} is close to noon, so the event tone discovery may be wrong")

        # We can now guess which tone sequence .TXT file to use for assigning
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
        return self.output_path.replace(".bdf", f".{self.kind}_artifact_mask.csv")

    def plot_output_path(self, name):
        return self.output_path.replace(".bdf", f".{self.kind}_{name}.png")

    def artifact_metadata_file(self):
        return self.output_path.replace(".bdf", f".{self.kind}_artifact_metadata.yaml")
    
    def events_file(self):
        return self.output_path.replace(".bdf", f".{self.kind}_events.npy")

    def plot(self, block=True, display_all=False):
        duration = 5.0
        if display_all:
            order = None
        else:
            order = [0, 3, 4, 5], # display only the 4 channels we care about
        self.raw.plot(
            block=block,
            remove_dc=True,
            events=self.events,
            event_color=EVENT_COLORS,
            highpass=self.highpass_artifact,
            duration=duration,
            order=order,
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
        self.raw.filter(l_freq=self.highpass, h_freq=self.lowpass, fir_design='firwin')

        # Epoching...
        picks = ['Cz', 'Fz', 'Pz', 'T8']
        if self.is_mmn():
            tmin, tmax = -0.1, 0.4
        else:
            tmin, tmax = -0.01, 0.015

        epochs_params = dict(events=self.events, event_id=self.event_id,
                            tmin=tmin, tmax=tmax,
                            picks=picks, reject=None, flat=None)

        self.epochs = mne.Epochs(self.raw, **epochs_params)
        return self.epochs


    def save_figure(self, fig, name):
        filename = self.plot_output_path(name)
        fig.savefig(filename)
        logging.info(f"Saved {name} plot to {filename}")

    
    def epoch_images(self):
        logging.warning("Plotting epoch image, VERY SLOW")
        fig = self.epochs.plot_image(cmap="YlGnBu_r", group_by=None,
                picks=['Cz', 'Fz', 'T8', 'Pz'], show=False)
        save_figure(fig, "epochs")

    def epoch_view(self):
        logging.info("Loading epoch viewer...")
        epochs.plot(block=True)

    def psd(self, high_freq):
        # Spectral density is go!
        # https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.plot_psd
        #fig = self.raw.plot_psd(0, high_freq, average=False, show=False, estimate='power')
        fig = self.raw.plot_psd(0, high_freq, area_mode='range', show=False, n_fft=60000)
        title = f"Power spectral density for {self.kind}"
        self.save_figure(fig, f"psd_to_{high_freq}")

    def topo(self):
        epochs = self.epochs

        joint_kwargs = dict(ts_args=dict(time_unit='s'),
                        topomap_args=dict(time_unit='s'),
                        show=False)
        if self.is_mmn():
            deviant = epochs["Deviant"].average()
            standard = epochs["Standard"].average()
            fig1 = deviant.plot_joint(**joint_kwargs)
            self.save_figure(fig1, "deviant_average")
            fig2 = standard.plot_joint(**joint_kwargs)
            self.save_figure(fig2, "deviant_average")
        else:
            average = epochs.average()
            fig = average.plot_joint(**joint_kwargs)
            self.save_figure(fig, "epoch_average")
