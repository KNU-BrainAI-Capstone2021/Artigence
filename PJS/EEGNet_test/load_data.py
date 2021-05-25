import mne
from mne import io
dpath = "C:/Users/USER/OneDrive - knu.ac.kr/Data/s01_pir.set"
eeglab_raw  = mne.io.read_raw_eeglab(dpath)
print(eeglab_raw.annotations[1])
print(len(eeglab_raw.annotations))
print(set(eeglab_raw.annotations.duration))
print(set(eeglab_raw.annotations.description))
print(eeglab_raw.annotations.onset[0])
custom_mapping = {'left': 1, 'right': 2}
(events_from_annot,event_dict) = mne.events_from_annotations(eeglab_raw, event_id=custom_mapping)
print(event_dict)
print(events_from_annot[:5])
event_id = dict(left=1,right=2)
epochs = mne.Epochs(eeglab_raw, events_from_annot, event_id)
labels = epochs.events[:,-1]
X = epochs.get_data( ) *1000 # format is in (trials, channels, samples)
y = labels
