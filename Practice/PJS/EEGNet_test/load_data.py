import glob
import mne
from mne import io
dpath = "C:/Users/USER/OneDrive - knu.ac.kr/Data/"
file_list = glob.glob(dpath + "*.set")
X=[]
Y=[]
for file in file_list:
    eeglab_raw  = mne.io.read_raw_eeglab(file)
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
    tmin = 1
    tmax = 2.9980
    epochs = mne.Epochs(eeglab_raw, events_from_annot, event_id, tmin, tmax,baseline = None)
    labels = epochs.events[:,-1]
    data = epochs.get_data( ) *1000 # format is in (trials, channels, samples)
    X.extend(data)
    Y.extend(labels)
print(len(X))
print(len(Y))
