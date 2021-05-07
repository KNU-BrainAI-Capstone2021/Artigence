import os
import numpy as np
import mne
import sklearn
sample_data_folder = mne.datasets.sample.data_path()

sample_data_raw_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                                    'sample_audvis_filt-0-40_raw.fif')
raw = mne.io.read_raw_fif(sample_data_raw_file)

print(raw)
print(raw.info)

raw.plot_psd(fmax=40)
raw.plot(duration=5, n_channels=30)

# set up and fit the ICA
ica = mne.preprocessing.ICA(n_components=20, random_state=97, max_iter=800)
ica.fit(raw)
ica.exclude = [1, 2]  # details on how we picked these are omitted here
ica.plot_properties(raw, picks=ica.exclude)