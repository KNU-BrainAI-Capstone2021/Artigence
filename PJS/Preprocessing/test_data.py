import os
import mne
import sklearn
import numpy as np
import matplotlib.pyplot as plt
file = "/Users/jaeseongpark/Desktop/Dataset/files/S001/S001R04.edf"
file_path = os.path.join(file)
raw_data = mne.io.read_raw_edf(file_path)
raw_get_data = raw_data.get_data()
# you can get the metadata included in the file and a list of all channels:
info = raw_data.info
channels = raw_data.ch_names
# print("raw_get_data_shape")
# print(raw_get_data.shape)
# print("raw_get_data")
# print(raw_get_data)
# print("raw_data")
# print(raw_data)
print("info")
print(info)
# print("channels")
# print(channels)
"""
data 확인
plot - chan별로 data 확인,( duration = 몇초까지, n_channels = 볼 chan 수)
plot_psd - 추파수별 mv강
"""
# raw_data.plot(duration = 5, n_channels = 64)
# raw_data.plot_psd()


"""
ICA - preprocessing의 핵심 , noise(눈을 깜빡여서 나오는 신호, 심장박동 등)을 제거해주는 것 
"""
# ica = mne.preprocessing.ICA(n_components=20, random_state=97, max_iter=800)
# ica.fit(raw_data)
# ica.exclude = [1, 2]  # details on how we picked these are omitted here
# ica.plot_properties(raw_data, picks=ica.exclude)


mag_channels = mne.pick_types(raw_data.info, eeg=True)
print(mag_channels)
raw_data.plot(duration=60, order=mag_channels, proj=False,
         n_channels=len(mag_channels), remove_dc=False)


"""bandpass filter을 통해 Data filtering"""
raw_highpass = raw_data.load_data().filter(l_freq=10, h_freq=60)
print(raw_highpass.info)

"""이벤트 찾"""
# annotation이 있는경우
events, event_id = mne.events_from_annotations(raw_data)
print(events, event_id)

# stim_cahennel이 있는 경우
#events = mne.find_events(raw_data, stim_channel=None)
# print(events[:5])  # show the first 5
"""
def add_arrows(axes):
    # add some arrows at 60 Hz and its harmonics
    for ax in axes:
        freqs = ax.lines[-1].get_xdata()
        psds = ax.lines[-1].get_ydata()

        idx = np.searchsorted(freqs, 60)
        # get ymax of a small region around the freq. of interest
        y = psds[(idx - 4):(idx + 5)].max()
        ax.arrow(x=freqs[idx], y=y + 18, dx=0, dy=-12, color='red',
                 width=0.1, head_width=3, length_includes_head=True)


fig = raw_data.plot_psd(fmax=80, average=True)
add_arrows(fig.axes[:2])
"""
raw_downsampled = raw_data.copy().resample(sfreq=60)
events, event_id = mne.events_from_annotations(raw_downsampled)
print(events, event_id)
print(raw_downsampled.info)
# for data, title in zip([raw_data, raw_downsampled], ['Original', 'Downsampled']):
#     fig = data.plot_psd(average=True)
#     fig.subplots_adjust(top=0.9)
#     fig.suptitle(title)
#     plt.setp(fig.axes, xlim=(0, 300))


