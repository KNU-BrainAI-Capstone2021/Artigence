import numpy as np # numpy : 파이썬 라이브러리 / 벡터 및 행렬 연산에 있어서 편리 / matplotlib 라이브러리의 기반으로 사용됨 / array(행렬) 단위로 데이터를 관리 및 연산 수행

# mne imports / mne : EEG 등 neurophysiological한 데이터들을 분석하기 위한 파이썬 라이브러리
import mne # mne 라이브러리 안에 있는 파일들을 사용할 수 있게 함
from mne import io # mne.io 대신 간략하게 함수명 io만을 사용하겠다. / mne.io : raw data를 읽는데 필요한 IO module
from mne.datasets import sample # mne.datasets : 원격 데이터셋을 가져오는 함수들

# EEGNet-specific imports
from EEGModels import EEGNet 
from tensorflow.keras import utils as np_utils # Public API(개방형 API. 모두에게 공개.) for tensorflow.keras.utils namespace
from tensorflow.keras.callbacks import ModelCheckpoint # Callback to save the Keras model or model weights at some frequency.
from tensorflow.keras import backend as K # keras 백엔드 API
'''
- tensorflow : 머신러닝 프레임워크 (디테일한 조작 가능) / keras : tensorflow 위에서 동작하는 프레임워크(간단한 것 제작. 빠른 시간 내에 프로토타이핑을 하고자 할 때.) 
=> tensorflow.keras : tensorflow 안에서 keras 사용 (주요 틀을 tensorflow.keras로 구현하고 tensorflow로 그 내용을 채워넣는 방법을 사용하는 것이 가장 좋은 옵션이 될 수 있다.)
- API(application programming interface) : 애플리케이션 소프트웨어를 구축하고 통합하기 위한 정의 및 프로토콜 세트. / 운영체제와 응용프로그램 사이의 통신에 사용되는 언어나 메시지 형식. / 프로그램들이 서로 상호작용하는 것을 도와주는 매개체. for 프로그래머
- namespace : 모든 변수 이름과 함수 이름을 겹치지 않게 정하기 어려우므로 하나의 이름이 통용될 수 있는 범위를 제한해서 소속된 namespace가 다르면 같은 이름이 다른 개체를 가리킬 수 있도록 해줌.
'''

# PyRiemann imports / PyRiemann : scikit-learn API 기반의 파이썬 머신 러닝 라이브러리 / Covariance Matrices(공분산 행렬 ; 2개의 확률변수의 선형 관계)의 조작과 그를 통한 다변량 신호들(특히 바이오 신호. ex. EEG)의 분류를 위한 라이브러리
# Decoding applied to EEG data in sensor space decomposed using Xdawn. 
# After spatial filtering, covariances matrices are estimated, then projected in the tangent space and classified with a logistic regression.
from pyriemann.estimation import XdawnCovariances # Compute xdawn, project the signal and compute the covariances
from pyriemann.tangentspace import TangentSpace # Tangent space project TransformerMixin
from pyriemann.utils.viz import plot_confusion_matrix # Plot the confusion matrix
from sklearn.pipeline import make_pipeline # Construct a Pipeline from the given estimators
from sklearn.linear_model import LogisticRegression # Logistic Regression classifier.
'''
- TransformerMixin : Mixin class for all transformers in scikit-learn. 이 클래스는 fit(),transform()메서드를 나만의 변환기에 생성하였을 경우 마지막 메서드인 fit_transform()를 자동으로 생성
- Pipeline : 여러가지의 변환기들을 하나로 연결
- Logistic Regression (로지스틱 회귀) : 변수를 입력할 경우 이를 0~1의 범위로 변환 후 이를 통해서 확률값을 계산하고 마지막으로 분류(어떤 클래스에 대한 확률이 가장 높은지 계산)하는 지도 학습 알고리즘.
'''

# tools for plotting confusion matrices (confusion matrix : 지도 학습으로 훈련된 분류 알고리즘의 성능을 시각화한 표 / 행과 열은 각각 예측 된 클래스의 인스턴스와 실제 클래스의 인스턴스를 나타냄)
from matplotlib import pyplot as plt # matplotlib : 파이썬 기반의 시각화 라이브러리. 여러 가지 그래프를 그려주는 함수들 내장.

# while the default tensorflow ordering is 'channels_last' we set it here 
# to be explicit in case if the user has changed the default ordering
K.set_image_data_format(‘channels_last’) # Sets the value of the image data format convention.

## Process, filter and epoch the data ##
data_path = sample.data_path() # mne.datasets.sample.data_path : 샘플 데이터셋의 복사본의 로컬 경로를 가져오는 함수

# Set parameters and read data
raw_fname = data_path + ‘/MEG/sample/sample_audvis_filt_0-40_raw.fif’
event_fname = data_path + ‘/MEG/sample/sample_audvis_filt_0-40_raw-eve.fif’
tmin, tmax = -0., 1 
event_id = dict(aud_l=1, aud_r=2, vis_l=3, vis_r=4)

# Setup for reading the raw data
raw = io.Raw(raw_fname, preload=False, verbose=False)
raw.filter(2, None, method=’iir’) # replace baselining with high-pass
events = mne.read_events(event_fname)
'''
mne.io.Raw : FIF 파일 형식의 raw data.
- raw_fname (str) : load할 raw file 파일명.
 파일명이 raw.fif, raw_sss.fif, raw_tsss.fif, _meg.fif, _eeg.fif, _ieeg.fif로 끝나야 오류 발생 안 함
- preload (bool or str)(default = False) 
- verbose (bool, str, int, or None) : 상세한 logging을 출력할지 말지 조정함. 
 False = 'WARNING'

mne.io.raw.filter(l_freq, h_freq, method='fir') : 채널의 subset(부분 집합)을 필터링하는 함수
- l_freq (float, None) :  IIR 필터 : the lower cutoff frequency
- h_freq (float, None) : None :  the data are only high-passed. 
⇒⇒ l_freq is not None and h_freq is None: high-pass filter
- method (str) :
 ‘iir’ ⇒ IIR forward-backward filtering 사용 (via filtfilt)

mne.read_events : FIF 파일 또는 텍스트 파일에서 이벤트를 읽어오는 함수
- event_fname (str) : input file 파일명. 
 확장자가 .fif일 경우, FIF 파일로 간주되고, 그 외의 경우, 텍스트 파일로 간주되어 이벤트를 읽어옴
'''

# Set up pick list
raw.info[‘bads’] = [‘MEG 2443’] # set bad channels (Mark this channel['MEG 2443'] as bad)
picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude=’bads’) # mne.pick_types : type과 names로 채널 선정
'''
- raw.info (dict) : the measurement information. 
- meg = False , stim = False , eog = False , eeg = True : include EEG channels
- exclude = 'bads' (list of str, str) : raw.info['bads']에 속한 채널들을 exclude(제외)
'''

# Read epochs / epoch : tmin ~ tmax (default : -0.2s ~ 0.5s). 실험(측정)이 1회 진행되는 구간
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=False, picks=picks, baseline=None, preload=True, verbose=False) 
labels = epochs.events[:, -1]
'''
mne.Epochs : raw 인스턴스로부터 추출된 epochs.
- raw (raw object) : An instance of Raw.
- events (array of int) : 
 read_events 함수에 의해 반환됨. 
 event_id에 지정된 event와 불일치 ⇒ drop log에 'IGNORED' 표시됨.
- event_id (int, list of int, dict, None) : 고려할 event의 id
 dict인 경우, key를 사용하여 해당 이벤트에 access 가능. / 예: dict(auditory=1, visual=3)
 int인 경우, id를 string(문자열)로 하는 dict가 생성됨.
 list인 경우, list에 지정된 id를 가진 모든 event들이 사용됨.
 None인 경우, 모든 event들이 사용됨. / event id integers(정수)에 해당하는 문자열 정수 이름으로 dict가 생성됨.
- tmin (float)(default = -0.2) : Start time of the epochs (초 단위)
- tmax (float)(default = 0.5) : End time of the epochs (초 단위)
- proj = False (bool) : no projections will be applied which is the recommended value if SSPs are not used for cleaning the data.
- picks (str, list, slice, None) : 포함할 채널들. 
 Can also be the string values “all” to pick all channels, or “data” to pick data channels.
 정수의 리스트와 슬라이스는 채널 인덱스로 해석됨.
 채널 타입 리스트일 경우(예 : ['meg', 'eeg']), 채널 타입 문자열이 해당 타입의 채널을 선정.
 채널명 리스트일 경우 (예 : ['MEG0111', 'MEG2623']) , 지정된 채널을 선정.
 문자열 "all"일 경우 모든 채널 선택,
 None ⇒ 모든 채널 선택
 info['bads']의 채널은 채널 이름 또는 인덱스가 명시적으로 제공된 경우에 포함됨.
- baseline (None, tuple of length 2) : baseline correction을 적용할 때 baseline(기준선)
tuple(a,b) 인 경우 ⇒ a≤t≤b (초 단위) (양 끝 점을 포함하여 a부터 b까지)
a=None ⇒ 데이터의 시작부터
b=None ⇒ 구간의 끝까지
'''

# extract raw data. scale by 1000 due to scaling sensitivity in deep learning
X = epochs.get_data()*1000 # format is in (trials, channels, samples)
y = labels
'''
get_data(picks=None, item=None) : Get all epochs as a 3D array.
- picks : channels to include. / None (default) ⇒ pick all channels
- item : the items to get. / None (default) ⇒ an alias for slice(None).
⇒ Returns  data array of shape (n_epochs, n_channels, n_times) : A view on epochs data.
'''

kernels, chans, samples = 1, 60, 151
'''
- kernels : ???????
(저차원 공간(low dimensional space)을 고차원 공간(high dimensional space)으로 매핑해주는 작업을 커널 트릭(Kernel Trick)이라고 합니다저차원 공간(low dimensional space)을 고차원 공간(high dimensional space)으로 매핑해주는 작업을 커널 트릭(Kernel Trick)이라고 함. 커널 트릭을 활용하여 먼저 고차원 공간에서의 linear separable line 구한 뒤 저차원 공간에서의 non linear separable line 구할 수 있음.)
- chans : number of EEG channels
- samples : number of time points in the EEG data
'''

# take 50/25/25 percent of the data to train/validate/test
# 훈련셋 검증셋 테스트셋 분리
X_train = X[0:144,]
Y_train = y[0:144]
# Training set : 모델을 학습
X_validate = X[144:216,]
Y_validate = y[144:216]
# Validation set : training set으로 만들어진 모델의 성능을 측정
X_test = X[216:,]
Y_test = y[216:]
# Test set : validation set으로 사용할 모델이 결정 된 후, 마지막으로 딱 한번 해당 모델의 예상되는 성능을 측정하기 위해 사용
'''
⇒ overfitting 방지 (overfitting : 모델이 내가 가진 학습 데이터에 너무 과적합되도록 학습한 나머지, 이를 조금이라도 벗어난 케이스에 대해서는 예측율이 현저히 떨어짐 / 오버피팅된 모델은 훈련 데이터에서는 100%에 가까운 성능을 내지만 테스트 데이터에서는 성능이 굉장히 떨어질 수 있음) 방지
⇒ 성능을 높히는 것과 오버피팅을 막는 것 사이의 균형을 잘 지켜야
'''

# convert labels to one-hot encodings.
# 데이터셋 전처리 : one-hot 인코딩
# keras.utils.np_utils.to_categorical : one-hot encoding을 해주는 함수
Y_train = np_utils.to_categorical(Y_train-1)
Y_validate = np_utils.to_categorical(Y_validate-1)
Y_test = np_utils.to_categorical(Y_test-1)
'''
one-hot encoding? 
- 머신은 숫자를 이해하지 텍스트를 이해하진 못한다 ⇒ 모든 문자열 값들을 숫자 형으로 인코딩하는 전처리 작업
- 10진 정수 형식을 특수한 2진 바이너리 형식으로 변경하는 parameter
- 표현하고 싶은 단어의 인덱스에 1의 값을 부여하고, 다른 인덱스에는 0을 부여하는 단어의 벡터 표현 방식
- 데이터에 연속적인 특성(순서)이 없다는 것을 컴퓨터에게 확실하게 알려주는 과정
'''

# convert data to NHWC (trials, channels, samples, kernels) format. Data
# NHWC = Num_samples x Height x Width x Channels
# contains 60 channels and 151 time-points. Set the number of kernels to 1.
X_train = X_train.reshape(X_train.shape[0], chans, samples, kernels)
X_validate = X_validate.reshape(X_validate.shape[0], chans, samples, kernels)
X_test = X_test.reshape(X_test.shape[0], chans, samples, kernels)
'''
reshape :  give a new shape to an array without changing its data. 배열의 차원을 수정하는 함수.
데이터를 Keras에 맞게 변환해주는 과정. Keras에서 CNN을 구현할 때 [pixels][width][height] 형태로 input을 받으므로, numpy의 reshape함수를 이용하여 X vector를 [pixels][width][height] 형태로 바꿔줌.
'''

print(‘X_train shape:’, X_train.shape)
print(X_train.shape[0], ‘train samples’)
print(X_test.shape[0], ‘test samples’)
