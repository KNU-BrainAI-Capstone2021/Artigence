close all
clear
clc

MainPath = ['/Users/jaeseongpark/Desktop/Dataset/'];
addpath([MainPath 'eeglab2021.0']);
addpath '/Users/jaeseongpark/Desktop/Dataset/test/sub1'
eeglab;


%% Load data
EEG = pop_loadset('s01_pir.set','/Users/jaeseongpark/Desktop/Dataset/test/sub1/');
EEG.chanlocs = readlocs('biosemi64.ced');
% EEG.chanlocs(57:64) = []
% EEG.chanlocs(33:38) = []
% EEG.chanlocs(20:30) = []
% EEG.chanlocs(1:7) = []
% EEG.data(57:64,:) = []
% EEG.data(33:38,:) = []
% EEG.data(20:30,:) = []
% EEG.data(1:7,:) = []
EEG.data = normalize(EEG.data,'norm',1)
eeglab redraw;
