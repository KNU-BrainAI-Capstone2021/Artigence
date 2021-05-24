MainPath = ['/Users/jaeseongpark/Desktop/Dataset/'];
addpath([MainPath 'eeglab2021.0']);
eeglab;
EEG_tuto = pop_loadset('s01_pir.set', '/Users/jaeseongpark/Desktop/Dataset/test/sub1/');
