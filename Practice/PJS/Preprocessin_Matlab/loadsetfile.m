MainPath = ['/Users/jaeseongpark/Desktop/Dataset/'];
addpath([MainPath 'eeglab2021.0']);
addpath '/Users/jaeseongpark/Desktop/Dataset/test/sub1'
eeglab;

%% Load .mat dat
% load_Data = load('/Users/jaeseongpark/Desktop/Dataset/test/sub1/s01.mat');
% EEG.data = [load_Data.eeg.imagery_left(1:64,:) load_Data.eeg.imagery_right(1:64,:)];
% EEG.srate = load_Data.eeg.srate;
% EEG.chanlocs = readlocs('biosemi64.ced');
% EEG=eeg_checkset(EEG);
% count = 1;
% 
% for i = 1:358400
%     if load_Data.eeg.imagery_event(i) == 1
%         EEG.event(count).type = 'left';
%         EEG.event(count).latency = i;
% %             EEG.event(count).epoch = count;
%          count = count+ 1;
%     end
% end
% for i = 1:358400
%     if load_Data.eeg.imagery_event(i) == 1
%         EEG.event(count).type = 'right';
%         EEG.event(count).latency = 358400+i;
% %             EEG.event(count).epoch = count;
%          count = count+ 1;
%     end
% end
% eeglab redraw;

%%
EEG = pop_loadset('s02_p.set','/Users/jaeseongpark/Desktop/Dataset/test/sub1/');
EEG=eeg_checkset(EEG);
EEG.chanlocs = readlocs('biosemi64.ced');
eeglab redraw;
%% Time-series plot
% figure();
% for i = 1 : 64,
%     plot(EEG_tuto.data(i,1:358400));
%     saveas(gcf,strcat('/Users/jaeseongpark/Desktop/Dataset/test/prepro_img/sub1/left/', int2str(i) ,'.png'));
% end