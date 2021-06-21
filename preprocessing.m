%% SET PATH
MainPath = ['D:\Artigence'];
addpath([MainPath '\eeglab2021.0']);
dPath=['D:\Artigence\data\'];
eeglab;

pop_editoptions( 'option_savetwofiles', 1,'option_single', 0);

cd(dPath);
subStruct=dir;
subStruct = subStruct(cellfun(@any,strfind({subStruct.name},'sub')));
nSub = length(subStruct);

lowCut=1;
highCut=50;

%% MAIN LOOP

 for iSub = 1 : nSub
        
    subId = subStruct(iSub).name;
    fileStruct = dir([subId  '/*.mat']);
    fileId = fileStruct(1).name;
    load_path = strcat(dPath,subId,"/",fileId);
    
    disp(['Sub ' num2str(iSub) ' Loading......... ' fileId]);
    load_Data = load(load_path);
    load_Data = struct(load_Data.eeg);
    
%% checkset
   
    EEG.data = [load_Data.imagery_left(1:64,:) load_Data.imagery_right(1:64,:)];
    EEG.srate = load_Data.srate;
    EEG.chanlocs = readlocs('biosemi64.ced');
    EEG=eeg_checkset(EEG); %***** 
       
%% Load event

     count = 1;
     
    for i = 1:358400
        if load_Data.imagery_event(i) == 1
            EEG.event(count).type = 'left';
            EEG.event(count).latency = i;
            EEG.event(count).epoch = count;
            count = count+ 1;
        end
    end
    for i = 1:358400 
        if load_Data.imagery_event(i) == 1
            EEG.event(count).type = 'right';
            EEG.event(count).latency = 358401 + i;
            EEG.event(count).epoch = count;
            count = count+ 1;
        end
    end
   
%% BPF
    
    disp(['band-pass filtering from '  num2str(lowCut) ' to ' num2str(highCut)  ' Hz']);
    EEG = pop_eegfiltnew(EEG, lowCut, highCut);
     
    
%% ASR
   
    EEG.etc.historychanlocs=EEG.chanlocs;
    EEG.etc.historychaninfo=EEG.chaninfo;
   
    EEG = clean_rawdata(EEG,5,-1,0.8,4,5,-1); % default setting
    EEG.etc.badchan=find(EEG.etc.clean_channel_mask==0); %Bad chananel information from ASR
    
    EEG.etc.originalEEG=EEG; % keep origianl EEG before interpolation
    EEG = pop_interp(EEG, EEG.etc.historychanlocs, 'spherical');
    
    
%% CAR

    EEG = pop_reref( EEG, []);
        
%% ICA

    EEG.rank=rank(double(EEG.data));
    EEG = pop_runica(EEG,'extended',1,'pca',EEG.rank);
        
%% IC Label
 
    EEG = pop_iclabel(EEG, 'default');
  
    rejIdx=[];
    cutProb=0.5; % 50 percent
    for iICA = 1 : EEG.rank
        [maxProb maxIdx]= max(EEG.etc.ic_classification.ICLabel.classifications(iICA, :));
        % 1: brain / 2: Muscle / 3: Eye / 4: Heart / 5: Line Noise / 6: Channel Noise / 7: Other
        if maxIdx ~= 1 && maxIdx ~= 7 && maxProb > cutProb
            rejIdx = [rejIdx iICA];
        end
    end
    
    EEG.etc.rejIdx = rejIdx;
    EEG = pop_subcomp( EEG, rejIdx, 0);
      
    pop_saveset(EEG,'filepath',[dPath subId],'filename',[fileId(1:end-4) '_pir.set']);
    
    EEG = eeg_emptyset
end