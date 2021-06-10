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
%%chan_labels = {'FP1','AF7','AF3','F1','F3','F5','F7','FT7','FC5','FC3','FC1','C1','C3','C5','T7','TP7','CP5','CP3','CP1','P1','P3','P5','P7','P9','PO7','PO3','O1','Iz','Oz','POz','Pz','CPz','Cz','FCz','Fz','AFz','FFz','FP2','AF8','AF4','F2','F4','F6','F8','FT8','FC6','FC4','FC2','C2','C4','C6','T8','TP8','CP6','CP4','CP2','P2','P4','P6','P8','P10','PO8','PO4','O2'};

%% MAIN LOOP

for iSub = 1 : nSub
    
    %iSub = 1   
    subId = subStruct(iSub).name;
    fileStruct = dir([subId  '/*.mat']);
    fileId = fileStruct(1).name;
    load_path = strcat(dPath,subId,"/",fileId);
    
    disp(['Sub ' num2str(iSub) ' Loading......... ' fileId]);
    load_Data = load(load_path);
    load_Data = struct(load_Data.eeg);

 %%------------------------------------------------------------- 
 
%% Downsampling
%     if load_Data.srate >= 1024
%     load_Data = pop_resample(load_Data, 512);
%     end 
    
%% checkset
    % Make checkset data/srate/chanlocs **반드시 넣어줘야함
    EEG.data = [load_Data.imagery_left(1:64,:) load_Data.imagery_right(1:64,:)];
    EEG.srate = load_Data.srate;
    EEG.chanlocs = readlocs('biosemi64.ced');
    EEG=eeg_checkset(EEG); %***** 
    
    
%     EEG.setname = "imagery_left_eeg_data";
%     EEG.xmin = 0;
%     EEG.xmax = 6.9980;
%     EEG.filename = fileId;
%     EEG.filepath = load_path;
    
%     %EEG.pnts = 3584;
%     %EEG.nbchan = 64;
%     %EEG.trials = 100;
%     EEG.chaninfo.plotrad = [];
%     EEG.reject.gcompreject = [];
%     EEG.comments = char(" ");
%% Load event

     count = 1;
%     
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
 
    % BPF
    disp(['band-pass filtering from '  num2str(lowCut) ' to ' num2str(highCut)  ' Hz']);
    EEG = pop_eegfiltnew(EEG, lowCut, highCut);
    
        %%[spec freq]= spectopo(EEG.data,0,EEG.srate);
        %%figure;plot(freq,spec);
        %size(EEG.data);
        
%         [spec freq]= spectopo(EEG_left.data,EEG_left.pnts,EEG_left.srate);
%         figure;plot(freq,spec);
%         a=spec(:,8:25);
%         size(EEG_left.data)
%         EEG_left = pop_epoch( EEG_left, unique({EEG_left.event(:).type}), [-2  5]);
%         size(EEG_left.data)
%         for iTrial = 1 : 100
%             spec(iTrial,:,:)=spectopo(EEG_left.data(:,:,iTrial),0,EEG_left.srate,'plot','off');
%         end

    %a=spec(:,chIdx,8:25);
    %chIdx = [4 5 6 9 10 11 12 13 14 17 18 19 39 40 41 44 45 46 49 50 51 54 55 56];

    %x_train = a([1:45 51:95],:,:);
    %y_train = [zeros(45,1);ones(45,1)];
    %x_test = a([46:50 96:100],:,:);
    %y_test = [zeros(5,1);ones(5,1)];
    
    
%% ASR
   
    %Z.fieldname = "Z"
%     for i = 1:64
%         EEG.chanlocs(i).labels = char(chan_labels(i))
%         EEG.chanlocs(i).X = load_Data.psenloc(i,1); 
%         EEG.chanlocs(i).Y = load_Data.psenloc(i,2);
%         EEG.chanlocs(i).Z = load_Data.psenloc(i,3);
%         EEG.chanlocs(i).urchan = i;
%     end

%     size(EEG.data)

  % ASR
    EEG.etc.historychanlocs=EEG.chanlocs;
    EEG.etc.historychaninfo=EEG.chaninfo;
    %EEG_left.etc.historychaninfo=EEG_left.chaninfo;
    EEG = clean_rawdata(EEG,5,-1,0.8,4,5,-1); % default setting
    EEG.etc.badchan=find(EEG.etc.clean_channel_mask==0); %Bad chananel information from ASR
    EEG.etc.originalEEG=EEG; % keep origianl EEG before interpolation
    EEG = pop_interp(EEG, EEG.etc.historychanlocs, 'spherical');
   
    %pop_saveset(EEG,'filepath',[dPath subId],'filename',[fileId(1:end-4) '_p.set']); % CAR 비교
    
%% CAR
    EEG = pop_reref( EEG, []);
    
   % pop_saveset(EEG,'filepath',[dPath subId],'filename',[fileId(1:end-4) '_p.set']);
%     
    
%% ICA
    EEG.rank=rank(double(EEG.data));
    EEG = pop_runica(EEG,'extended',1,'pca',EEG.rank);
   % pop_saveset(EEG,'filepath',[dPath subId],'filename',[fileId(1:end-4) '_pi.set']);
    
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
    
   
    end

 %% load .set file
% MainPath = ['C:\Users\oo\Desktop\Artigence\'];
% addpath([MainPath 'eeglab2021.0']);
% eeglab;
% EEG_tuto = pop_loadset('s03_pir.set', 'C:\Users\oo\Desktop\Artigence\data\sub3');    
    