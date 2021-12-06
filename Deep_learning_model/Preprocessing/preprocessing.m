%% SET PATH
MainPath = ['D:\Artigence'];
addpath([MainPath '\eeglab2021.0']);
dPath=['D:\Artigence\data\'];
eeglab;
% Matlab tool인 eeglab을 불러온다

pop_editoptions( 'option_savetwofiles', 1,'option_single', 0);
% Edit eeglab options stored in the eeg_options Matlab file

cd(dPath);
subStruct=dir;
subStruct = subStruct(cellfun(@any,strfind({subStruct.name},'sub')));
nSub = length(subStruct);

lowCut=1;
highCut=50;
% Bandpassfilter를 하기 위해 lowcut frequency와 highcut frequency를 알려준다
%% MAIN LOOP

 for iSub = 1 : nSub
        
    subId = subStruct(iSub).name;
    fileStruct = dir([subId  '/*.mat']);
    fileId = fileStruct(1).name;
    load_path = strcat(dPath,subId,"/",fileId);
    
    disp(['Sub ' num2str(iSub) ' Loading......... ' fileId]);
    load_Data = load(load_path);
    load_Data = struct(load_Data.eeg);
    % 우리가 요청해서 얻은 데이터들을 불러온다
%% checkset
   
    EEG.data = [load_Data.imagery_left(1:64,:) load_Data.imagery_right(1:64,:)];
    EEG.srate = load_Data.srate;
    EEG.chanlocs = readlocs('biosemi64.ced');
    EEG=eeg_checkset(EEG); %***** 
    % 이후 EEGnet 모델의 학습에 필요한 정보인 srate나 chanlocs 등을 불러온다  
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
    % 왼쪽 이벤트를 불러오는 것이다
    for i = 1:358400 
        if load_Data.imagery_event(i) == 1
            EEG.event(count).type = 'right';
            EEG.event(count).latency = 358401 + i;
            EEG.event(count).epoch = count;
            count = count+ 1;
        end
    end
   % 오른쪽 이벤트를 불러오는 것이다
%% BPF
    
    disp(['band-pass filtering from '  num2str(lowCut) ' to ' num2str(highCut)  ' Hz']);
    EEG = pop_eegfiltnew(EEG, lowCut, highCut);
   % 위에서 주어진 lowCut Frequency와 highCut Frequency를 이용해서 band-pass Filtering을 한다.  
    
%% ASR
   
    EEG.etc.historychanlocs=EEG.chanlocs;
    EEG.etc.historychaninfo=EEG.chaninfo;
    % 받아온 정보들을 이제 ASR 전처리를 실행하기 위해 준비한다
    EEG = clean_rawdata(EEG,5,-1,0.8,4,5,-1); % default setting
    EEG.etc.badchan=find(EEG.etc.clean_channel_mask==0); %Bad chananel information from ASR
    
    EEG.etc.originalEEG=EEG; % keep origianl EEG before interpolation
    EEG = pop_interp(EEG, EEG.etc.historychanlocs, 'spherical');
    % 뒤에서 전처리를 계속 해야 해 interpolation을 통해 채널 개수를 64개로 맞춰준다
    % ASR의 목적은 뇌파의 신호가 들어올때 원하는 Channel 정보뿐 아니라
    % noise와 불필요한 Badchannel도 들어오기 때문에 그것을 찾아 제거와 축소를 위한 것이 목적이다
%% CAR

    EEG = pop_reref( EEG, []);
    % Cephalic electrode reference(머리 중심으로부터 계산 하는 것)을
    % Common average reference(전체 합의 평균으로 계산 하는 것)으로 바꿔주는 전처리이다.
%% ICA

    EEG.rank=rank(double(EEG.data));
    % EEG data의 rank를 저장한다
    EEG = pop_runica(EEG,'extended',1,'pca',EEG.rank);
    % run ica on dataset using pop_runica function   
%% IC Label
 
    EEG = pop_iclabel(EEG, 'default'); % EEG data의 iclabel을 불러온다
  
    rejIdx=[]; % rejIdx matrix 형태로 초기화
    cutProb=0.5; % 50 percent
    for iICA = 1 : EEG.rank
        [maxProb maxIdx]= max(EEG.etc.ic_classification.ICLabel.classifications(iICA, :));
        % 1: brain / 2: Muscle / 3: Eye / 4: Heart / 5: Line Noise / 6: Channel Noise / 7: Other
        if maxIdx ~= 1 && maxIdx ~= 7 && maxProb > cutProb
            rejIdx = [rejIdx iICA];
        % 2번부터 6번 영역이고 max 확률이 cut 확률보다 높을시 reject 해준다
        end
    end
    
    EEG.etc.rejIdx = rejIdx;
    EEG = pop_subcomp( EEG, rejIdx, 0);
    % rej영역의 정보를 얻었기 때문에 불필요한 채널을 없애준다  
    pop_saveset(EEG,'filepath',[dPath subId],'filename',[fileId(1:end-4) '_pir.set']);
    % 이후 전처리가 완료된 데이터를 저장한다.
    EEG = eeg_emptyset
end