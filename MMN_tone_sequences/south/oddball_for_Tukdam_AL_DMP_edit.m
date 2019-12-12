%%%%%%  Auditory Oddball Roving Paradigm %%%%%%%
%            Marta I. Garrido                  %
%  This function generates an auditory stimuli %
%    with roving standards - change in pitch   %
%%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% adaptation Antoine Lutz Nov.2012 for Tukdam project%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%% Seed the random number generator so tone sequence is the same each
%%%% time we run this. If the tonesequence is unsatisfactory, try a
%%%% different seed.
random_seed=103;
rng(random_seed);

%%%%%% you need to create the library   %%%%%%%%%%%%%%%%%%%%
addpath(genpath('/Volumes/exp/NCCAM/scripts/preprocessing/temp'))

%% constant definitions
srate=44100;    % sampling rate
dur = 0.070;      % duration of each event
dur_click=0.01;%10 milli seconde
dur_click_apriori_in_spl_pts=4; 
freq=500;       % defines frequency of principal
dev = [0 0.10 0.20 0.30 0.40 0.50 0.60]; % creates a vector with deviance from 0 to 60% from freq.
loud=0.2; % loudness
SOA=0.5;
%% build waves for pure tones

freqd = freq*(1+dev(:)); %    calculates new frequency
t=0:1/srate:dur;        % during 'dur' does
tones=zeros(length(dev),length(t)); %cell containing sounds

for j = 1:length(dev)
    tone = sin(2*pi*freqd(j)*t);      % creates all the wave forms to be presented and stores them in a matrix called tones
    amp = loud*tone;
    amp=wind(srate,10,amp');  % makes a ramp of 10 ms to avoid clicking
    tones(j,:) = amp'; % stores the wave forms
end

%%  writes sounds

%for j=1:length(dev)
%    wavwrite(tones(j,:), srate,['tone_freq' num2str(freqd(j))]);
%end


%% loads sounds - writes them in buffer

%for j=1:length(dev)
%    loadsound(['tone_freq' num2str(freqd(j)) '.wav'], j);        % writes sound in the buffer with code 1, 3, 5,...
%end

%% build repetitions
rep=[ones(1,50)*1, ones(1,50)*2, ones(1,75)*3, ones(1,75)*4, ones(1,250)*5, ones(1,250)*6, ones(1,250)*7, ones(1,250)*8, ones(1,250)*9, ones(1,250)*10, ones(1,250)*11]; % creates a vector with # of repetions of standards varying between 5 and 11
rep=rep(randperm(length(rep)));  % makes repetitions random
code = 1:length(dev); % index for percentage deviance from principal frequency
lcode = length(code);

asind = [];
ocode=code(1);

for n = 1:length(rep)  % loops through all repetitions (2000)
    rep1 = rep(n);
    while code(1)==ocode               %
        code = code(randperm(lcode));  % makes the deviance from principal random
    end                                % and loops through all possible deviances (or freqd) %%%%DMP: OK, WTF? that is the stupidest way I've ever seen to pick a random number that is different from the previous random number.
    ocode=code(1);
    for r = 1:rep1       % for the number of repetitions as indicated by rep
        sind = ocode;    % assigns the deviance as indicated by code
        asind = [asind, sind];  % builds a vector (increasing through the loop) with numbers coding for which freqd and how many times it's presented
    end    % if you run this (together with the definitions above) you should get a 14675 vector with numbers from 1 to 7 which repeat themselves as many times as
    % told by rep so between 1 and 11 (sorry I think the comment I made in the past for repetitions between 5 and 11 is actually wrong)
end


%% codes for repetition order
rsind=zeros(1,length(asind));
temp=0;

for n=1:length(asind)
    if asind(n)==temp;
        io=io+1;
        rsind(n)=io;
    else
        rsind(n)=1;
        io=1;
    end
    temp=asind(n);
end



%%% we just need to read and save these files....

%%%% now all we need is to play them for 2000 sounds....

baseline=10;%10sec before sounds;
nb_total_stim=2000;

% make a list of the things that will be added up into the variable
% one time: zeros(1,srate*baseline)
% nb_total_stim times: tones(asind(stim),:)
% nb_total_stim times: zeros(1,srate*(SOA-dur))
tone_length=size(tones,2)
inter_length=srate*(SOA-dur)
total_length=srate*baseline + tone_length*nb_total_stim + inter_length*nb_total_stim

% wave_task=[];
% wave_task=[zeros(1,srate*baseline)];
% wave_trigger=[zeros(1,srate*baseline)];

wave_task=zeros(1,total_length);
wave_trigger=zeros(1,total_length);


% starting position, after the baseline
current_position=srate*baseline + 1;

for stim=1:nb_total_stim
    currtone=tones(asind(stim),:);
    currio = rsind(stim);
    % is it an oddball index? i.e. is this one the first one of a new
    % series?
    oddball = ((currio == 1) && (stim ~= 1));
    %fprintf('%d: tone %d wave_task %d wave_trigger %d\n', [stim, size(currtone,2), size(wave_task,2), size(wave_trigger,2)]);
    fprintf('tone number %04d: %d\n', [stim,asind(stim)]);

    
    %wave_task=[wave_task, tones(asind(stim),:),zeros(1,srate*(SOA-dur))];
    wave_task(current_position:current_position+tone_length-1) = tones(asind(stim),:);
    wave_trigger(current_position:current_position+44) = -1;
    % mark the oddball ones distinctly, with double clicks
    if oddball
        wave_trigger(current_position+220:current_position+264) = -1;
    end
    current_position=current_position+tone_length;
    %wave_task(current_position:current_position+inter_length-1) = zeros(1,srate*(SOA-dur));
    current_position=current_position+inter_length;
    
%     temp=tones(asind(stim),:)*0;
%     %temp(1:dur_click_apriori_in_spl_pts)=1;%% the trigger lasts 10 microseconds (duration of a click)
%     temp(1:round(srate*dur_click))=1;%% the trigger lasts 10 microseconds (duration of a click)           
%     wave_trigger=[wave_trigger, temp,zeros(1,srate*(SOA-dur))];

end

filename=['MMN_roving_with_trigger_dpdb02_seed_', int2str(random_seed), '_', date]

wavwrite( [wave_task', wave_trigger'], srate, filename)

% Save the tone sequence in the variable asind
seqfile=[filename '_tone_sequence.txt']
%save seqfile asind -ASCII
dlmwrite(seqfile,asind);



% if 0 %%% it is important to run simultaneously the audio file and the back mac file ... otherwise, there is no way to know the correct timing....
% name=['MMN_roving_paradigm_500Hz_50Hzstep_800Hz_2000_stim_SOA_.5_with_trigger',date];
% cmd=strrep('wavwrite( [wave_task'',wave_trigger''],  srate,''name'');','name',name);eval(cmd);
% wavwrite([wave_task', wave_trigger'], srate, name);
% 
% name=['MMN_roving_paradigm_500Hz_50Hzstep_800Hz_2000_stim_SOA_.5_seq_stimuli',date];
% cmd=strrep('save name wave_task wave_trigger asind tones dev freq srate;','name',name);eval(cmd);
% name=['MMN_roving_paradigm_500Hz_50Hzstep_800Hz_2000_stim_SOA_.5_mono_audio',date];
% cmd=strrep('wavwrite( wave_task'',  srate,''name'');','name',name);eval(cmd);
% 
% else 
% name=['MMN_roving_paradigm_500Hz_50Hzstep_800Hz_2000_stim_SOA_.5_with_trigger_4pts',date];
% cmd=strrep('wavwrite( [wave_task'',wave_trigger''],  srate,''name'');','name',name);eval(cmd);
% name=['MMN_roving_paradigm_500Hz_50Hzstep_800Hz_2000_stim_SOA_.5_seq_stimuli',date];
% cmd=strrep('save name wave_task wave_trigger asind tones dev freq srate;','name',name);eval(cmd);
% name=['MMN_roving_paradigm_500Hz_50Hzstep_800Hz_2000_stim_SOA_.5_mono_audio',date];
% cmd=strrep('wavwrite( wave_task'',  srate,''name'');','name',name);eval(cmd);
% 
% end
% 
% %%%% this version with 10millisec
% if dur_click>=0.01
% name=['MMN_roving_paradigm_500Hz_50Hzstep_800Hz_2000_stim_SOA_.5_with_trigger_10millisec',date];
% cmd=strrep('wavwrite( [wave_task'',wave_trigger''],  srate,''name'');','name',name);eval(cmd);
% name=['MMN_roving_paradigm_500Hz_50Hzstep_800Hz_2000_stim_SOA_.5_seq_stimuli',date];
% cmd=strrep('save name wave_task wave_trigger asind tones dev freq srate;','name',name);eval(cmd);
% name=['MMN_roving_paradigm_500Hz_50Hzstep_800Hz_2000_stim_SOA_.5_mono_audio',date];
% cmd=strrep('wavwrite( wave_task'',  srate,''name'');','name',name);eval(cmd);
% end
