function [  ] = retinotopicMappingStimulus(  )
% Beta retinotopic mapping code
%   Written KS 180123

clear
close all

newTiming_flag = 1;

%% Preparing movie database, change this path to point to the directory containing '4direction_stim.mat'
movFiles = dir(fullfile('C:\Users\Goard Lab\Dropbox\StimulusPresentation\Widefield\RetinotopicMappingStimulusNEW\stim_movies'));

% Load the movie
load([movFiles(3).folder '\' movFiles(3).name]);

% Concatenating into one big movie
full_stim = uint8(cat(4,forward_stim,backward_stim,upward_stim,downward_stim));

%prepare screen function for psychtoolbox
screenid = 1;
mov_length = size(forward_stim,3); % each presentation of the stimulus will be this length


%% Stimulus parameters (change these to your liking)
repeats = 1; %repeats of total movie list(default = 30)
offTime = 2; %gray screen in between each direction
refreshRate = Screen('NominalFrameRate', screenid);
DAQ_flag = 0; % For triggering camera, you should change this to match your system

background = 0.5;
waitframes = 2; %reduce frame rate to 30Hz (wait 2 frames in between each retrace)
onTime = round(mov_length/(refreshRate/2)); %movie length (default = 30)
% Customization

%Length of stimulus set
stim_dur = onTime + offTime;
repDur = stim_dur*4; % four directions
totalDur = repDur*repeats;
disp(['Stimulus duration: ' num2str(totalDur) ' sec'])

%% Psych toolbox set up
Screen('Preference','SkipSyncTests',1);

Screen('Preference','VisualDebugLevel',0);
Screen('Preference','SuppressAllWarnings',1);

% patch size & location
patchArea = [0 0 1 1];
[width,height] = Screen('WindowSize',screenid);
patchArea([1 3]) = patchArea([1 3])*width;  % scale area of patch (width)
patchArea([2 4]) = patchArea([2 4])*height;  % scale area of patch (height)

%  window
win = Screen('OpenWindow', screenid, background*255);
ifi = Screen('GetFlipInterval', win);
vbl = Screen('Flip', win);
priorityLevel = MaxPriority(win);
Priority(priorityLevel);

%% Triggering the camera, change this as necessary
if DAQ_flag
    s = daq.createSession('ni');
    addAnalogOutputChannel(s, 'Dev1', 'ao0', 'Voltage');
    outputSingleScan(s, 0);
    % Trigger
    %% new timing
    if newTiming_flag
        %generating a square wave for trigger
        oneCycle = cat(2,repmat(5,[1 75]), repmat(0,[1 25])); %45samples (ms) on, 5 samples(ms) off
        totalDur_samples = totalDur*s.Rate;
        numCycles = ceil(totalDur_samples/length(oneCycle));
        outputMat = repmat(oneCycle,[1 numCycles+20]);
        
        queueOutputData(s,outputMat');
        s.startBackground();
    else
        % Timing
        outputSingleScan(s,5);
    end
end

tstart = GetSecs;

%% Display stimulus
% Displays the pre-recorded movies, one in each direction
for rep = 1:repeats
    for drxn = 1:4
        % Draw a blank screen
        blank_on(rep,drxn) = GetSecs-tstart
        tclose = (rep-1)*repDur+(drxn-1)*stim_dur+offTime;
        DrawBlank(win,background,tclose,tstart)
        blank_end(rep,drxn) = GetSecs-tstart

        %display movies
        mov_on(rep,drxn) = GetSecs-tstart
        tclose = (rep-1)*repDur+(drxn)*stim_dur;
            frameIdx = 1; %default = 100
            while (GetSecs-tstart)<tclose % Play sign mapping movie
               % disp(num2str(frameIdx))
                [movie] = Screen('MakeTexture',win,full_stim(:,:,frameIdx,drxn));
                Screen('DrawTexture',win,movie,[],[0 0 width height]);
                %[rate] = Screen('FrameRate',win, [2], [30]);
                %Screen('Flip',win, (ifi*.2)); 
               % vbl = Screen('Flip',win,vbl+(waitframes-.5)*ifi);
               vbl = Screen('Flip',win,vbl+(waitframes-0.5)*ifi); % correct timing to 30hz
                frameIdx = min(frameIdx+1,mov_length);
            end
            
        mov_end(rep,drxn) = GetSecs-tstart        
    end
end
tfinal = (GetSecs-tstart)

%% Clean up
if DAQ_flag == 1
    if newTiming_flag
        stop(s);
    end
    outputSingleScan(s, 0);
end
Screen('CloseAll')
Priority(0);

%% Save stimulus data file

Stimdata.blank_on = blank_on;
Stimdata.blank_end = blank_end;
Stimdata.mov_on = mov_on;
Stimdata.mov_end = mov_end;
Stimdata.off_time = offTime;
Stimdata.repeats = repeats;
Stimdata.on_time = onTime;

uisave('Stimdata')
