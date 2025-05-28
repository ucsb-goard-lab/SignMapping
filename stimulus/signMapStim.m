function [  ] = retinotopicMappingStimulus_newscope(  )
% Beta retinotopic mapping code
%   Written KS 180123

clear
close all

%% Preparing movie database, change this path to point to the directory containing '4direction_stim.mat'

% Load the movie
load('4directions_stim.mat');

% Concatenating into one big movie
full_stim = uint8(cat(4,forward_stim,backward_stim,upward_stim,downward_stim));

%prepare screen function for psychtoolbox
screenid = 1;
mov_length = size(forward_stim,3); % each presentation of the stimulus will be this length
    

%% Stimulus parameters (change these to your liking)
repeats = 20; %repeats of total movie list(default = 30)
offTime = 2; %gray screen in between each direction
refreshRate = Screen('NominalFrameRate', screenid);
DAQ_flag = 1; % For triggering camera, you should change this to match your system

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

% patch size & location/
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
    
    wf = WidefieldTriggerer(DAQ_flag);
    wf.initialize(totalDur);
    wf.start();

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
               vbl = Screen('Flip',win,vbl+(waitframes-0.5)*ifi); % correct timing to 30hz
                frameIdx = min(frameIdx+1,mov_length);
            end
            
        mov_end(rep,drxn) = GetSecs-tstart        
    end
end
tfinal = (GetSecs-tstart)

%% Clean up
wf.finish();
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
