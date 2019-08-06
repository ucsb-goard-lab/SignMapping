function autoGetSignMap()

% Adding necessary paths to access the code for sign mapping
curr_script_dir = mfilename('fullpath');
parent_script_dir = curr_script_dir(1:end-17);
addpath(fullfile(parent_script_dir,'Juavinett et al 2017 Code'));

%% Usage
sm = SignMapper(); % Create the sign mapping object
sm.autoRunMapping(); % Run a default, autorun, no user control

end