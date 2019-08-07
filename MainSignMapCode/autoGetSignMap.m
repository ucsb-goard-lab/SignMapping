function autoGetSignMap()

fprintf(['\nThis is highly not recommended, you''ll have NO control over the outcome.\n\n' ...
    'I would highly recommend instead using signMapping_master, which is about\n'...
    'as easy to use, but lets you have more control.\n\n'...
    'If you understand, and wish to continue, press any key...\n'])
pause

% Adding necessary paths to access the code for sign mapping
curr_script_dir = mfilename('fullpath');
parent_script_dir = curr_script_dir(1:end-length(mfilename));
addpath(fullfile(parent_script_dir,'Juavinett et al 2017 Code'));

%% Usage
sm = SignMapper(); % Create the sign mapping object
sm.autoRunMapping(); % Run a default, autorun, no user control
end