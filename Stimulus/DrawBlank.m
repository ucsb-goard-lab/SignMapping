function DrawBlank(win,background,tclose,tstart)
% Written MG 160504, modified form PsychToolbox, DriftDemo4
%
% Display a blank using the new Screen('DrawTexture') command.
%
% Input parameters:
% 'brightness' = brightness of patch (0 = white, 1 = black)
% 'duration' = Time to stop display of blank (in CPU time)
% 'patchSize' = Size of 2D grating patch in pixels.

% Make sure this is running on OpenGL Psychtoolbox:
AssertOpenGL;

% Make sure the GLSL shading language is supported:
AssertGLSL;

% Calculate patch location
[screenXpixels, screenYpixels] = Screen('WindowSize', win);
patchRect = [0 0 screenXpixels screenYpixels];

% Retrieve video redraw interval for later control of our animation timing:
ifi = Screen('GetFlipInterval', win);

% Draw a blank rectabgle with user-defined brightness
Screen('FillRect', win, background*255, patchRect);
    
% Update some grating animation parameters:
vbl = Screen('Flip', win);

while (GetSecs-tstart)<tclose  

    Screen('FillRect', win, background*255, patchRect);

    % Show it at next retrace:
    vbl = Screen('Flip', win, vbl + 0.5 * ifi);
end

return;
