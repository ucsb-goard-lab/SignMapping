function varargout = frequencySearcherGUI(varargin)

%% GUI Frequency Searcher
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Written 12Apr2017 KS
% Last Updated: 
% 30Aug2017KS - Added header information

% Presents a GUI for manual choosing of proper phase maps

%%% Necessary Subfunctions %%%
% None

%%% Inputs %%%
% phaseSearch                   Output from subfcn_frequencySearch, cell array of phase maps
% target                        Which frequency bins to examine (usually 1:10)
% target_dat                    Which phase data (azimuth or altitude)

%%% Outputs %%%
% k                             Chosen frequency bin of interest

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% FREQUENCYSEARCHERGUI MATLAB code for frequencySearcherGUI.fig
%      FREQUENCYSEARCHERGUI, by itself, creates a new FREQUENCYSEARCHERGUI or raises the existing
%      singleton*.
%
%      H = FREQUENCYSEARCHERGUI returns the handle to a new FREQUENCYSEARCHERGUI or the handle to
%      the existing singleton*.
%
%      FREQUENCYSEARCHERGUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in FREQUENCYSEARCHERGUI.M with the given input arguments.
%
%      FREQUENCYSEARCHERGUI('Property','Value',...) creates a new FREQUENCYSEARCHERGUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before frequencySearcherGUI_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to frequencySearcherGUI_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help frequencySearcherGUI

% Last Modified by GUIDE v2.5 01-May-2017 08:39:01

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
    'gui_Singleton',  gui_Singleton, ...
    'gui_OpeningFcn', @frequencySearcherGUI_OpeningFcn, ...
    'gui_OutputFcn',  @frequencySearcherGUI_OutputFcn, ...
    'gui_LayoutFcn',  [] , ...
    'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before frequencySearcherGUI is made visible.
function frequencySearcherGUI_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to frequencySearcherGUI (see VARARGIN)


% Plotting data
handles.p1 = varargin{1};
handles.t = varargin{2};
handles.t_d = varargin{3};
handles.a = 1;
handles.fs = varargin{4};
imagesc(imgaussfilt(handles.p1{handles.a},2))
colormap jet
title(['Peak #: ' num2str(handles.a) ...
    ' | k-value: ' num2str(handles.t(handles.a)) ...
    ' | Frequency: ' num2str(handles.fs*(handles.t(handles.a)/2)/size(handles.t_d,3)) 'Hz'],...
    'FontSize', 20)
set(gca,'XTickLabel','')
set(gca,'YTickLabel', '')
hcb=colorbar;
set(hcb,'YTick',[])


% Choose default command line output for frequencySearcherGUI
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);


% UIWAIT makes frequencySearcherGUI wait for user response (see UIRESUME)
 uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.

function varargout = frequencySearcherGUI_OutputFcn(hObject, eventdata, handles)
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;
close all;



% --- Executes on button press in Previous.
function Previous_Callback(hObject, eventdata, handles)
% hObject    handle to Previous (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
handles.a = handles.a - 1;
imagesc(imgaussfilt(handles.p1{handles.a},2))
colormap jet
title(['Peak #: ' num2str(handles.a) ...
    ' | k-value: ' num2str(handles.t(handles.a)) ...
    ' | Frequency: ' num2str(handles.fs*(handles.t(handles.a)/2)/size(handles.t_d,3)) 'Hz'],...
    'FontSize', 20)
set(gca,'XTickLabel','')
set(gca,'YTickLabel', '')
hcb=colorbar;
set(hcb,'YTick',[])

guidata(hObject, handles);



% --- Executes on button press in Next.
function Next_Callback(hObject, eventdata, handles)
% hObject    handle to Next (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
handles.a = handles.a + 1;
imagesc(imgaussfilt(handles.p1{handles.a},2))
colormap jet
title(['Peak #: ' num2str(handles.a) ...
    ' | k-value: ' num2str(handles.t(handles.a)) ...
    ' | Frequency: ' num2str(handles.fs*(handles.t(handles.a)/2)/size(handles.t_d,3)) 'Hz'],...
    'FontSize', 20)
set(gca,'XTickLabel','')
set(gca,'YTickLabel', '')
hcb=colorbar;
set(hcb,'YTick',[])

guidata(hObject, handles);



% --- Executes on button press in Choose.
function Choose_Callback(hObject, eventdata, handles)
% hObject    handle to Choose (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

handles.output = handles.t(handles.a);
waitfor(msgbox(['Selected frequency bin: ' num2str(handles.t(handles.a))]))
guidata(hObject, handles);
close all;



function foi_Callback(hObject, eventdata, handles)
% hObject    handle to foi (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of foi as text
%        str2double(get(hObject,'String')) returns contents of foi as a double


% --- Executes during object creation, after setting all properties.
function foi_CreateFcn(hObject, eventdata, handles)
% hObject    handle to foi (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in Go.
function Go_Callback(hObject, eventdata, handles)
% hObject    handle to Go (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

foi = str2num(char(get(handles.foi,'String')));
k_val = round(foi*size(handles.t_d,3));
tmp = abs(handles.t - k_val);
[~, handles.a] = min(tmp);
imagesc(imgaussfilt(handles.p1{handles.a},1))
colormap jet
title(['Peak #: ' num2str(handles.a) ...
    ' | k-value: ' num2str(handles.t(handles.a)) ...
    ' | Frequency: ' num2str(handles.fs*(handles.t(handles.a)/2)/size(handles.t_d,3)) 'Hz'],...
    'FontSize', 20)
set(gca,'XTickLabel','')
set(gca,'YTickLabel', '')
hcb=colorbar;
set(hcb,'YTick',[])

guidata(hObject, handles);


% --- Executes when user attempts to close figure1.
function figure1_CloseRequestFcn(hObject, eventdata, handles)
% hObject    handle to figure1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% Hint: delete(hObject) closes the figure
if isequal(get(hObject, 'waitstatus'), 'waiting')
% The GUI is still in UIWAIT, us UIRESUME
uiresume(hObject);
else
% The GUI is no longer waiting, just close it
delete(hObject);
end
