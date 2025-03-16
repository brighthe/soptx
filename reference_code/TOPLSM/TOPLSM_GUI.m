function varargout = TOPLSM_GUI(varargin)
%================================================%
%  TOPLSM_GUI is a graphic user interface for TOPLSM
% Developed by: Michael Yu WANG and Shikui CHEN 
% Department of Aotomation and Computer-Aided Engineering, 
% The Chinese University of Hong Kong
% Email: yuwang@acae.cuhk.edu.hk
%================================================%
%      TOPLSM_GUI, by itself, creates a new TOPLSM_GUI or raises the existing
%      singleton*.
%
%      H = TOPLSM_GUI returns the handle to a new TOPLSM_GUI or the handle to
%      the existing singleton*.
%
%      TOPLSM_GUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in TOPLSM_GUI.M with the given input arguments.
%
%      TOPLSM_GUI('Property','Value',...) creates a new TOPLSM_GUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before TopLSM_GUI_OpeningFunction gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to TopLSM_GUI_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help TopLSM_GUI

% Last Modified by GUIDE v2.5 29-Nov-2005 10:13:28

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @TopLSM_GUI_OpeningFcn, ...
                   'gui_OutputFcn',  @TopLSM_GUI_OutputFcn, ...
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
clc;
clear;


% --- Executes just before TopLSM_GUI is made visible.
function TopLSM_GUI_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to TopLSM_GUI (see VARARGIN)

% Choose default command line output for TopLSM_GUI
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes TopLSM_GUI wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = TopLSM_GUI_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;



function StrDW_Callback(hObject, eventdata, handles)
% hObject    handle to StrDW (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of StrDW as text
%        str2double(get(hObject,'String')) returns contents of StrDW as a double


% --- Executes during object creation, after setting all properties.
function StrDW_CreateFcn(hObject, eventdata, handles)
% hObject    handle to StrDW (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function StrDH_Callback(hObject, eventdata, handles)
% hObject    handle to StrDH (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of StrDH as text
%        str2double(get(hObject,'String')) returns contents of StrDH as a double


% --- Executes during object creation, after setting all properties.
function StrDH_CreateFcn(hObject, eventdata, handles)
% hObject    handle to StrDH (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --------------------------------------------------------------------
function Untitled_1_Callback(hObject, eventdata, handles)
% hObject    handle to Untitled_1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function Untitled_2_Callback(hObject, eventdata, handles)
% hObject    handle to Untitled_2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)



function StrEleNx_Callback(hObject, eventdata, handles)
% hObject    handle to StrEleNx (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of StrEleNx as text
%        str2double(get(hObject,'String')) returns contents of StrEleNx as a double


% --- Executes during object creation, after setting all properties.
function StrEleNx_CreateFcn(hObject, eventdata, handles)
% hObject    handle to StrEleNx (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function StrEleNy_Callback(hObject, eventdata, handles)
% hObject    handle to StrEleNy (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of StrEleNy as text
%        str2double(get(hObject,'String')) returns contents of StrEleNy as a double


% --- Executes during object creation, after setting all properties.
function StrEleNy_CreateFcn(hObject, eventdata, handles)
% hObject    handle to StrEleNy (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in radiobutton1.
function radiobutton1_Callback(hObject, eventdata, handles)
% hObject    handle to radiobutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
h = figure;
text(h, 'This program is developed by Michael Yu WANG and Shikui CHEN.');

% Hint: get(hObject,'Value') returns toggle state of radiobutton1


% --- Executes on button press in radiobutton2.
function radiobutton2_Callback(hObject, eventdata, handles)
% hObject    handle to radiobutton2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radiobutton2


% --------------------------------------------------------------------
function AboutUs_Callback(hObject, eventdata, handles)
% hObject    handle to AboutUs (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)



function StrL4V_Callback(hObject, eventdata, handles)
% hObject    handle to StrL4V (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of StrL4V as text
%        str2double(get(hObject,'String')) returns contents of StrL4V as a double


% --- Executes during object creation, after setting all properties.
function StrL4V_CreateFcn(hObject, eventdata, handles)
% hObject    handle to StrL4V (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function StrL4C_Callback(hObject, eventdata, handles)
% hObject    handle to StrL4C (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of StrL4C as text
%        str2double(get(hObject,'String')) returns contents of StrL4C as a double


% --- Executes during object creation, after setting all properties.
function StrL4C_CreateFcn(hObject, eventdata, handles)
% hObject    handle to StrL4C (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pushbutton_Reset.
function pushbutton_Reset_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_Reset (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in pushbutton_Start.
function pushbutton_Start_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_Start (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
[DomainWidth, DomainHight, EleNumPerRow, EleNumPerCol, LM4Vol,...
    LM4Curv,FEAInterval, PlotInterval, TotalItNum] = GetInputData(hObject, eventdata, handles);
LSgrid = TOPLSM(DomainWidth, DomainHight, EleNumPerRow, EleNumPerCol, LM4Vol, LM4Curv,FEAInterval,...
    PlotInterval, TotalItNum,hObject, eventdata, handles);


% --- Executes on button press in pushbutton_Apply.
function pushbutton_Apply_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_Apply (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
set(handles.pushbutton_Start, 'Enable', 'on');

[DomainWidth, DomainHight, EleNumPerRow, EleNumPerCol, LM4Vol,...
    LM4Curv,FEAInterval, PlotInterval, TotalItNum] = GetInputData(hObject, eventdata, handles);

EW = DomainWidth / EleNumPerRow;
EH = DomainHight / EleNumPerCol;
[ x, y ] = meshgrid( [0 : EW : DomainWidth], [0 : EH : DomainHight] );
cx = DomainWidth / 200 * [ 33.33  100  166.67  0   66.67  133.33  200  33.33  100  166.67  0   66.67  133.33  200  33.33  100  166.67];
cy = DomainHight / 100 * [   0     0     0     25   25      25     25    50    50    50    75    75     75     75    100   100   100]; 
 
for i = 1 : length( cx )
       tmpPhi( :, i ) = - sqrt ( (  x(:) - cx ( i ) ) .^2 + (  y(:)  - cy ( i ) ) .^2 ) + DomainHight/10;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
end;
Phi = - (max(tmpPhi.')).';
Phi = reshape(Phi, size(x));

if get(handles.StrContour_RButn,'Value') == 1
tmpy1 =[0 : 2*EH : DomainHight];
tmpx1 = zeros(size(tmpy1));
tmpx2 = tmpx1 - EW;
tmpy2 = tmpy1- EH;

for i = 1: length(tmpx1)
plot([tmpx1(i)  tmpx2(i)], [tmpy1(i)  tmpy2(i)],'-r','LineWidth',3);
hold on
end;
title('Initial Condition');  
axis equal;
plot([0 DomainWidth DomainWidth 0 0], [0 0 DomainHight  DomainHight 0],'-g','LineWidth',2.5 );
hold on
quiver(DomainWidth+0.5*EW, 0.5*DomainHight ,0 , -5*EH, '-r','LineWidth',3);
hold on
text(DomainWidth+EW, 0.5*DomainHight, 'F', 'HorizontalAlignment','left', 'FontSize',16);
contourf(x,y,Phi, [0 0])
k = size(x)-1;
for i = 1: k(1)
    plot([0 , DomainWidth ] , [i *EH, i * EH]);
end;

for i = 1: k(2)
    plot([i*EW , i*EW ] , [0, DomainHight]);
end;

grid on
hold off
else
    contourf( x, y, Phi, [0  0]);  alpha(0.05); 
    hold on;
    title('Initial Level Set function');  
    h = surface(x, y, -Phi);  view([37.5  30]);  axis equal;  grid on;
    set(h,'FaceLighting','phong','FaceColor','interp', 'AmbientStrength',0.6); 
    hold off
end;
    

% --------------------------------------------------------------------
function Menu_Reset_Callback(hObject, eventdata, handles)
% hObject    handle to Menu_Reset (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
set(handles.StrDW, 'String', '2.0');
set(handles.StrDH, 'String', '1.0');
set(handles.StrEleNx, 'String', '80');
set(handles.StrEleNy, 'String', '40');
set(handles.StrL4V, 'String', '100');
set(handles.StrL4C, 'String', '1');
set(handles.StrItNum, 'String', '100');
set(handles.StrFEIntv, 'String', '1');
set(handles.StrPlotIntv, 'Value', 1);


% --------------------------------------------------------------------
function Menu_Exit_Callback(hObject, eventdata, handles)
% hObject    handle to Menu_Exit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
exit

% --------------------------------------------------------------------
function Untitled_6_Callback(hObject, eventdata, handles)
% hObject    handle to Untitled_6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)




% --- Executes on selection change in popupmenu1.
function popupmenu1_Callback(hObject, eventdata, handles)
% hObject    handle to popupmenu1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = get(hObject,'String') returns popupmenu1 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from popupmenu1


% --- Executes during object creation, after setting all properties.
function popupmenu1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to popupmenu1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in StrPlotIntv.
function StrPlotIntv_Callback(hObject, eventdata, handles)
% hObject    handle to StrPlotIntv (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = get(hObject,'String') returns StrPlotIntv contents as cell array
%        contents{get(hObject,'Value')} returns selected item from StrPlotIntv


% --- Executes during object creation, after setting all properties.
function StrPlotIntv_CreateFcn(hObject, eventdata, handles)
% hObject    handle to StrPlotIntv (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function StrFEIntv_Callback(hObject, eventdata, handles)
% hObject    handle to StrFEIntv (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of StrFEIntv as text
%        str2double(get(hObject,'String')) returns contents of StrFEIntv as a double


% --- Executes during object creation, after setting all properties.
function StrFEIntv_CreateFcn(hObject, eventdata, handles)
% hObject    handle to StrFEIntv (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function StrItNum_Callback(hObject, eventdata, handles)
% hObject    handle to StrItNum (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of StrItNum as text
%        str2double(get(hObject,'String')) returns contents of StrItNum as a double


% --- Executes during object creation, after setting all properties.
function StrItNum_CreateFcn(hObject, eventdata, handles)
% hObject    handle to StrItNum (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function  [DomainWidth, DomainHight, EleNumPerRow, EleNumPerCol, LM4Vol,...
    LM4Curv,FEAInterval, PlotInterval, TotalItNum] = GetInputData(hObject, eventdata, handles)
DomainWidth = str2double(get(handles.StrDW,'String'));
DomainHight = str2double(get(handles.StrDH,'String'));
EleNumPerRow = str2double(get(handles.StrEleNx,'String'));
EleNumPerCol = str2double(get(handles.StrEleNy,'String'));
LM4Vol = str2double(get(handles.StrL4V,'String'));
LM4Curv = str2double(get(handles.StrL4C,'String'));
FEAInterval = str2double(get(handles.StrFEIntv,'String'));
PlotInterval = get(handles.StrPlotIntv,'Value');
TotalItNum = str2double(get(handles.StrItNum,'String'));


% --- Executes on button press in StrContour_RButn.
function StrContour_RButn_Callback(hObject, eventdata, handles)
% hObject    handle to StrContour_RButn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of StrContour_RButn
if get(hObject,'Value') == 1
set(handles.StrLS_RButn, 'Value',0);
set(handles.pushbutton_Start, 'Enable','off');
end;

% --- Executes on button press in StrLS_RButn.
function StrLS_RButn_Callback(hObject, eventdata, handles)
% hObject    handle to StrLS_RButn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of StrLS_RButn
if get(hObject,'Value') == 1
set(handles.StrContour_RButn, 'Value',0);
set(handles.pushbutton_Start, 'Enable','off');
end;



% --------------------------------------------------------------------
function Menu_Clear_Callback(hObject, eventdata, handles)
% hObject    handle to Menu_Clear (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
set(handles.StrDW, 'String', '0');
set(handles.StrDH, 'String', '0');
set(handles.StrEleNx, 'String', '0');
set(handles.StrEleNy, 'String', '0');
set(handles.StrL4V, 'String', '0');
set(handles.StrL4C, 'String', '0');
set(handles.StrItNum, 'String', '0');
set(handles.StrFEIntv, 'String', '0');
set(handles.StrPlotIntv, 'Value', 0);



% --------------------------------------------------------------------
function Menu_Help_Callback(hObject, eventdata, handles)
% hObject    handle to Menu_Help (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function Menu_About_Callback(hObject, eventdata, handles)
% hObject    handle to Menu_About (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

h1 = figure;
set(h1,'Name', 'About TopLSM GUI')

text('Hello!')
