% Load the .mat file
load('./DTU/S1.mat'); % Replace 'yourfile.mat' with the actual file name

% Convert the expinfo table to a struct
expinfoStruct = table2struct(expinfo);

% Optionally, save the new struct to a .mat file
save('./DTU/S1.mat');

fprintf('done');

% fileDir = 'DTU/';
% filePattern = fullfile(fileDir, '*.mat');
% matFiles = dir(filePattern);

% for k = 1:length(matFiles)
%     fprintf('Processing file %d of %d: %s\n', k, length(matFiles), matFiles(k).name);

%     baseFilename = matFiles(k).name;
%     fullFileName = fullfile(fileDir, baseFilename);
%     load(fullFileName);

%     expinfoStruct = table2struct(expinfo)
%     save()