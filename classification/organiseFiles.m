% Resolve paths relative to the project root
scriptFolder = fileparts(mfilename('fullpath'));
projectRoot = fileparts(scriptFolder);

sourceFolder = fullfile(projectRoot, '17flowers', '17flowers');
destFolder = fullfile(projectRoot, '17flowers_split');

if ~exist(destFolder, 'dir')
    mkdir(destFolder);

    files = dir(fullfile(sourceFolder, '*.jpg'));

    % Sort image filenames
    [~, idx] = sort({files.name});
    files = files(idx);

    disp(['Number of images found: ', num2str(numel(files))]);

    for i = 1:17
        classFolder = fullfile(destFolder, sprintf('class%d', i));
        mkdir(classFolder);

        for j = 1:80
            idx = (i - 1) * 80 + j;

            copyfile( ...
                fullfile(sourceFolder, files(idx).name), ...
                fullfile(classFolder, files(idx).name));
        end
    end

    disp('Dataset reorganized successfully.');
else
    disp('Organized dataset already exists.');
end