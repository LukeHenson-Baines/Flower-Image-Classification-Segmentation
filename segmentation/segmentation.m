% Resolve paths relative to the project root
scriptFolder = fileparts(mfilename('fullpath'));
projectRoot = fileparts(scriptFolder);

imageFolder = fullfile(projectRoot, ...
    'daffodilSeg', 'daffodilSeg', 'ImagesRsz256');

labelFolder = fullfile(projectRoot, ...
    'daffodilSeg', 'daffodilSeg', 'LabelsRsz256');

modelsFolder = fullfile(projectRoot, 'models');

if ~exist(modelsFolder, 'dir')
    mkdir(modelsFolder);
end

% Step 1: Load Data
imds = imageDatastore(imageFolder);

classNames = ["background", "flower"];
labelIDs = [0, 1];

pxds = pixelLabelDatastore(labelFolder, classNames, labelIDs);

% Step 2: Split the data into training and validation sets
[imdsTrain, imdsVal, pxdsTrain, pxdsVal] = partitionSemanticSegmentationData(imds, pxds, 0.7);

% Combine image and pixel label datastores for training and validation
trainingData = combine(imdsTrain, pxdsTrain); 
validationData = combine(imdsVal, pxdsVal);   

% Step 3: Define the U-Net layers
inputSize = [256, 256, 3]; 
numClasses = 2; % 2 classes: background and flower
lgraph = unetLayers(inputSize, numClasses);

% Step 4: Set up the training options
options = trainingOptions('adam', ...
    'InitialLearnRate', 1e-4, ...
    'MaxEpochs', 15, ...
    'MiniBatchSize', 4, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', validationData, ...
    'ValidationFrequency', 30, ...
    'Verbose', true, ...
    'Plots', 'training-progress');

% Step 5: Train the network
segnet = trainNetwork(trainingData, lgraph, options);

% Step 6: Save the trained network
save(fullfile(modelsFolder, 'segnet.mat'), 'segnet');


function [imdsTrain, imdsVal, pxdsTrain, pxdsVal] = partitionSemanticSegmentationData(imds, pxds, trainRatio)    
    % Randomly shuffle image indices
    numImages = numel(imds.Files);
    idx = randperm(numImages);

    % Calculate training split
    numTrain = floor(numImages * trainRatio);

    % Create training and validation image datastores
    imdsTrain = subset(imds, idx(1:numTrain));
    imdsVal = subset(imds, idx(numTrain + 1:end));

    % Create corresponding pixel-label datastores
    pxdsTrain = subset(pxds, idx(1:numTrain));
    pxdsVal = subset(pxds, idx(numTrain + 1:end));
end