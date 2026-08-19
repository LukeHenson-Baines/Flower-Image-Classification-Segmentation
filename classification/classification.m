% Resolve paths relative to the project root
scriptFolder = fileparts(mfilename('fullpath'));
projectRoot = fileparts(scriptFolder);

datasetFolder = fullfile(projectRoot, '17flowers_split');
modelsFolder = fullfile(projectRoot, 'models');
resultsFolder = fullfile(projectRoot, 'results');

% Create output directories if required
if ~exist(modelsFolder, 'dir')
    mkdir(modelsFolder);
end

if ~exist(resultsFolder, 'dir')
    mkdir(resultsFolder);
end

% Step 1: Load and Prepare the Image Data
imds = imageDatastore(datasetFolder, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% Split the dataset into training and validation (70% training, 30% validation)
[imdsTrain, imdsVal] = splitEachLabel(imds, 0.7, 'randomized');

% Step 2: Data Preprocessing and Training Augmentation
inputSize = [256 256 3];

% Data Augmentation: Random transformations to improve generalisation
imageAugmenter = imageDataAugmenter( ...
    'RandRotation', [-30, 30], ...  % Rotation range
    'RandScale', [0.8, 1.2], ...    % Scaling range
    'RandXTranslation', [-10, 10], ... % Translation range for X
    'RandYTranslation', [-10, 10], ... % Translation range for Y
    'RandXReflection', true);      % Random horizontal flip

% Create augmented image datastores for training and validation
augimdsTrain = augmentedImageDatastore(inputSize, imdsTrain, 'DataAugmentation', imageAugmenter);
augimdsVal = augmentedImageDatastore(inputSize, imdsVal);


% Step 3: Define the CNN Architecture
layers = [
    imageInputLayer(inputSize, 'Name', 'input')

    convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv_1')
    batchNormalizationLayer('Name', 'batchnorm_1')
    reluLayer('Name', 'relu_1')

    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool_1')

    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv_2')
    batchNormalizationLayer('Name', 'batchnorm_2')
    reluLayer('Name', 'relu_2')

    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool_2')

    convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'conv_3')
    batchNormalizationLayer('Name', 'batchnorm_3')
    reluLayer('Name', 'relu_3')

    fullyConnectedLayer(256, 'Name', 'fc_1')
    reluLayer('Name', 'relu_4')
    dropoutLayer(0.5, 'Name', 'dropout')

    fullyConnectedLayer(17, 'Name', 'fc_2')  % 17 classes for flower categories
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')
];

% Step 4: Set Training Options
options = trainingOptions('adam', ...
    'MaxEpochs', 30, ...
    'InitialLearnRate', 1e-4, ...  
    'ValidationData', augimdsVal, ...
    'ValidationFrequency', 30, ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...  
    'ExecutionEnvironment', 'gpu');

% Step 5: Train the Network
classnet = trainNetwork(augimdsTrain, layers, options);

% Step 6: Save the Trained Model
save(fullfile(modelsFolder, 'classnet.mat'), 'classnet');

% Step 7: Evaluate the Model
predictedLabels = classify(classnet, augimdsVal);
trueLabels = imdsVal.Labels;

% Calculate the accuracy
accuracy = sum(predictedLabels == trueLabels) / numel(trueLabels);
disp(['Validation accuracy: ', num2str(accuracy)]);

% Step 8: Plot and Save Confusion Matrix
figure;
confusionchart(trueLabels, predictedLabels);
title('Confusion Matrix');
exportgraphics(gcf, fullfile(resultsFolder, 'confusion_matrix.png'));

% Step 9: Plot and Save ROC Curves
[YPredScores] = predict(classnet, augimdsVal);
classNames = categories(trueLabels);

figure;
hold on;
for i = 1:numel(classNames)
    trueBinary = (trueLabels == classNames{i});
    scores = YPredScores(:, i);
    [X, Y, ~, AUC] = perfcurve(trueBinary, scores, true);
    plot(X, Y, 'DisplayName', sprintf('%s (AUC = %.2f)', classNames{i}, AUC));
end
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('ROC Curves for Each Class');
legend('show');
grid on;
hold off;
exportgraphics(gcf, fullfile(resultsFolder, 'roc_curves.png'));
