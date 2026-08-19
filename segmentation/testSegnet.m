% Test the trained U-Net segmentation model

% Resolve paths relative to the project root
scriptFolder = fileparts(mfilename('fullpath'));
projectRoot = fileparts(scriptFolder);

modelsFolder = fullfile(projectRoot, 'models');
resultsFolder = fullfile(projectRoot, 'results');
imageFolder = fullfile(projectRoot, ...
    'daffodilSeg', 'daffodilSeg', 'ImagesRsz256');

% Ensure results directory exists
if ~exist(resultsFolder, 'dir')
    mkdir(resultsFolder);
end

% Step 1: Load Trained U-Net Model
load(fullfile(modelsFolder, 'segnet.mat'));

% Step 2: Load and Preprocess Input Image
inputImagePath = fullfile(imageFolder, 'image_0001.png');
inputImage = imread(inputImagePath);

inputSize = [256 256];
resizedImage = imresize(inputImage, inputSize);

% Step 3: Perform Semantic Segmentation
segmentedLabels = semanticseg(resizedImage, segnet);

% Step 4: Create Binary Mask for the 'flower' class
flowerMask = segmentedLabels == 'flower';

% Step 5: Save Binary Mask
maskOutputPath = fullfile(resultsFolder, 'flower_mask.png');
imwrite(uint8(flowerMask) * 255, maskOutputPath);

% Step 6: Create and Save Segmentation Overlay
overlayImage = labeloverlay(resizedImage, segmentedLabels);
overlayOutputPath = fullfile(resultsFolder, 'segmentation_overlay.png');
imwrite(overlayImage, overlayOutputPath);

disp('Segmentation complete. Mask and overlay saved to the results directory.');