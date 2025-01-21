% Th? m?c ch?a ?nh hu?n luy?n
trainFolder = 'E:/MY_PROJECT/FaceQualityAssessment/data/Flickr-Faces-lite';

% T?o ImageDatastore t? th? m?c
imds = imageDatastore(trainFolder, 'FileExtensions', {'.jpg', '.png'});
% Danh s�ch l?u ?nh
% trainImages = [];
% for i = 1:length(imageFiles)
%     % ??c ?nh v� chuy?n v? grayscale
%     img = imread(fullfile(trainFolder, imageFiles(i).name));
%     grayImg = rgb2gray(img);
%     trainImages = cat(3, trainImages, grayImg);
% end

% % Hu?n luy?n m� h�nh NIQE
% model = fitniqe(imds);
% 
% % L?u m� h�nh ra file (n?u c?n)
% save('niqe_model.mat', 'model');

I = imread("data/VN-celeb/7/29.png");
imshow(I)

niqeI = niqe(I,model);
disp("NIQE score for the image is "+niqeI)