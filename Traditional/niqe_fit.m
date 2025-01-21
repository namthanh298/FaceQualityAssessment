% Th? m?c ch?a ?nh hu?n luy?n
trainFolder = 'E:/MY_PROJECT/FaceQualityAssessment/data/Flickr-Faces-lite';

% T?o ImageDatastore t? th? m?c
imds = imageDatastore(trainFolder, 'FileExtensions', {'.jpg', '.png'});
% Danh sách l?u ?nh
% trainImages = [];
% for i = 1:length(imageFiles)
%     % ??c ?nh và chuy?n v? grayscale
%     img = imread(fullfile(trainFolder, imageFiles(i).name));
%     grayImg = rgb2gray(img);
%     trainImages = cat(3, trainImages, grayImg);
% end

% % Hu?n luy?n mô hình NIQE
% model = fitniqe(imds);
% 
% % L?u mô hình ra file (n?u c?n)
% save('niqe_model.mat', 'model');

I = imread("data/VN-celeb/7/29.png");
imshow(I)

niqeI = niqe(I,model);
disp("NIQE score for the image is "+niqeI)