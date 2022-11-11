% calculation:
% training set: 170*25*0.7+7=2982 testing set: 170*25*0.3+3=1278
% in this project first 25 of CUM PIE is selected
clear;
clc;
for k=1:5
orig = imread(['./EE5907/',num2str(k-1),'.jpg']);	%jpg格式或者其他图片格式,自行传入文件绝对路径
[h,w,rgb] = size(orig);	%获取高度h和宽度w
m = 5;  %高度缩小倍数
n = 5;  %宽度缩小倍数
for i = 1:floor(h/m)
    for j = 1:floor(w/n)
        suo(i,j,:) = uint8( mean(mean(orig((i-1)*m+1:i*m,(j-1)*n+1:j*n,:),1),2) );
    end
end
imwrite(suo,['./EE5907/test',num2str(k-1),'.jpg'])

end