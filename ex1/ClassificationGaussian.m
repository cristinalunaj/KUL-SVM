%% Simple example of classification: 2 gaussians: 
clc;clear;close all
X1 = 1 + randn(50,2);
X2 = -1 + randn(51,2);
x = [-6:.1:6];
figure
norm1 = normpdf(x,1,1);
norm2 = normpdf(x,-1,1);

plot(x,norm1, 'r')
hold on
plot(x,norm2, 'b')
legend('class1', 'class2');
title('Classes distribution');
hold off


Y1 = ones(50,1);
Y2 = -ones(51,1);

X = [X1;X2];
Y = [Y1;Y2];

figure;
hold on;
plot(X1(:,1),X1(:,2),'ro');
plot(X2(:,1),X2(:,2),'bo');
% xlabel('X1')
% yabel('X2')
legend('class1', 'class2');
hold off;