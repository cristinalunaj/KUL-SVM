%% see histograms per feature

% data = load('shuttle.dat','-ascii'); function_type = 'c';  data = data(1:end,:);
% 
% 
% class1 = find(Y==1)
% class0 = [find(Y==2);find(Y==3);find(Y==4);find(Y==5);find(Y==6);find(Y==7)]
% 
% 
% for feat = 1:size(X,2)
% 
% %     max = max(Xt(:,feat));
% %     min = min(Xt(:,feat));
% %     range = max-min;
%     figure;
%     hold on;
%     histogram(X(class1,feat),50)
%     hold on;
%     histogram(X(class0,feat),50)
%     grid on
%     title(strcat('Shuttle DataSet - x',num2str(feat)))
%     xlabel(strcat('X',num2str(feat)))
% %     saveas(gcf,strcat('DIABETES_IMG/feat',num2str(feat),'.jpg'))
% %     savefig(strcat('DIABETES_IMG/feat',num2str(feat),'.fig'))
% 
% end

%% see histogram per feature removing 1
clc;clear;close all
data = load('shuttle.dat','-ascii'); function_type = 'c';  data = data(1:end,:);

X = data(1:43500,1:end-1);
Y = data(1:43500,end);
% class1 = find(Y==1)
% X(class1,:)=[];
% Y(class1)=[];

class2detect = 5;
class0 = find(Y==class2detect) % class2 = 37 samples; class3 = 132; class4 =6748; class5=2458; class6=6; class7=11
% otherClass = [find(Y==2);find(Y==3);find(Y==4);find(Y==5);find(Y==6)]

%%class4 
indexClass1 = find(Y==1);
otherClass = [indexClass1(1:3000);find(Y==2);find(Y==3);find(Y==4);find(Y==6);find(Y==7)]



for feat = 1:size(X,2)

%     max = max(Xt(:,feat));
%     min = min(Xt(:,feat));
%     range = max-min;
    figure;
    hold on;
    histogram(X(otherClass,feat),50)
    hold on;
    histogram(X(class0,feat),50)
    grid on
    title(strcat('Shuttle DataSet - x',num2str(feat)))
    xlabel(strcat('X',num2str(feat)))
    saveas(gcf,strcat('visualization/shuttle/CLASS_',num2str(class2detect),'_feat',num2str(feat),'.jpg'))
    savefig(strcat('visualization/shuttle/CLASS_',num2str(class2detect),'_feat',num2str(feat),'.fig'))

end
