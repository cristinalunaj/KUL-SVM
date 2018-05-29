clc;clear;close all
%data = load('breast_cancer_wisconsin_data.mat','-ascii'); function_type = 'c';
data = load('shuttle.dat','-ascii'); function_type = 'c';  data = data(1:end,:);
% data = load('california.dat','-ascii'); function_type = 'f';



addpath('../LSSVMlab')

X = data(1:43500,1:end-1);
Y = data(1:43500,end);
testX = data(43501:end,1:end-1);
testY = data(43501:end,end);

meanTraining = mean(X);
stdTrainign = std(X);
minTraining = min(X);
maxTrainign = max(X);
%standarization
X = ((X-meanTraining)./(stdTrainign));
%standaruzation
testX = ((testX-meanTraining)./(stdTrainign));


indexClass1 = [find(Y==2);find(Y==3);find(Y==4);find(Y==5);find(Y==6);find(Y==7)];
indexClass1Test = [find(testY==2);find(testY==3);find(testY==4);find(testY==5);find(testY==6);find(testY==7)];
Y(indexClass1) = -1;
testY(indexClass1Test) = -1;


% Parameter for input space selection
% Please type >> help fsoperations; to get more information  

k = 2;
% function_type = 'c'; %'c' - classification, 'f' - regression  
kernel_type = 'RBF_kernel'; % or 'lin_kernel', 'poly_kernel'
global_opt = 'csa'; % 'csa' or 'ds'

%Process to be performed
user_process={'FS-LSSVM', 'SV_L0_norm'};
window = [15,20,25];

[model,e,s,t] = fslssvm(X,Y,k,function_type,kernel_type,global_opt,user_process,window,[],[]);



%TEST
% svX = load('modelvariables.mat','-ascii')
% [gam,sig]=tunefslssvm({X,Y,'c',[],[],kernel_type,global_opt},svX,10,'misclass','simplex');
%svx=subset of SVs
% features = AFEm(svX,kernel_type,sig,X);
% testfeatures = AFEm(svX,kernel_type,sig,testX);
% [w,b,testYh] = ridgeregress(features,Y,gam,svX,testfeatures);
% testYh = sign(testYh);
% err = sum(testYh~=testY)/length(testYh);

%% model 2 for distinguish between classes 4 and 5

% 
% clc;clear;close all
% %data = load('breast_cancer_wisconsin_data.mat','-ascii'); function_type = 'c';
% data = load('shuttle.dat','-ascii'); function_type = 'c';  data = data(1:end,:);
% X = data(1:43500,1:end-1);
% Y = data(1:43500,end);
% testX = data(43501:end,1:end-1);
% testY = data(43501:end,end);
% 
% meanTraining = mean(X);
% stdTrainign = std(X);
% minTraining = min(X);
% maxTrainign = max(X);
% %standarization
% X = ((X-meanTraining)./(stdTrainign));
% %standaruzation
% testX = ((testX-meanTraining)./(stdTrainign));
% 
% 
% 
% class2Detect = 4;
% xvalOrTest='x';%x or 't' --x=xval; t=tes
% k = 6;
% % function_type = 'c'; %'c' - classification, 'f' - regression  
% kernel_type = 'RBF_kernel'; % or 'lin_kernel', 'poly_kernel'
% global_opt = 'csa'; % 'csa' or 'ds'
% 
% indexClass1 = find(Y==1);
% indexotherClasses = [indexClass1(1:3000);find(Y==2);find(Y==3);find(Y==6);find(Y==7);find(Y==5)]
% %indexotherClasses = [find(Y==2);find(Y==3);find(Y==1);find(Y==5);find(Y==6);find(Y==7)];
% 
% 
% Y(find(Y==class2Detect))=1;
% Y(indexotherClasses)=-1;
% 
% X(indexClass1(3001:end),:)=[];
% Y(indexClass1(3001:end))=[];
% 
% %TEST
% indexotherClassestest = [find(testY==1);find(testY==2);find(testY==3);find(testY==6);find(testY==7);find(testY==5)]
% testY(find(testY==class2Detect))=1;
% testY(indexotherClassestest)=-1;
% 
% %Process to be performed
% user_process={'FS-LSSVM', 'SV_L0_norm'};
% window = [15,20,25];
% 
% if(xvalOrTest=='x')
%     tup = 'xval_'
%     [model,e,s,t] = fslssvm(X,Y,k,function_type,kernel_type,global_opt,user_process,window,[],[]);
% elseif(xvalOrTest=='t')
%     tup='test_'
%     [model, e,s,t] = fslssvm(X,Y,k,function_type,kernel_type,global_opt,user_process,window,testX,testY);
% else
%     a = 'erorrrr'
% end
% csvwrite(strcat('SHUTTLE_RESULTS/CLASS',num2str(class2Detect),'/ncstd',tup,kernel_type,'error_k',num2str(k),'.mat'),model);
% 
% close all
% figure;
% boxplot(e,'Label',user_process);
% hold on
% ylabel('Error estimate');
% title('Error Comparison for different approaches (user processes)');
% hold off
% savefig(strcat('SHUTTLE_RESULTS/CLASS',num2str(class2Detect),'/ncstd',tup,kernel_type,'error_k',num2str(k),'.fig'))
% close all
% 
% figure;
% boxplot(s,'Label',user_process);
% hold on
% ylabel('SV estimate');
% title('Number of SV for different approaches (user processes)');
% hold off
% savefig(strcat('SHUTTLE_RESULTS/CLASS',num2str(class2Detect),'/ncstd',tup,kernel_type,'SV_k',num2str(k),'.fig'))
% close all
% 
% figure;
% boxplot(t,'Label',user_process);
% hold on
% ylabel('Time estimate');
% title('Comparison for time taken by different approaches (user processes)');
% hold off
% savefig(strcat('SHUTTLE_RESULTS/CLASS',num2str(class2Detect),'/ncstd',kernel_type,tup,'time_k',num2str(k),'.fig'))
% close all

%% TESTING


clc;clear;close all
%data = load('breast_cancer_wisconsin_data.mat','-ascii'); function_type = 'c';
data = load('shuttle.dat','-ascii'); function_type = 'c';  data = data(1:end,:);

addpath('../LSSVMlab')

X = data(1:43500,1:end-1);
Y = data(1:43500,end);
testX = data(43501:end,1:end-1);
testY = data(43501:end,end);
meanTraining = mean(X);
stdTrainign = std(X);
minTraining = min(X);
maxTrainign = max(X);
% standarization
X = ((X-meanTraining)./(stdTrainign));


indexClass1 = [find(Y==2);find(Y==3);find(Y==4);find(Y==5);find(Y==6);find(Y==7)];
indexClass1Test = [find(testY==2);find(testY==3);find(testY==4);find(testY==5);find(testY==6);find(testY==7)];
Y(indexClass1) = -1;
testY(indexClass1Test) = -1;
global_opt = 'csa'
kernel_type = 'RBF_kernel';
testX = ((testX-meanTraining)./(stdTrainign));



% % TEST
load('SHUTTLE_RESULTS/CLASS1/ncstd_xvalRBF_kernelerror_k6.mat','-ascii')
svX = ncstd_xvalRBF_kernelerror_k6(:,2:end-1)
%[gam,sig]=tunefslssvm({X,Y,'c',[],[],kernel_type,global_opt},svX,10,'misclass','simplex');
% [tim,err,newsvX,newsvY] = operations(X,Y,train,validation,svX,svY,subset,sig,gam,kernel_type,'c',process_type(k),[]);
% gam = 24.968963;
% sig=8705.3758;
% svx=subset of SVs
% fo k=3 goood
% gam = 7.17409;
% sig=16.0603;
%for k = 6: 
gam = 0.51124;
sig = 2.1887;

features = AFEm(svX,kernel_type,sig,X);
testfeatures = AFEm(svX,kernel_type,sig,testX);
[w,b,testYh] = ridgeregress(features,Y,gam,svX,testfeatures);
testYh = sign(testYh);
err = sum(testYh~=testY)/length(testYh);

C = confusionmat(testY,testYh)

% % plotconfusion(testY,testYh)

noInputNextModelIndx = find(testYh==1);
% testX(noInputNextModelIndx, :) = [];
% testY(noInputNextModelIndx) = [];
% csvwrite(strcat('SHUTTLE_RESULTS/CLASS1/k3RBFindexClass1Detected.mat'),noInputNextModelIndx);

testFeatures1Index = find(testY==1);
testFeaturesnon1Index = find(testY~=1);


index1 = 1;
index2 = 7; 
index3 = 9;

figure;
plot3(testfeatures(testFeatures1Index,index1),             testfeatures(testFeatures1Index,index2),             testfeatures(testFeatures1Index,index3),'b*'); hold on;
plot3(testfeatures(testFeaturesnon1Index,index1),             testfeatures(testFeaturesnon1Index,index2),             testfeatures(testFeaturesnon1Index,index3),'g*'); hold on;


plot3(features(ncstd_xvalRBF_kernelerror_k6(:,1),index1),features(ncstd_xvalRBF_kernelerror_k6(:,1),index2),features(ncstd_xvalRBF_kernelerror_k6(:,1),index3),'r+','linewidth',6); hold off;
title('feature space for test set - class 1')
grid on
pause(1)
class2detect = 1;
saveas(gcf,strcat('visualization/shuttle/CLASS_',num2str(class2detect),'AFEm_test.jpg'))
savefig(strcat('visualization/shuttle/CLASS_',num2str(class2detect),'AFEm_test.fig'))
close all

% 
% trainFeatures1Index = find(Y==1);
% trainFeaturesnon1Index = find(Y~=1);
% 
% figure;
% plot3(features(trainFeatures1Index,index1),             features(trainFeatures1Index,index2),             features(trainFeatures1Index,index3),'b*'); hold on;
% plot3(features(trainFeaturesnon1Index,index1),             features(trainFeaturesnon1Index,index2),             features(trainFeaturesnon1Index,index3),'g*'); hold on;
% 
% 
% plot3(features(ncstd_xvalRBF_kernelerror_k6(:,1),index1),features(ncstd_xvalRBF_kernelerror_k6(:,1),index2),features(ncstd_xvalRBF_kernelerror_k6(:,1),index3),'r+','linewidth',6); hold off;
% title('feature space for train set - class 1')
% grid on
% pause(1)
% 
% saveas(gcf,strcat('visualization/shuttle/CLASS_',num2str(class2detect),'AFEm_train.jpg'))
% savefig(strcat('visualization/shuttle/CLASS_',num2str(class2detect),'AFEm_train.fig'))

%class4 model

%class5 model

%class3 model 

%class2 model

%class6&7 model



%% VISUALIZATION

