% 
% clc;clear;close all
% 
% %data = load('breast_cancer_wisconsin_data.mat','-ascii'); function_type = 'c';
% data = load('shuttle.dat','-ascii'); function_type = 'c';  data = data(1:end,:);
% 
% for k=[3,6,10,20]
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
% % function_type = 'c'; %'c' - classification, 'f' - regression  
% kernel_type = 'RBF_kernel'; % or 'lin_kernel', 'poly_kernel'
% global_opt = 'csa'; % 'csa' or 'ds'
% 
% indexClass1 = find(Y==1);
% indexotherClasses = [indexClass1(1:3000);find(Y==2);find(Y==3);find(Y==5);find(Y==6);find(Y==7)]
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
% % indexotherClassestest = [find(testY==1);find(testY==2);find(testY==3);find(testY==5);find(testY==6);find(testY==7)]
% % testY(find(testY==class2Detect))=1;
% % testY(indexotherClassestest)=-1;
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
% 
% end

%% TEST
clc;clear;close all
data = load('shuttle.dat','-ascii'); function_type = 'c';  data = data(1:end,:);
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

class2Detect = 4;
kernel_type = 'RBF_kernel'; % or 'lin_kernel', 'poly_kernel'
global_opt = 'csa'; % 'csa' or 'ds'

%PREPATARION OF TRAINING
indexClass1 = find(Y==1);
indexotherClasses = [indexClass1(1:3000);find(Y==2);find(Y==3);find(Y==5);find(Y==6);find(Y==7)]
Y(find(Y==class2Detect))=1;
Y(indexotherClasses)=-1;

X(indexClass1(3001:end),:)=[];
Y(indexClass1(3001:end))=[];
%PREPARATION OF TEST 
load('SHUTTLE_RESULTS/CLASS1/k3RBFindexClass1Detected.mat','-ascii')
indexOriginal = 1:14500;
indexOriginal(k3RBFindexClass1Detected) =[]; 
testX(k3RBFindexClass1Detected, :) = [];
testY(k3RBFindexClass1Detected) = [];
% a = find(testY==1)
indexotherClassesTest = [find(testY==2);find(testY==3);find(testY==1);find(testY==5);find(testY==6);find(testY==7)];
testY(find(testY==class2Detect))=1;
testY(indexotherClassesTest) = -1;

% % TEST
load('SHUTTLE_RESULTS/CLASS4/ncstdxval_RBF_kernelerror_k6.mat','-ascii')
svX = ncstdxval_RBF_kernelerror_k6(:,2:end-1)
%[gam,sig]=tunefslssvm({X,Y,'c',[],[],kernel_type,global_opt},svX,10,'misclass','simplex');
gam = 0.51632; 
sig = 1.8028;

features = AFEm(svX,kernel_type,sig,X);
testfeatures = AFEm(svX,kernel_type,sig,testX);
[w,b,testYh] = ridgeregress(features,Y,gam,svX,testfeatures);
testYh = sign(testYh);
err = sum(testYh~=testY)/length(testYh);
C = confusionmat(testY,testYh)
% % plotconfusion(testY,testYh)

testYh = [indexOriginal', testYh];
indexClass4Detected = find(testYh(:,2)==1);

testYh = testYh(indexClass4Detected,:);
originalIndexingclass4 = testYh(:,1);
noInputNextModelIndx = sort([k3RBFindexClass1Detected;originalIndexingclass4]);
% testX(noInputNextModelIndx, :) = [];
% testY(noInputNextModelIndx) = [];
% csvwrite(strcat('SHUTTLE_RESULTS/CLASS4/k3RBFindexClass4Detected.mat'),originalIndexingclass4);
% csvwrite(strcat('SHUTTLE_RESULTS/CLASS4/k3RBFindexClass1-4Detected.mat'),noInputNextModelIndx);


% 
% testFeatures1Index = find(testY==1);
% testFeaturesnon1Index = find(testY~=1);
% 
% 
% index1 = 1;
% index2 = 7; 
% index3 = 3;
% 
% plot3(testfeatures(testFeatures1Index,index1),             testfeatures(testFeatures1Index,index2),             testfeatures(testFeatures1Index,index3),'b*'); hold on;
% plot3(testfeatures(testFeaturesnon1Index,index1),             testfeatures(testFeaturesnon1Index,index2),             testfeatures(testFeaturesnon1Index,index3),'g*'); hold on;
% 
% 
% plot3(features(ncstdxval_RBF_kernelerror_k6(:,1),index1),features(ncstdxval_RBF_kernelerror_k6(:,1),index2),features(ncstdxval_RBF_kernelerror_k6(:,1),index3),'r+','linewidth',6); hold off;
% title('feature space for test set - class 4')
% grid on
% pause(1)
% class2detect=4
% saveas(gcf,strcat('visualization/shuttle/CLASS_',num2str(class2detect),'AFEm_test.jpg'))
% savefig(strcat('visualization/shuttle/CLASS_',num2str(class2detect),'AFEm_test.fig'))
% close all
% trainFeatures1Index = find(Y==1);
% trainFeaturesnon1Index = find(Y~=1);
% 
% figure;
% plot3(features(trainFeatures1Index,index1),             features(trainFeatures1Index,index2),             features(trainFeatures1Index,index3),'b*'); hold on;
% plot3(features(trainFeaturesnon1Index,index1),             features(trainFeaturesnon1Index,index2),             features(trainFeaturesnon1Index,index3),'g*'); hold on;
% 
% 
% plot3(features(ncstdxval_RBF_kernelerror_k6(:,1),index1),features(ncstdxval_RBF_kernelerror_k6(:,1),index2),features(ncstdxval_RBF_kernelerror_k6(:,1),index3),'r+','linewidth',6); hold off;title('feature space for train set - class 1')
% title('feature space for train set - class 4')
% grid on
% pause(1)
% 
% 
% saveas(gcf,strcat('visualization/shuttle/CLASS_',num2str(class2detect),'AFEm_train.jpg'))
% savefig(strcat('visualization/shuttle/CLASS_',num2str(class2detect),'AFEm_train.fig'))
% 
% 
% 
