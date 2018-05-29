% 
% clc;clear;close all
% %data = load('breast_cancer_wisconsin_data.mat','-ascii'); function_type = 'c';
% for k = [1,2,3,4,5,6,8,10]
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
% class2Detect = 7;
% xvalOrTest='x';%x or 't' --x=xval; t=tes
% 
% % function_type = 'c'; %'c' - classification, 'f' - regression  
% kernel_type = 'RBF_kernel'; % or 'lin_kernel', 'poly_kernel'
% global_opt = 'csa'; % 'csa' or 'ds'
% 
% 
% % indexotherClasses = [find(Y==6)]
% %indexotherClasses = [find(Y==2);find(Y==3);find(Y==1);find(Y==5);find(Y==6);find(Y==7)];
% 
% 
% index2delete = sort([find(Y==1);find(Y==2);find(Y==3);find(Y==4);find(Y==5)]);
% X(index2delete,:)=[];
% Y(index2delete)=[];
% 
% Y(find(Y==class2Detect))=1;
% Y(find(Y==6))=-1;
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


kernel_type = 'RBF_kernel'; % or 'lin_kernel', 'poly_kernel'
global_opt = 'csa'; % 'csa' or 'ds'

%PREPATARION OF TRAINING
class2Detect = 7;
%indexotherClasses = [find(Y==2);find(Y==3);find(Y==1);find(Y==5);find(Y==6);find(Y==7)];


index2delete = sort([find(Y==1);find(Y==2);find(Y==3);find(Y==4);find(Y==5)]);
X(index2delete,:)=[];
Y(index2delete)=[];

Y(find(Y==class2Detect))=1;
Y(find(Y==6))=-1;


%PREPARATION OF TEST 
load('SHUTTLE_RESULTS/CLASS2/k3RBFindexClass4-1-5-3-2Detected.mat','-ascii')
indexOriginal = 1:14500;
indexOriginal(k3RBFindexClass4_1_5_3_2Detected) =[]; 
testX(k3RBFindexClass4_1_5_3_2Detected, :) = [];
testY(k3RBFindexClass4_1_5_3_2Detected) = [];
indexotherClassesTest = [find(testY==3);find(testY==5);find(testY==1);find(testY==4);find(testY==6);find(testY==2)];
testY(find(testY==class2Detect))=1;
testY(indexotherClassesTest) = -1;


% % TEST
load('SHUTTLE_RESULTS/CLASS7/ncstdxval_RBF_kernelerror_k1.mat','-ascii')
svX = ncstdxval_RBF_kernelerror_k1(:,2:end-1)
% [gam,sig]=tunefslssvm({X,Y,'c',[],[],kernel_type,global_opt},svX,10,'misclass','simplex');
gam = 1.7118865;
sig = 3534.4793;

features = AFEm(svX,kernel_type,sig,X);
testfeatures = AFEm(svX,kernel_type,sig,testX);
[w,b,testYh] = ridgeregress(features,Y,gam,svX,testfeatures);
testYh = sign(testYh);
err = sum(testYh~=testY)/length(testYh);
C = confusionmat(testY,testYh)
% % plotconfusion(testY,testYh)

testYh = [indexOriginal', testYh];
indexClass7Detected = find(testYh(:,2)==1);
indexClass6Detected = find(testYh(:,2)==-1);

testYh7 = testYh(indexClass7Detected,:);
testYh6 = testYh(indexClass6Detected,:);

originalIndexingclass7 = testYh7(:,1);
originalIndexingclass6 = testYh6(:,1);


noInputNextModelIndx = sort([k3RBFindexClass4_1_5_3_2Detected;originalIndexingclass7]);

 
csvwrite(strcat('SHUTTLE_RESULTS/CLASS7/k3RBFindexClass4-1-5-3-2-7Detected.mat'),noInputNextModelIndx);
csvwrite(strcat('SHUTTLE_RESULTS/CLASS7/k3RBFindexClass7Detected.mat'),originalIndexingclass7);
csvwrite(strcat('SHUTTLE_RESULTS/CLASS7/k3RBFindexClass6Detected.mat'),originalIndexingclass6);