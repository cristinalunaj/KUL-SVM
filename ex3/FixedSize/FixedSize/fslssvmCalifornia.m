% clc;clear;close all
% %data = load('breast_cancer_wisconsin_data.mat','-ascii'); function_type = 'c';
% %data = load('shuttle.dat','-ascii'); function_type = 'c';  data = data(1:end,:);
% data = load('california.dat','-ascii'); function_type = 'f';
% 
% %index_DS = randperm(20640)
% load('index_california.mat')
% data = data(index_DS,:);
% 
% 
% X = data(1:14448,1:end-1);
% Y = data(1:14448,end);
% meanTraining = mean(X);
% stdTrainign = std(X);
% minTraining = min(X);
% maxTrainign = max(X);
% 
% %standarization
% X = ((X-meanTraining)./(stdTrainign));
% 
% 
% % %feature scaling
% % % X(:,1) = (X(:,1)-meanTraining(1))./(maxTrainign(1)-minTraining(1));
% % X = ((X-minTraining)./(maxTrainign-minTraining))
% % % X = X-meanTraining./stdTrainign;
% newMin = min(X)
% newMax = max(X)
% 
% testX = data(14449:end,1:end-1);
% testY = data(14449:end,end);
% testX = ((testX-meanTraining)./(stdTrainign));
% % Xtest = ((Xtest-meanTraining)./(maxTrainign-minTraining))
% % 
% % histogram(Y, 50)
% % hold on
% % histogram(testY,50)
% % legend('train', 'test')
% % % LABELS STADISTICS
% % maxLabel = max(data(:,end))
% % minLabel = min(data(:,end))
% % avgLabel = mean(data(:,end))
% % stdLabel = std(data(:,end))
% % 
% 
% xvalOrTest='x';%x or 't' --x=xval; t=test
% 
% 
% 
% % function_type = 'c'; %'c' - classification, 'f' - regression  
% kernel_type = 'poly_kernel'; % or 'lin_kernel', 'poly_kernel'
% global_opt = 'csa'; % 'csa' or 'ds'
% 
% %Process to be performed
% user_process={'FS-LSSVM','SV_L0_norm'};%, 'SV_L0_norm'
% window = [15,20,25];
% 
% 
% for k = [2]
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
% csvwrite(strcat('CALIFORNIA_HOUSING/modelVariables',kernel_type,tup,'error_k',num2str(k),'.mat'),model);
% 
% close all
% figure;
% boxplot(e,'Label',user_process);
% hold on
% ylabel('Error estimate');
% title('Error Comparison for different approaches (user processes)');
% hold off
% savefig(strcat('CALIFORNIA_HOUSING/',kernel_type,tup,'error_k',num2str(k),'.fig'))
% close all
% 
% 
% figure;
% boxplot(s,'Label',user_process);
% hold on
% ylabel('SV estimate');
% title('Number of SV for different approaches (user processes)');
% hold off
% savefig(strcat('CALIFORNIA_HOUSING/',kernel_type,tup,'SV_k',num2str(k),'.fig'))
% close all
% 
% 
% figure;
% boxplot(t,'Label',user_process);
% hold on
% ylabel('Time estimate');
% title('Comparison for time taken by different approaches (user processes)');
% hold off
% savefig(strcat('CALIFORNIA_HOUSING/',kernel_type,tup,'time_k',num2str(k),'.fig'))
% close all
% 
% end
%% testing model

clc;clear;close all
load('CALIFORNIA_HOUSING/modelVariablesRBF_kernelxval_error_k15.mat', '-ascii')
data = load('california.dat','-ascii'); function_type = 'f';
load('index_california.mat')
data = data(index_DS,:);
svX = modelVariablesRBF_kernelxval_error_k15(:,2:end-1);
%data = data(index_DS,:) 
X = data(1:14448,1:end-1);
Y = data(1:14448,end);
meanTraining = mean(X);
stdTrainign = std(X);
minTraining = min(X);
maxTrainign = max(X);




X = ((X-meanTraining)./(stdTrainign));

testX = data(14449:end,1:end-1);
testY = data(14449:end,end);

testX = ((testX-meanTraining)./(stdTrainign));
%kernel_type = 'RBF_kernel';
%standarization

%INITIALIZATION: 
data = [X];
N = size(X,1);
avg_training = mean(data);
std_training = std(data);
C = data-repmat(avg_training,length(data),1);  
D = std_training;
for i=1:size(C,2)
    if (D(i)~=0)
        C(:,i) = C(:,i)/D(i);
    end;
end;
X = C;

data = [Y];
avg_training_Y = mean(data);
std_training_Y = std(data);
C = data-repmat(avg_training_Y,length(data),1);  
D = std_training_Y;%std(data);
for i=1:size(C,2)
    if (D(i)~=0)
        C(:,i) = C(:,i)/D(i);
    end;
end;
Y = C;


data = [testX];
C = data-repmat(avg_training,length(data),1);  
D = std_training;
for i=1:size(C,2)
    if (D(i)~=0)
        C(:,i) = C(:,i)/D(i);
    end;
end;
testX = C;

data = [testY];
C = data-repmat(avg_training_Y,length(data),1);  
D = std_training_Y;
for i=1:size(C,2)
    if (D(i)~=0)
        C(:,i) = C(:,i)/D(i);
    end;
end;
testY = C;


%training
kernel_type = 'RBF_kernel';

%[gam,sig]=tunefslssvm({X,Y,'f',[],[],kernel_type,'csa'},svX,10,'mse','simplex','whuber');
%RBF, k=3
% gam = 0.0126207;
% sig = 20.6934;
%RBF, k=15
gam = 0.002594857042035;
sig = 7.782657971359048;


%TESTING
% [tim,err,newsvX,newsvY] = operations(X,Y,train,validation,svX,svY,subset,sig,gam,kernel_type,'f',process_type(k),[]);
% [err,newsvX,newsvY] = modsparseoperations(X,Y,train,validation,svX,svY,subset,sig,gam,kernel_type,function_type,process_type,windowsize);
%svx = support vectors =361x8
features = AFEm(svX,kernel_type,sig,X); %features train
featuresTest = AFEm(svX,kernel_type,sig,testX);

%Perform the FS-LSSVM based regression
if (function_type=='f')
    trainY = Y;
    [w,b,testYh] = ridgeregress(features,trainY,gam,svX,featuresTest);
    %testYh2=(testYh*std_training_Y)+avg_training_Y;
    errMSE = mse(testYh-testY);
    errMAE = mae(testYh-testY); 
end

%% representation
% data = load('california.dat','-ascii'); function_type = 'f';
% load('index_california.mat')
% 
% data = data(index_DS,:);
% X = data(1:14448,1:end-1);
% Y = data(1:14448,end);
% labels = {'X1' 'X2' 'X3' 'X4' 'X5' 'X6' 'X7' 'X8' 'Y'};
% figure
% data = [X(:,1) X(:,2) X(:,3) X(:,4) X(:,5) X(:,6) X(:,7) X(:,8) Y];
% 
% [h,ax] = plotmatrix(data); 
% 
% for i = 1:9                                      % label the plots
%   xlabel(ax(9,i), labels{i})
%   ylabel(ax(i,1), labels{i})
% end

index1 = 1;
index2 = 4; 
index3 = 5;
figure;
plot3(featuresTest(:,index1),             featuresTest(:,index2),             featuresTest(:,index3),'b*'); hold on;
plot3(features(modelVariablesRBF_kernelxval_error_k15(:,1),index1),features(modelVariablesRBF_kernelxval_error_k15(:,1),index2),features(modelVariablesRBF_kernelxval_error_k15(:,1),index3),'r+','linewidth',6); hold off;
title('feature space for test set')
grid on;

figure;
plot3(features(:,index1),             features(:,index2),             features(:,index3),'g*'); hold on;
plot3(features(modelVariablesRBF_kernelxval_error_k15(:,1),index1),features(modelVariablesRBF_kernelxval_error_k15(:,1),index2),features(modelVariablesRBF_kernelxval_error_k15(:,1),index3),'r+','linewidth',6); hold off;
title('feature space for training set')
grid on

