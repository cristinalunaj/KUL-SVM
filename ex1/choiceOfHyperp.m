%% section 1.3 
% set the parameters to some value
clc;clear;close all
load iris
load('CH_Xtrain.mat')
load('CH_Ytrain.mat')
load('CH_Xval.mat')
load('CH_Yval.mat')
% generate random indices
% idx = randperm(size(X,1));
% create the training and validation sets
% using the randomized indices
% Xtrain = X(idx(1:80),:);
% Ytrain = Y(idx(1:80));
% Xval = X(idx(81:100),:);
% Yval = Y(idx(81:100));
% save('CH_Xtrain.mat','Xtrain')
% save('CH_Ytrain.mat','Ytrain')
% save('CH_Xval.mat','Xval')
% save('CH_Yval.mat','Yval')


sigList = [0.1,1,10,100,1000];
gammaList = [1,10,100,500,1000];
sig2new = 20
err = []
errlist = []
type = 'c'
for gam = gammaList
    for sig2=sigList,
        disp(['gam : ', num2str(gam), '   sig2 : ', num2str(sig2)]),
        [alpha,b] = trainlssvm({Xtrain,Ytrain,'c',gam,sig2,'RBF_kernel'});

        % Plot the decision boundary of a 2-d LS-SVM classifier
        plotlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel'},{alpha,b});
        estYval = simlssvm({Xtrain,Ytrain,'c',gam,sig2,'RBF_kernel'},{alpha,b},Xval);
        % Obtain the output of the trained classifier
        err = [err, sum(estYval~=Yval)]; 
    end
    errlist=[errlist; err];
    err=[];
end

figure;
for i=1:5
    plot(log(sigList), errlist(i,:), 'o-')
    hold on
    
end
xlabel('log(sig2)'), ylabel('number of misclass'),
legend('gamma1','gamma10','gamma100','gamma500','gamma1000');

gam = 1
sig2 = 5

%% xvalidation
close all
err = []
errlist=[]
sigList = [0.1,1,10,100,1000];
gammaList = [1,10,100,500,1000];
for gam = gammaList
    for sig2=sigList,
        disp(['gam : ', num2str(gam), '   sig2 : ', num2str(sig2)]),
        % Plot the decision boundary of a 2-d LS-SVM classifier
        performance = leaveoneout({X,Y,'c',gam,sig2,'RBF_kernel'},'misclass')%crossvalidate({X,Y,'c',gam,sig2,'RBF_kernel'}, 10,'misclass');
        % Obtain the output of the trained classifier
        err = [err, performance]; 
    end
    errlist=[errlist; err];
    err=[];
end

% performance = crossvalidate({X,Y,'c',gam,sig2,'RBF_kernel'},10,'misclass')
% 
% performanceLeave1 = leaveoneout({X,Y,'c',gam,sig2,'RBF_kernel'},'misclass')

%% OPTIMIZE PARAMETERS 

par1 = ["csa","ds"];
par2 = ["simplex","gridsearch"];

parametersList = []
for i=1:2
    for j=1:2
        tic
        p1 = char(par1(i))
        p2 = char(par2(j))
        model = {X,Y,'c',[],[],'RBF_kernel',p1};
        [gam,sig2,cost] = tunelssvm(model,p2,'crossvalidatelssvm',{10,'misclass'});
        time = toc
        parametersList = [parametersList; [gam,sig2,cost, time]]
    end
end

%% ROC
gam = 1
sig2 = 10
[alpha,b] = trainlssvm({Xtrain,Ytrain,'c',gam,sig2, ...
'RBF_kernel'});
[Ysim,Ylatent] = simlssvm({Xtrain,Ytrain,'c',gam,sig2, ...
'RBF_kernel'},{alpha,b},Xval);
roc(Ylatent,Yval);





