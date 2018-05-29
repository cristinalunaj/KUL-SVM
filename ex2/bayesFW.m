% 2.4 Application of the Bayesian Framework
% regression of cosines
load('cosX')
load('cosY')
%The training/validation and test sets are created:
Xtrain = X(1:2:length(X));
Ytrain = Y(1:2:length(Y));
Xtest = X(2:2:length(X));
Ytest = Y(2:2:length(Y));


sig2 = 0.05; gam = 10;
type = 'f'
sigList = [0.01,0.05, 0.1,1];
gammaList = [100];
for gamIni = gammaList
    for sig2Ini=sigList,
        criterion_L1 = bay_lssvm({Xtrain,Ytrain,type,gamIni,sig2Ini},1)
        criterion_L2 = bay_lssvm({Xtrain,Ytrain,type,gamIni,sig2Ini},2)
        criterion_L3 = bay_lssvm({Xtrain,Ytrain,type,gamIni,sig2Ini},3)

        % The model can be optimized with respect to these criteria:
        [~,alpha,b] = bay_optimize({Xtrain,Ytrain,type,gamIni,sig2Ini},1);
        [~,gam] = bay_optimize({Xtrain,Ytrain,type,gamIni,sig2Ini},2);
        [~,sig2] = bay_optimize({Xtrain,Ytrain,type,gam,sig2Ini},3);
        % For regression, the error bars can be computed by: 
        sig2e = bay_errorbar({Xtrain,Ytrain,type,gam,sig2},'figure');
        saveas(gcf,strcat('plots24/regression_gamm_',num2str(gamIni),'_sig_',num2str(sig2Ini),'.jpg'))
        savefig(strcat('plots24/regression_gamm_',num2str(gamIni),'_sig_',num2str(sig2Ini),'.fig'))
        close all
    end 
end
%% CLASIFICATION: 

clear;
load iris;
gam = 5; sig2 = 0.75;
type = 'c'

sigList = [0.01,0.1,0.75,1, 10];
gammaList = [0.01,0.1,1, 5, 10];
for gam = gammaList
    for sig2=sigList,
        bay_modoutClass({X,Y,type,gam,sig2},'figure');
        saveas(gcf,strcat('plots24/classification_gamm_',num2str(gam),'_sig_',num2str(sig2),'.jpg'))
        savefig(strcat('plots24/classification_gamm_',num2str(gam),'_sig_',num2str(sig2),'.fig'))
        close all
    end
end


%% Automatic Relevance Determination
gam = 5
sig2=0.75
X = 10.*rand(100,3)-3;
Y = cos(X(:,1)) + cos(2*(X(:,1))) + 0.3.*randn(100,1);
[selected, ranking] = bay_lssvmARD({X,Y,'class',gam,sig2});
% figure
% plot(X(:,1),Y, 'bo')
% figure
% plot(X(:,2),Y, 'ro')
% figure
% plot(X(:,3),Y, 'go')
labels = {'X1' 'X2' 'X3' 'Y'};
figure
data = [X(:,1) X(:,2) X(:,3) Y];

[h,ax] = plotmatrix(data); 

for i = 1:4                                       % label the plots
  xlabel(ax(4,i), labels{i})
  ylabel(ax(i,1), labels{i})
end
% plot(Y, 'r-o')
% hold on
% plot(X(:,1), 'b-o')
% hold on
% plot(X(:,2), 'g-o')
% hold on
% plot(X(:,3), 'm-o')
% hold on
% plot(X(:,2), Y, 'g')
% hold on
% plot(X(:,3), Y, 'b')

%scatter3(X(:,1),X(:,2),X(:,3),40,Y,'filled')    % draw the scatter plot(X(:,1),X(:,2),X(:,3),Y)  
% xlabel('X1')
% ylabel('X2')
% zlabel('X3')



