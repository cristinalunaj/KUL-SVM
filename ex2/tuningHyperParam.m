% % hyperParameters tuning
% load('cosX')
% load('cosY')
% %The training/validation and test sets are created:
% Xtrain = X(1:2:length(X));
% Ytrain = Y(1:2:length(Y));
% Xtest = X(2:2:length(X));
% Ytest = Y(2:2:length(Y));
% 
% 
% type = 'f';
% kernel = 'RBF_kernel';
% 
% optFun = 'gridsearch';
% globalOptFun = 'csa';
% 
% 
% par1 = ["csa","ds"];
% par2 = ["simplex","gridsearch"];
% 
% parametersList = []
% for times = 1:10
%     for i=1:2
%         for j=1:2
%             close all
%             p1 = char(par1(i))
%             p2 = char(par2(j))
%             tic
%             [gam,sig2,cost_crossval] = tunelssvm({Xtrain,Ytrain,type,[],[],kernel,p1},p2,'crossvalidatelssvm',{10,'mse'})
%             time=toc
% %             cost_crossval = crossvalidate({Xtrain,Ytrain,type,gam,sig2},10);
%             cost_loo = leaveoneout({Xtrain,Ytrain,type,gam,sig2});
%             
%             [alpha,b] = trainlssvm({Xtrain,Ytrain,type,gam,sig2});
%             plotlssvm({Xtrain,Ytrain,type,gam,sig2},{alpha,b});
%             saveas(gcf,strcat('23HyperPar/tuning_RBF_train_gamm_',num2str(gam),'_sig_',num2str(sig2),'.jpg'))
%             savefig(strcat('23HyperPar/tuning_RBF_train_gamm_',num2str(gam),'_sig_',num2str(sig2),'.fig'))
%             close all
%             parametersList = [parametersList; [gam, sig2, cost_crossval,cost_loo, time]];
%         end 
%     end
% end
% 
% ini_index = 1; 
% results_matrix_hyperp_csaSimplex = [mean(parametersList(ini_index:4:end,1)), mean(parametersList(ini_index:4:end,2)), mean(parametersList(ini_index:4:end,3)), mean(parametersList(ini_index:4:end,4)),mean(parametersList(ini_index:4:end,5));
%                                     std(parametersList(ini_index:4:end,1)), std(parametersList(ini_index:4:end,2)), std(parametersList(ini_index:4:end,3)), std(parametersList(ini_index:4:end,4)),std(parametersList(ini_index:4:end,5));
%                                     min(parametersList(ini_index:4:end,1)), min(parametersList(ini_index:4:end,2)), min(parametersList(ini_index:4:end,3)), min(parametersList(ini_index:4:end,4)),min(parametersList(ini_index:4:end,5));
%                                     max(parametersList(ini_index:4:end,1)), max(parametersList(ini_index:4:end,2)), max(parametersList(ini_index:4:end,3)), max(parametersList(ini_index:4:end,4)),max(parametersList(ini_index:4:end,5))];
%     
% 
% ini_index = 2; 
% results_matrix_hyperp_csaGrid = [mean(parametersList(ini_index:4:end,1)), mean(parametersList(ini_index:4:end,2)), mean(parametersList(ini_index:4:end,3)), mean(parametersList(ini_index:4:end,4)),mean(parametersList(ini_index:4:end,5));
%                                     std(parametersList(ini_index:4:end,1)), std(parametersList(ini_index:4:end,2)), std(parametersList(ini_index:4:end,3)), std(parametersList(ini_index:4:end,4)),std(parametersList(ini_index:4:end,5));
%                                     min(parametersList(ini_index:4:end,1)), min(parametersList(ini_index:4:end,2)), min(parametersList(ini_index:4:end,3)), min(parametersList(ini_index:4:end,4)),min(parametersList(ini_index:4:end,5));
%                                     max(parametersList(ini_index:4:end,1)), max(parametersList(ini_index:4:end,2)), max(parametersList(ini_index:4:end,3)), max(parametersList(ini_index:4:end,4)),max(parametersList(ini_index:4:end,5))];
%     
%             
% ini_index = 3; 
% results_matrix_hyperp_dsSimplex = [mean(parametersList(ini_index:4:end,1)), mean(parametersList(ini_index:4:end,2)), mean(parametersList(ini_index:4:end,3)), mean(parametersList(ini_index:4:end,4)),mean(parametersList(ini_index:4:end,5));
%                                     std(parametersList(ini_index:4:end,1)), std(parametersList(ini_index:4:end,2)), std(parametersList(ini_index:4:end,3)), std(parametersList(ini_index:4:end,4)),std(parametersList(ini_index:4:end,5));
%                                     min(parametersList(ini_index:4:end,1)), min(parametersList(ini_index:4:end,2)), min(parametersList(ini_index:4:end,3)), min(parametersList(ini_index:4:end,4)),min(parametersList(ini_index:4:end,5));
%                                     max(parametersList(ini_index:4:end,1)), max(parametersList(ini_index:4:end,2)), max(parametersList(ini_index:4:end,3)), max(parametersList(ini_index:4:end,4)),max(parametersList(ini_index:4:end,5))];
%     
% ini_index = 4; 
% results_matrix_hyperp_dsGrid = [mean(parametersList(ini_index:4:end,1)), mean(parametersList(ini_index:4:end,2)), mean(parametersList(ini_index:4:end,3)), mean(parametersList(ini_index:4:end,4)),mean(parametersList(ini_index:4:end,5));
%                                     std(parametersList(ini_index:4:end,1)), std(parametersList(ini_index:4:end,2)), std(parametersList(ini_index:4:end,3)), std(parametersList(ini_index:4:end,4)),std(parametersList(ini_index:4:end,5));
%                                     min(parametersList(ini_index:4:end,1)), min(parametersList(ini_index:4:end,2)), min(parametersList(ini_index:4:end,3)), min(parametersList(ini_index:4:end,4)),min(parametersList(ini_index:4:end,5));
%                                     max(parametersList(ini_index:4:end,1)), max(parametersList(ini_index:4:end,2)), max(parametersList(ini_index:4:end,3)), max(parametersList(ini_index:4:end,4)),max(parametersList(ini_index:4:end,5))];
%     
%                                 
%             
% save('23HyperPar/csaSimplex', 'results_matrix_hyperp_csaSimplex');
% save('23HyperPar/csaGrid', 'results_matrix_hyperp_csaGrid');            
% save('23HyperPar/dsSimplex', 'results_matrix_hyperp_dsSimplex');                     
% save('23HyperPar/dsGrid', 'results_matrix_hyperp_dsGrid');                    
%             
            
            
            
%% test ds
% hyperParameters tuning
load('cosX')
load('cosY')
%The training/validation and test sets are created:
Xtrain = X(1:2:length(X));
Ytrain = Y(1:2:length(Y));
Xtest = X(2:2:length(X));
Ytest = Y(2:2:length(Y));


type = 'f';
kernel = 'RBF_kernel';

optFun = 'gridsearch';
globalOptFun = 'csa';


par1 = ["ds"];
par2 = ["simplex"];

parametersList = []
for times = 1:1
    for i=1:1
        for j=1:1
            close all
            p1 = char(par1(i))
            p2 = char(par2(j))
            tic
            [gam,sig2,cost_crossval] = tunelssvm({Xtrain,Ytrain,type,[],[],kernel,p1},p2,'crossvalidatelssvm',{10,'mse'})
            time=toc
%             cost_crossval = crossvalidate({Xtrain,Ytrain,type,gam,sig2},10);
            cost_loo = leaveoneout({Xtrain,Ytrain,type,gam,sig2});
            
            [alpha,b] = trainlssvm({Xtrain,Ytrain,type,gam,sig2});
            plotlssvm({Xtrain,Ytrain,type,gam,sig2},{alpha,b});
            saveas(gcf,strcat('23HyperPar/tuning_RBF_train_gamm_',num2str(gam),'_sig_',num2str(sig2),'.jpg'))
            savefig(strcat('23HyperPar/tuning_RBF_train_gamm_',num2str(gam),'_sig_',num2str(sig2),'.fig'))
            close all
            parametersList = [parametersList; [gam, sig2, cost_crossval,cost_loo, time]];
        end 
    end
end
            