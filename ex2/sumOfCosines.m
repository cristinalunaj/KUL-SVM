% 2.2 A Simple Example: Sum of Cosines
% X = (-10:0.1:10)';
% Y = cos(X) + cos(2*X) + 0.1.*randn(length(X),1);
% save('cosX', 'X');
% save('cosY', 'Y');
load('cosX')
load('cosY')
%The training/validation and test sets are created:
Xtrain = X(1:2:length(X));
Ytrain = Y(1:2:length(Y));
Xtest = X(2:2:length(X));
Ytest = Y(2:2:length(Y));

gam = 100;
sig2 = 1.0;
[alpha,b] = trainlssvm({Xtrain,Ytrain,'f',gam,sig2,'RBF_kernel'});

plotlssvm({Xtrain,Ytrain,'f',gam,sig2,'RBF_kernel'},{alpha,b});
saveas(gcf,strcat('plots22/RBF_train_gamm_',num2str(gam),'_sig_',num2str(sig2),'.jpg'))
savefig(strcat('plots22/RBF_train_gamm_',num2str(gam),'_sig_',num2str(sig2),'.fig'))
close all
YtestEst = simlssvm({Xtrain,Ytrain,'f',gam,sig2,'RBF_kernel'},{alpha,b},Xtest);
MSE_test = mse(YtestEst-Ytest);
plot(Xtest,Ytest,'.');
hold on;
plot(Xtest,YtestEst,'r-+');
title(strcat('gamma:',num2str(gam),', sigma2:',num2str(sig2), ',MSE:', num2str(MSE_test)))
legend('Ytest','YtestEst');
saveas(gcf,strcat('plots22/RBF_test_gamm_',num2str(gam),'_sig_',num2str(sig2),'.jpg'))
savefig(strcat('plots22/RBF_test_gamm_',num2str(gam),'_sig_',num2str(sig2),'.fig'))
close all



%% 
sigList = [0.001,0.01,0.1];
gammaList = [100,1000, 10000, 50000,100000];
parametersErrorList = [];
type = 'f'
for gam = gammaList
    for sig2=sigList,
        close all
        disp(['gam : ', num2str(gam), '   sig2 : ', num2str(sig2)]),
        [alpha,b] = trainlssvm({Xtrain,Ytrain,'f',gam,sig2,'RBF_kernel'});

        plotlssvm({Xtrain,Ytrain,'f',gam,sig2,'RBF_kernel'},{alpha,b});
        saveas(gcf,strcat('plots22/RBF_train_gamm_',num2str(gam),'_sig_',num2str(sig2),'.jpg'))
        savefig(strcat('plots22/RBF_train_gamm_',num2str(gam),'_sig_',num2str(sig2),'.fig'))
        close all
        YtestEst = simlssvm({Xtrain,Ytrain,'f',gam,sig2,'RBF_kernel'},{alpha,b},Xtest);
        MSE_test = mse(YtestEst-Ytest);
        plot(Xtest,Ytest,'.');
        hold on;
        plot(Xtest,YtestEst,'r-+');
        title(strcat('gamma:',num2str(gam),', sigma2:',num2str(sig2), ',MSE:', num2str(MSE_test)))
        legend('Ytest','YtestEst');
        saveas(gcf,strcat('plots22/RBF_test_gamm_',num2str(gam),'_sig_',num2str(sig2),'.jpg'))
        savefig(strcat('plots22/RBF_test_gamm_',num2str(gam),'_sig_',num2str(sig2),'.fig'))
        close all
        parametersErrorList = [parametersErrorList; [gam,sig2,MSE_test]];
        
    end
end

