% % %% 2.6 HOMEWORK PROBLEMS
% load ('logmap.mat')
% type = 'f'
% kernel = 'RBF_kernel'
% order = 10;
% W = windowize(Z,1:order+1);
% X = W(:,1:order);
% Y = W(:,end);
% %Now a model can be built using these data points:
% gam = 10;
% sig2 = 10;
% model = {X,Y,type,gam,sig2,kernel};
% [alpha,b] = trainlssvm(model);
% 
% %Starting point of the prediction: 
% Xs = Z(end-order+1:end,1);
% %This is the last point of the training set. The test set Ztest presents 
% %data points after this point, which we will try to predict.
% % ** where nb indicates how many time points we want to predict
% nb = 50;
% prediction = predict(model,Xs,nb);
% MSE_test = mse(Ztest-prediction)
% %performance of the predictor
% figure
% hold on
% plot(Ztest, 'k')
% hold on
% plot(prediction, 'r')
% legend('Ztest', 'predictions')

%% Bayess 
% 
% clc;clear; close all
% load ('logmap.mat')
% type = 'f'
% kernel = 'RBF_kernel'
% order = 10;
% W = windowize(Z,1:order+1);
% X = W(:,1:order);
% Y = W(:,end);
% nb=50
% sig2Ini = 10
% gamIni = 10
% criterion_L1 = bay_lssvm({X,Y,type,gamIni,sig2Ini},1)
% criterion_L2 = untitledbay_lssvm({X,Y,type,gamIni,sig2Ini},2)
% criterion_L3 = bay_lssvm({X,Y,type,gamIni,sig2Ini},3)
% 
% % The model can be optimized with respect to these criteria:
% [~,alpha,b] = bay_optimize({X,Y,type,gamIni,sig2Ini},1);
% [~,gam] = bay_optimize({X,Y,type,gamIni,sig2Ini},2);
% [~,sig2] = bay_optimize({X,Y,type,gam,sig2Ini},3);
% % For regression, the error bars can be computed by: 
% sig2e = bay_errorbar({X,Y,type,gam,sig2},'figure');
% 
% % Xs = Z(end-order+1:end,1);
% 
% % prediction = predict({X,Y,type,gam,sig2,kernel},Xs,nb);
% % sig2e = bay_errorbar({X,Y,type,gam,sig2},'figure');


%% xvalidation same model

clc;clear; close all
load ('logmap.mat')
type = 'f'
kernel = 'RBF_kernel'

gam = 6.022932245103818e+04;
sig2=44.433340237958916;


nb = 50;
mse_values = []
mse_values_Aux = []
parametersList = []
mse_10times = []
mse_MEAN_values = []
zmean = mean(Z);
zstd = std(Z);
for order = 0:2:60
  
   if order==0
       order=1
   end
%     Z = Z-zmean;
%     Ztest = Ztest-zmean;
%     Ztest_withoutnoise = Ztest_withoutnoise-zmean;
    W = windowize(Z,1:order+1);
    
    X = W(:,1:order);
    Y = W(:,end);
    

    performance = crossvalidate({X,Y,'f',gam,sig2,'RBF_kernel'},10,'mse');
    [alpha,b] = trainlssvm({X,Y,'f',gam,sig2,kernel});
    %PREDICTION TRAINING
    xtraining = Z(1:order,1);
    prediction_training =predict({X,Y,type,gam,sig2,kernel},xtraining,(size(X,1)));
    MSE_train = mse(Y-prediction_training)
    %MSE_train_normal = mse(Y-prediction_training)
    R_train = corrcoef(Y,prediction_training)
    figure
    hold on
    plot(Y, 'k')
    hold on
    plot(prediction_training, 'b')
    legend('Ztrain', 'predictions')
    title (strcat('Logmap training - Order:', num2str(order),', Gamma: ',num2str(gam),', Sigma: ',num2str(sig2)))
    grid on
    saveas(gcf,strcat('plot26_2/timeSeriesLogmap_train','_order_',num2str(order),'_gamm_',num2str(gam),'_sig_',num2str(sig2),'.jpg'))
    savefig(strcat('plot26_2/timeSeriesLogmap_train','_order_',num2str(order),'_gamm_',num2str(gam),'_sig_',num2str(sig2),'.fig'))
    close all
    
    %Prediction - take this numbers as inputs for the next prediction
    Xs = Z(end-order+1:end,1);
    prediction = predict({X,Y,type,gam,sig2,kernel},Xs,nb);
    %Ztest_withoutnoise=Ztest_withoutnoise-zmean;
    
    MSE_test = mse(Ztest-prediction)
    MSE_test_noNoise = mse(Ztest_withoutnoise-prediction)
    R_test = corrcoef(Ztest,prediction)
    R_test_withouthnoise = corrcoef(Ztest_withoutnoise,prediction)
    figure
    hold on
    plot(Ztest, 'k')
    hold on
    plot(Ztest_withoutnoise, 'g')
    hold on
    plot(prediction, 'r')
    legend('Z test with noise', 'Ztest without noise', 'predictions')
    title (strcat('Logmap test - Order:', num2str(order),', Gamma: ',num2str(gam),', Sigma: ',num2str(sig2)))
    grid on
    saveas(gcf,strcat('plot26_2/timeSeriesLogmap_test','_order_',num2str(order),'_gamm_',num2str(gam),'_sig_',num2str(sig2),'.jpg'))
    savefig(strcat('plot26_2/timeSeriesLogmap_test','_order_',num2str(order),'_gamm_',num2str(gam),'_sig_',num2str(sig2),'.fig'))
   close all
    parametersList = [parametersList; [gam,sig2,order,performance, MSE_test, MSE_test_noNoise]]
    mse_10times = [mse_10times; [performance,MSE_test, MSE_test_noNoise, R_train(1,2),R_test(1,2),R_test_withouthnoise(1,2) ]];
end


mse_10times = mse_10times';
x=0:2:60
plot(x, mse_10times(1,:), 'b-o')
hold on
plot(x, mse_10times(2,:), 'r-o')
hold on
plot(x, mse_10times(3,:), 'g-o')
legend('xvalidation MSE', 'test MSE', 'testMSE without noise')
xlabel('Order')
ylabel('MSE')
title(strcat('MSE vs Order for gamma=',num2str(gam),'& sig2=' ,num2str(sig2)))
savefig(strcat('plot26_2/timeSeriesLogmap_MSE_order_gam',num2str(gam),'_sig',num2str(sig2),'.fig'))
saveas(gcf,strcat('plot26_2/timeSeriesLogmap_MSE_order_gam',num2str(gam),'_sig',num2str(sig2),'.jpg'))
save(strcat('plot26_2/logmapParametersList_gamgam',num2str(gam),'_sig',num2str(sig2)), 'parametersList')





%% xvalidation - normal
clc;clear; close all
load ('logmap.mat')
type = 'f'
kernel = 'RBF_kernel'

par1 = ["csa"];
par2 = ["simplex","gridsearch"];
nb = 50;
mse_values = []
mse_values_Aux = []
parametersList = []
mse_10times = []
mse_MEAN_values = []
for order = 1:5:60
    for times = 1:10
        for i=1:1
            for j=1:1
                
               if order==0
                   order=1
               elseif order==20
                   order = 21    
               end
                W = windowize(Z,1:order+1);
                X = W(:,1:order);
                Y = W(:,end);
                
                p1 = char(par1(i))
                p2 = char(par2(j))
                model = {X,Y,type,[],[],'RBF_kernel',p1};
                [gam,sig2,cost] = tunelssvm(model,p2,'crossvalidatelssvm',{10,'mse'});
                [alpha,b] = trainlssvm({X,Y,'f',gam,sig2,kernel});
                
                %PREDICTION TRAINING
                xtraining = Z(1:order,1);
                prediction_training =predict({X,Y,type,gam,sig2,kernel},xtraining,(size(X,1)));
                MSE_train = mse(Y-prediction_training)
                MSE_train_normal = mse(Y-(-1)*prediction_training)
                R_train = corrcoef(Y,prediction_training)
                figure
                hold on
                plot(Y, 'k')
                hold on
                plot(prediction_training, 'b')
                legend('Ztrain', 'predictions')
                title (strcat('Logmap training - Order:', num2str(order),', Gamma: ',num2str(gam),', Sigma: ',num2str(sig2)))
                grid on
                saveas(gcf,strcat('plot26_2/timeSeriesLogmap_train','_order_',num2str(order),'_gamm_',num2str(gam),'_sig_',num2str(sig2),'.jpg'))
                savefig(strcat('plot26_2/timeSeriesLogmap_train','_order_',num2str(order),'_gamm_',num2str(gam),'_sig_',num2str(sig2),'.fig'))
                close all
                
                
                
                %Prediction - take this numbers as inputs for the next prediction
                Xs = Z(end-order+1:end,1);
                prediction = predict({X,Y,type,gam,sig2,kernel},Xs,nb);
                MSE_test = mse(Ztest-prediction)
                MSE_test_noNoise = mse(Ztest_withoutnoise-prediction)
                R_test = corrcoef(Ztest,prediction)
                R_test_withouthnoise = corrcoef(Ztest_withoutnoise,prediction)
                figure
                hold on
                plot(Ztest, 'k')
                hold on
                plot(prediction, 'r')
                plot(Ztest_withoutnoise, 'g')
                legend('Ztest', 'predictions','Ztest without noise')
                title (strcat('Logmap - Order:', num2str(order),', Gamma: ',num2str(gam),', Sigma: ',num2str(sig2)))
                grid on
                saveas(gcf,strcat('plot26_2/timeSeriesLogmap_test','_order_',num2str(order),'_gamm_',num2str(gam),'_sig_',num2str(sig2),'.jpg'))
                savefig(strcat('plot26_2/timeSeriesLogmap_test','_order_',num2str(order),'_gamm_',num2str(gam),'_sig_',num2str(sig2),'.fig'))
                close all
                parametersList = [parametersList; [gam,sig2,order,cost, MSE_test]]
                mse_10times = [mse_10times; [cost,MSE_test, MSE_test_noNoise, R_train(1,2),R_test(1,2),R_test_withouthnoise(1,2) ]];
            end 
        end
    end
    meanValues = mean(mse_10times);
    mse_MEAN_values = [mse_MEAN_values; meanValues];
    mse_10times = []
end


mse_MEAN_values = mse_MEAN_values';
x=1:5:60
plot(x, mse_MEAN_values(1,:), 'b-o')
hold on
plot(x, mse_MEAN_values(2,:), 'r-o')
hold on
plot(x, mse_MEAN_values(3,:), 'g-o')
legend('training MSE', 'test MSE','test MSE without noise')
xlabel('Order')
ylabel('MSE')
title('MSE vs Order')
savefig(strcat('plot26_2/TUNEtimeSeriesLogmap_MSE_order_1STAttempt.fig'))
saveas(gcf,strcat('plot26_2/TUNEtimeSeriesLogmap_MSE_order_1STAttempt.jpg'))
save('plot26_2/TUNElogmapParametersList2ndtAttempt', 'parametersList')

%% xvalidation robust
clc;clear; close all
load ('logmap.mat')
type = 'f'
kernel = 'RBF_kernel'
wFun = 'whuber'
costFun = 'rcrossvalidatelssvm'
mse_values = []
mse_values_Aux = []
parametersList = []
mse_10times = []
mse_MEAN_values = []
 nb=50;
for order = 0:5:60
    for times = 1:10
        for i=1:1
            for j=1:1
                if order==0
                   order=1
               end
                W = windowize(Z,1:order+1);
                X = W(:,1:order);
                Y = W(:,end);


                model = initlssvm(X,Y,type,[],[],'RBF_kernel');
                costFun = 'rcrossvalidatelssvm';
                model = tunelssvm(model,'simplex',costFun,{10,'mae'},wFun);
                
                model = robustlssvm(model)
                gam = model.gam(end)
                sig2 = model.kernel_pars
                %PREDICTION TRAINING
                xtraining = Z(1:order,1);
                prediction_training =predict(model,xtraining,(size(X,1)));
                MSE_train = mse(Y-prediction_training)
                R_train = corrcoef(Y,prediction_training)
                figure
                hold on
                plot(Y, 'k')
                hold on
                plot(prediction_training, 'b')
                legend('Ztrain', 'predictions')
                title (strcat('Logmap training - Order:', num2str(order),', Gamma: ',num2str(gam),', Sigma: ',num2str(sig2)))
                grid on
                saveas(gcf,strcat('plot26_2/ROBUSTtimeSeriesLogmap_train','_order_',num2str(order),'_gamm_',num2str(gam),'_sig_',num2str(sig2),'.jpg'))
                savefig(strcat('plot26_2/ROBUSTtimeSeriesLogmap_train','_order_',num2str(order),'_gamm_',num2str(gam),'_sig_',num2str(sig2),'.fig'))
                close all
                
                
                %TEST PREDICTIONS
                Xs = Z(end-order+1:end,1);
                prediction = predict(model,Xs,nb);
                MSE_test = mse(Ztest-prediction)
                MSE_test_noNoise = mse(Ztest_withoutnoise-prediction)
                R_test = corrcoef(Ztest,prediction)
                R_test_withouthnoise = corrcoef(Ztest_withoutnoise,prediction)
                figure
                hold on
                plot(Ztest, 'k')
                hold on
                plot(prediction, 'r')
                plot(Ztest_withoutnoise, 'g')
                legend('Ztest', 'predictions','Ztest without noise')
                title (strcat('Logmap - Order:', num2str(order),', Gamma: ',num2str(gam),', Sigma: ',num2str(sig2)))
                grid on
                
                saveas(gcf,strcat('plot26_2/Robust_timeSeriesLogmap_test','_order_',num2str(order),'_gamm_',num2str(gam),'_sig_',num2str(sig2),'.jpg'))
                savefig(strcat('plot26_2/Robust_timeSeriesLogmap_test','_order_',num2str(order),'_gamm_',num2str(gam),'_sig_',num2str(sig2),'.fig'))
                
                
                close all
                parametersList = [parametersList; [model.gam(end),model.kernel_pars,order, MSE_test]]
                mse_10times = [mse_10times; [MSE_train,MSE_test, MSE_test_noNoise, R_train(1,2),R_test(1,2),R_test_withouthnoise(1,2) ]];
            end
        end
    end
    meanValues = mean(mse_10times);
    mse_MEAN_values = [mse_MEAN_values; meanValues];
    mse_10times = []
end


mse_MEAN_values = mse_MEAN_values';
x=0:5:60
plot(x, mse_MEAN_values(1,:), 'b-o')
hold on
plot(x, mse_MEAN_values(2,:), 'r-o')
hold on
plot(x, mse_MEAN_values(3,:), 'g-o')
legend('training MSE', 'test MSE','test MSE without noise')
xlabel('Order')
ylabel('MSE')
title('MSE vs Order')
savefig(strcat('plot26/timeSeriesLogmap_MSE_order_2ndAttempt.fig'))
save('plot26/logmapParametersList2ndtAttempt', 'parametersList')
