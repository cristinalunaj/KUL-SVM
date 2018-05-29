clc;clear; close all
load ('santafe.mat')
% [Z,settingMapMINMAX] = mapminmax(Z')
% [Z,settingMapSTD] = mapstd(Z)
% Z=Z';
gam = 10;
sig2 = 50;
type = 'f'
kernel = 'RBF_kernel'
mse_values = []
mse_values_Aux = []
parametersList = []
mse_10times = []
mse_MEAN_values = []
trainingSamples = 500
% for order = 10:10:100
  
%    if order==0
%        order=1
%    end
   order = 50
    W = windowize(Z,1:order+1);
    
    
    X = W(:,1:order);
    Y = W(:,end);
    
%     X_train = W(1:trainingSamples,1:order)
%     Y_train = W(1:trainingSamples,end)
%     X_val = W(trainingSamples+1:end,1:order)
%     Y_val = W(trainingSamples+1:end,end)
    
    %performance = simlssvm({W(:,1:order),W(:,end),'f',10,50,'RBF_kernel'},X_train, Y_train, X_val, Y_val);
    
    
    [gam,sig2, cost] = tunelssvm({X,Y,'f',[],[],'RBF_kernel'},'gridsearch','crossvalidatelssvm',{10, 'mae'});
    
    order = bay_lssvmARD({X,Y,'f',gam,sig2,'RBF_kernel'});
    order = order(1)
    W = windowize(Z,1:order+1);
    
    
    X = W(:,1:order);
    Y = W(:,end);
    
    
    [alpha,b] = trainlssvm({X,Y,'f',gam,sig2,'RBF_kernel'});
    
    Xs = Z(end-order+1:end,1);
    prediction = predict({X,Y,'f',gam,sig2,'RBF_kernel'},Xs, 200);
    mae_test = mae(prediction-Ztest);
    plot([prediction Ztest])
    legend('pred', 'real')
    title(strcat('order: ',num2str(order),',gamma: ',num2str(gam), ',sigma2:', num2str(sig2),'MAE:', num2str(mae_test)))
    xlabel('Samples')
    grid on
    
    saveas(gcf,strcat('plot26_santaFE/BAYEStest','_order_',num2str(order),'_gamm_',num2str(gam),'_sig_',num2str(sig2),'.jpg'))
    savefig(strcat('plot26_santaFE/BAYEStest','_order_',num2str(order),'_gamm_',num2str(gam),'_sig_',num2str(sig2),'.fig'))
%     
%     close all
%     parametersList = [parametersList; [gam,sig2,order, cost,mae_test]]
% end
    %VALIDATION
%     Xval = Z(end-order+1:end,1);
%     prediction = predict({X,Y,'f',gam,sig2,'RBF_kernel'},Xs, 200);
%     mae_test = mae(prediction-Ztest);
%     plot([prediction Ztest]);
%     legend('pred', 'real');
    
%     order = bay_lssvmARD({X,Y,'f',gam,sig2,'RBF_kernel'});
%     X = W(:,1:order);
%     Y = W(:,end);
%     [alpha,b] = trainlssvm({X,Y,'f',gam,sig2,'RBF_kernel'});
%     Xs = Z(end-order+1:end,1);
%     prediction = predict({X,Y,'f',gam,sig2,'RBF_kernel'},Xs, 200);
%     figure
%     plot([prediction Ztest]);
 
    
    
    
    
%     
%     model = {X,Y,type,gam,sig2,kernel};
%     [alpha,b] = trainlssvm(model);
%    
%     xtraining = Z_train(1:order,1);
%     prediction_training =predict({X,Y,type,gam,sig2,kernel},xtraining,(size(X,1)));
%     MSE_train = mse(Y-prediction_training)
%     R_train = corrcoef(Y,prediction_training)
%     figure
%     hold on
%     plot(Y, 'k')
%     hold on
%     plot(prediction_training, 'b')
%     legend('Train', 'predictions')
%     title (strcat('Santa Fe laser data training - Order:', num2str(order),', Gamma: ',num2str(gam),', Sigma: ',num2str(sig2)))
%     grid on
%     saveas(gcf,strcat('plot26_santaFE/train','_order_',num2str(order),'_gamm_',num2str(gam),'_sig_',num2str(sig2),'.jpg'))
%     savefig(strcat('plot26_santaFE/train','_order_',num2str(order),'_gamm_',num2str(gam),'_sig_',num2str(sig2),'.fig'))
%     close all
%     % Starting point of the prediction: 
%     Xval = Z_train(end-order+1:end,1);
%     % This is the last point of the training set. The test set Ztest presents 
%     % data points after this point, which we will try to predict.
%     %** where nb indicates how many time points we want to predict
%     nb = 200;
%     prediction_val = predict(model,Xval,nb);
%     MSE_val = mse(Z_val-prediction_val)
%     R_val = corrcoef(Z_val,prediction_val)
%     % figure
%     % hold on
%     % plot(Y, 'k')
%     % hold on
%     % plot(prediction_training, 'r')
%     % legend('Ztrain', 'predictions')
%     Xtest = Z_train(end-order+1:end,1);
%     prediction_test = predict(model,Xtest,nb);
%     MSE_test = mse(Ztest-prediction_test)
%     R_test = corrcoef(Ztest,prediction_test)
%     % performance of the predictor
%     figure
%     hold on
%     plot(Ztest, 'k')
%     hold on
%     plot(prediction_test, 'g')
%     legend('Ztest', 'predictions')
%     xlabel('Samples')
%     grid on
%     title('Comparison between predictions and real values')
%     saveas(gcf,strcat('plot26_santaFE/test','_order_',num2str(order),'_gamm_',num2str(gam),'_sig_',num2str(sig2),'.jpg'))
%     savefig(strcat('plot26_santaFE/test','_order_',num2str(order),'_gamm_',num2str(gam),'_sig_',num2str(sig2),'.fig'))
%     parametersList = [parametersList; [gam,sig2,order,performance,MSE_train,MSE_val, MSE_test]]
%     mse_10times = [mse_10times; [MSE_train,performance, MSE_val,MSE_test, R_train(1,2),R_val(1,2),R_test(1,2)]];


% mse_10times = mse_10times';
% figure
% x=0:5:60
% plot(x, mse_10times(1,:), 'b-o')
% hold on
% plot(x, mse_10times(2,:), 'r-o')
% hold on
% plot(x, mse_10times(3,:), 'm-o')
% hold on
% plot(x, mse_10times(4,:), 'g-o')
% legend('training MSE', 'xvalidation MSE','Normal Validation MSE', 'test MSE')
% xlabel('Order')
% ylabel('MSE')
% title(strcat('MSE vs Order for gamma=',num2str(gam),'& sig2=' ,num2str(sig2)))
% savefig(strcat('plot26_2/timeSeriesLogmap_MSE_order_gam',num2str(gam),'_sig',num2str(sig2),'.fig'))
% saveas(gcf,strcat('plot26_2/timeSeriesLogmap_MSE_order_gam',num2str(gam),'_sig',num2str(sig2),'.jpg'))
% save(strcat('plot26_2/logmapParametersList_gamgam',num2str(gam),'_sig',num2str(sig2)), 'parametersList')





%% new validation method
clc;clear; close all
load ('santafe.mat')

%Z_val = Z(end-200+1:end);
Z_train = Z(1:end);
nFolds = 5
type = 'f'
kernel = 'RBF_kernel'
order = 50;
result_models = []
x=10:10:100

for order=x
    W = windowize(Z,1:order+1);
    X = W(:,1:order);
    Y = W(:,end);
    
   % [gam,sig2, cost] = tunelssvm({X,Y,'f',[],[],'RBF_kernel'},'gridsearch','crossvalidatelssvm',{10, 'mae'});
   gam = 1863.3633
   sig2 = 36.4599
   %validation
    nb = 200;
    results_validation = []
    maxFolds = 4
    W = windowize(Z,1:order+1);
    [rows,cols] = size(W);
    X = W(:,1:order);
    Y = W(:,end);
    samplesPerPack = floor(rows/(maxFolds+1)); 
    for nFold = 1:maxFolds
        
        
        trainingSamples = samplesPerPack*nFold;

        X_train = W(1:trainingSamples,1:order)
        Y_train = W(1:trainingSamples,end)
        X_val = W(trainingSamples+1:trainingSamples+samplesPerPack,1:order)
        Y_val = W(trainingSamples+1:trainingSamples+samplesPerPack,end)


        [alpha,b] = trainlssvm({X_train,Y_train,'f',gam,sig2,'RBF_kernel'});

        model = {X_train,Y_train,type,gam,sig2,kernel};
        [alpha,b] = trainlssvm(model);

        Xval = Z(trainingSamples-order+1:trainingSamples+order,1);
        prediction_val = predict(model,Xval,samplesPerPack);
        MAE_val = mae(Y_val-prediction_val)
        results_validation= [results_validation;MAE_val]
    end
%     while(std(results_validation)>10))
%         [val,idx] = max(results_validation);
%         results_validation(idx) = [];
%     end
    results_validation_stage = mean(results_validation)
    Xs = Z(end-order+1:end,1);
    [alpha,b] = trainlssvm({X,Y,'f',gam,sig2,'RBF_kernel'});
    model = {X,Y,type,gam,sig2,kernel};
    prediction_test = predict(model,Xs,nb);
    
    MAE_test = mae(Ztest-prediction_test)
    plot([prediction_test Ztest])
    legend('pred', 'real')
    title(strcat('order: ',num2str(order),',gamma: ',num2str(gam), ',sigma2:', num2str(sig2),'MAE:', num2str(MAE_test)))
    xlabel('Samples')
    grid on
    saveas(gcf,strcat('plot26_santaFE/2test','_order_',num2str(order),'_gamm_',num2str(gam),'_sig_',num2str(sig2),'.jpg'))
    savefig(strcat('plot26_santaFE/2test','_order_',num2str(order),'_gamm_',num2str(gam),'_sig_',num2str(sig2),'.fig'))
    close all
    result_models = [result_models;[gam,sig2,results_validation_stage,MAE_test]]
end


result_models = result_models';

plot(x, result_models(3,:), 'r-o')
hold on
plot(x, result_models(4,:), 'g-o')
legend('validation MAE', 'test MAE')
xlabel('Order')
ylabel('MAE')
% title(strcat('MSE vs Order for gamma=',num2str(gam),'& sig2=' ,num2str(sig2)))






%% xvalidation - normal
clc;clear; close all
load ('santafe.mat')
type = 'f'
kernel = 'RBF_kernel'



par1 = ["csa"];
par2 = ["simplex","gridsearch"];
nb = 200;
mse_values = []
mse_values_Aux = []
parametersList = []
mse_10times = []
mse_MEAN_values = []
for order = 10:10:100
    for times = 1:10
        for i=1:1
            for j=1:1

                W = windowize(Z,1:order+1);
                X = W(:,1:order);
                Y = W(:,end);


                p1 = char(par1(i))
                p2 = char(par2(j))
                model = {X,Y,type,[],[],'RBF_kernel',p1};
                [gam,sig2,cost] = tunelssvm(model,p2,'crossvalidatelssvm',{10,'mse'});
                [alpha,b] = trainlssvm({X,Y,'f',gam,sig2,kernel});


                %Prediction - take this numbers as inputs for the next prediction
                Xs = Z(end-order+1:end,1);
                prediction = predict({X,Y,type,gam,sig2,kernel},Xs,nb);
                MSE_test = mse(Ztest-prediction)
                figure
                hold on
                plot(Ztest, 'k')
                hold on
                plot(prediction, 'r')
                legend('Ztest', 'predictions')
                title (strcat('Logmap - Order:', num2str(order),', Gamma: ',num2str(gam),', Sigma: ',num2str(sig2), 'MSE_TEST:', num2str(MSE_test)))
                grid on
                saveas(gcf,strcat('plot26/SANTAFE_test','_order_',num2str(order),'_gamm_',num2str(gam),'_sig_',num2str(sig2),'.jpg'))
                savefig(strcat('plot26/SANTAFE_test','_order_',num2str(order),'_gamm_',num2str(gam),'_sig_',num2str(sig2),'.fig'))
                close all
                parametersList = [parametersList; [gam,sig2,order,cost, MSE_test]]
                mse_10times = [mse_10times; [cost,MSE_test]];
            end 
        end
    end
    meanValues = mean(mse_10times);
    mse_MEAN_values = [mse_MEAN_values; meanValues];
    mse_10times = []
end


mse_MEAN_values = mse_MEAN_values';
x=10:10:100
plot(x, mse_MEAN_values(1,:), 'r-o')
hold on
plot(x, mse_MEAN_values(2,:), 'b-o')
legend('training MSE', 'test MSE')
xlabel('Order')
ylabel('MSE')
title('MSE vs Order')
% savefig(strcat('plot26/SANTAFE_MSE_order_2ndAttempt.fig'))
% save('plot26/SANTAFEParametersList2ndtAttempt', 'parametersList')
% 


