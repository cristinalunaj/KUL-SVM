%% 2.5 Robust Regression

% No robust version
X = (-10:0.2:10)';
Y = cos(X) + cos(2*X) + 0.1.*rand(size(X));

type = 'f'
gam = 100; sig2 = 0.1;

cost_crossval = crossvalidate({X,Y,type,gam,sig2},10, 'mae');
[alpha,b] = trainlssvm({X,Y,type,gam,sig2});
plotlssvm({X,Y,type,gam,sig2},{alpha,b});
saveas(gcf,strcat('plot25/noRob_withOutOUT_reg_gamm_',num2str(gam),'_sig_',num2str(sig2),'.jpg'))
savefig(strcat('plot25/noRob_withOutOUT_reg_gamm_',num2str(gam),'_sig_',num2str(sig2),'.fig'))
close all
%outliers
out = [15 17 19];
Y(out) = 0.7+0.3*rand(size(out));
out = [41 44 46];
Y(out) = 1.5+0.2*rand(size(out));
save('X25', 'X')
save('Y25','Y')

cost_crossval = crossvalidate({X,Y,type,gam,sig2},10, 'mae');
[alpha,b] = trainlssvm({X,Y,type,gam,sig2});
plotlssvm({X,Y,type,gam,sig2},{alpha,b});
saveas(gcf,strcat('plot25/noRob_with_reg_gamm_',num2str(gam),'_sig_',num2str(sig2),'.jpg'))
savefig(strcat('plot25/noRob_with_reg_gamm_',num2str(gam),'_sig_',num2str(sig2),'.fig'))
close all
functionList = ["whuber", "wlogistic","wmyriad", "whampel"]
results = []


% Robust versiom
for func=functionList
    model = initlssvm(X,Y,type,[],[],'RBF_kernel');
    costFun = 'rcrossvalidatelssvm';
    wFun = char(func)
    model = tunelssvm(model,'simplex',costFun,{10,'mae'},wFun);
    model = robustlssvm(model);
    plotlssvm(model);
    results = [results;model.costCV];
    saveas(gcf,strcat('plot25/Rob_reg_func_',func,'_gamm_',num2str(gam),'_sig_',num2str(sig2),'.jpg'))
    savefig(strcat('plot25/Rob_reg_func_',func,'_gamm_',num2str(gam),'_sig_',num2str(sig2),'.fig'))
    close all
end





