clc;clear;
load iris

% train LS-SVM classifier with linear kernel 
%
type='c'; 
gammaList = [0.1,1,10]
for gam = gammaList
    disp('Linear kernel'),
    gam
    [alpha,b] = trainlssvm({X,Y,type,gam,[],'lin_kernel'});
    figure; plotlssvm({X,Y,type,gam,[],'lin_kernel'},{alpha,b});
    saveas(gcf,strcat('images/lineal/linear_',num2str(gam),'.jpg'))
    savefig(strcat('images/lineal/linear_',num2str(gam),'.fig'))
    [Yht, Zt] = simlssvm({X,Y,type,gam,[],'lin_kernel'}, {alpha,b}, Xt);
    err = sum(Yht~=Yt); 
    fprintf('\n on test: #misclass = %d, error rate = %.2f%%\n', err, err/length(Yt)*100)
end

%% polynomial kernel
clc;clear;close all

load iris
type='c'; 
gam = 1; 
disp('Polynomial kernel'),
kernel = 'poly_kernel'
t = 0
degree = 3
gammaList = [0.1,1,10]
for gam = gammaList
    disp('Linear kernel'),
    [alpha,b] = trainlssvm({X,Y,type,gam,[t;degree],kernel});

    figure; plotlssvm({X,Y,type,gam,[t;degree],kernel},{alpha,b});
%     saveas(gcf,strcat('images/poly/poly_',num2str(degree),'_',num2str(gam),'.jpg'))
%     savefig(strcat('images/poly/poly_',num2str(degree),'_',num2str(gam),'.fig'))
    [Yht, Zt] = simlssvm({X,Y,type,gam,[t;degree],kernel}, {alpha,b}, Xt);
    err = sum(Yht~=Yt); 
    fprintf('\n on test: #misclass = %d, error rate = %.2f%%\n', err, err/length(Yt)*100)
end


%%  use RBF kernel
%
% 
% % tune the sig2 while fix gam
% 
% disp('RBF kernel')
% gam = 1; sig2list=[0.01, 0.05,0.1,0.5, 1, 5, 10,15, 25];
% 
% errlist=[];
% 
% for sig2=sig2list,
%     disp(['gam : ', num2str(gam), '   sig2 : ', num2str(sig2)]),
%     [alpha,b] = trainlssvm({X,Y,type,gam,sig2,'RBF_kernel'});
% 
%     % Plot the decision boundary of a 2-d LS-SVM classifier
%     plotlssvm({X,Y,type,gam,sig2,'RBF_kernel'},{alpha,b});
% 
%     % Obtain the output of the trained classifier
%     [Yht, Zt] = simlssvm({X,Y,type,gam,sig2,'RBF_kernel'}, {alpha,b}, Xt);
%     err = sum(Yht~=Yt); errlist=[errlist; err];
%     fprintf('\n on test: #misclass = %d, error rate = %.2f%% \n', err, err/length(Yt)*100)
%     disp('Press any key to continue...'), pause,         
% end
% 
% 
% 
% 
% %
% % make a plot of the misclassification rate wrt. sig2
% %
% figure;
% plot(log(sig2list), errlist, '*-'), 
% xlabel('log(sig2)'), ylabel('number of misclass'),

%%

disp('RBF kernel')
gammaList =[0.01, 0.05,0.1,0.5, 1, 5, 10,15, 25, 50, 100, 1000];

errlist=[];
sig2 = 1;
type = 'c'
for gam=gammaList,
    disp(['gam : ', num2str(gam), '   sig2 : ', num2str(sig2)]),
    [alpha,b] = trainlssvm({X,Y,type,gam,sig2,'RBF_kernel'});

    % Plot the decision boundary of a 2-d LS-SVM classifier
    plotlssvm({X,Y,type,gam,sig2,'RBF_kernel'},{alpha,b});

    % Obtain the output of the trained classifier
    [Yht, Zt] = simlssvm({X,Y,type,gam,sig2,'RBF_kernel'}, {alpha,b}, Xt);
    err = sum(Yht~=Yt); errlist=[errlist; err];
    fprintf('\n on test: #misclass = %d, error rate = %.2f%% \n', err, err/length(Yt)*100)
    disp('Press any key to continue...'), pause,         
end

figure;
plot(log(gammaList), errlist, '*-'), 
xlabel('log(gamma)'), ylabel('number of misclass'),
