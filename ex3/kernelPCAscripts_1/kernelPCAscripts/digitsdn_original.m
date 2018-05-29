%
% Experiments on the handwriting data set on kPCA for reconstruction and denoising
%
clc;clear;close all;
load digits; clear size
[N, dim]=size(X);
Ntest=size(Xtest1,1);
minx=min(min(X)); 
maxx=max(max(X));



%
% Add noise to the digit maps
%

noisefactor =1;

noise = noisefactor*maxx; % sd for Gaussian noise


Xn = X; 
for i=1:N;
  randn('state', i);
  Xn(i,:) = X(i,:) + noise*randn(1, dim);
end

Xnt = Xtest1; 
for i=1:size(Xtest1,1);
  randn('state', N+i);
  Xnt(i,:) = Xtest1(i,:) + noise*randn(1,dim);
end

Xtest = Xtest2; 
for i=1:size(Xtest2,1);
  randn('state', N+10+i);
  Xtest(i,:) = Xtest2(i,:) + noise*randn(1,dim);
end
%
% select training set
%
Xtr = X(1:1:end,:);



sig2 =dim*mean(var(Xtr)); % rule of thumb

sigmafactor = 0.7;

sig2=sig2*sigmafactor;




%
% kernel based Principal Component Analysis using the original training data
%


disp('Kernel PCA: extract the principal eigenvectors in feature space');
disp(['sig2 = ', num2str(sig2)]);


% linear PCA
[lam_lin,U_lin] = pca(Xtr);

% kernel PCA
[lam,U] = kpca(Xtr,'RBF_kernel',sig2,[],'eig',240); 
[lam, ids]=sort(-lam); lam = -lam; U=U(:,ids);

%
% % Visualize the ith eigenvector 
% %
% disp(' ');
% disp(' Visualize the eigenvectors');
% 
% % define the number of eigen vectors to visualize
% nball = min(length(lam),length(lam_lin));
% eigs = [1:10];
% ne=length(eigs); 
% 
% % compute the projections of the ith canonical basis vector e_i, i=1:240
% k = kernel_matrix(Xtr,'RBF_kernel',sig2,eye(dim))'; proj_e=k*U;
% figure; colormap(gray); eigv_img=zeros(ne,dim);  
% 
% for i=1:ne; 
%     ieig=eigs(i);
% 
%     % linear PCA 
%     if ieig<=length(lam_lin),
%       subplot(3, ne, i); 
%       pcolor(1:15,16:-1:1,reshape(real(U_lin(:,ieig)), 15, 16)'); shading interp; 
%       set(gca,'xticklabel',[]);set(gca,'yticklabel',[]);
%       title(['\lambda',sprintf('%d\n%.4f', ieig, lam_lin(ieig))],'fontSize',6) 
%       if i==1, ylabel('linear'), end
%       drawnow
%     end
%       
%     % kPCA  
%     % The preimage of the eigenvector in the feature space might not exist! 
%     if ieig<=length(lam),
%       eigv_img(i,:) = preimage_rbf(Xtr,sig2,U(:,ieig),zeros(1,dim),'d');
%       subplot(3, ne, i+ne); 
%       pcolor(1:15,16:-1:1,reshape(real(eigv_img(i,:)),15, 16)'); shading interp;
%       set(gca,'xticklabel',[]);set(gca,'yticklabel',[]);
%       title(['\lambda',sprintf('%d\n%.4f', ieig, lam(ieig))],'fontSize',6) 
%       if i==1, ylabel('kernel'); end
%       drawnow
%     end
% 
%     if ieig<=size(proj_e,2),
%       subplot(3, ne, i+ne+ne); 
%       pcolor(1:15,16:-1:1,reshape(real(proj_e(:,ieig)), 15, 16)'); shading interp; 
%       set(gca,'xticklabel',[]);set(gca,'yticklabel',[]); 
%       title(['\lambda',sprintf('%d\n%.4f', ieig, lam(ieig))],'fontSize',6) 
%       if i==1, ylabel('kernel'); end
%       drawnow
%     end
%     
% end

%
% Denoise using the first principal components
%
disp(' ');
disp(' Denoise using the first PCs');

% choose the digits for test
digs=[0:9]; ndig=length(digs);
m=2; % Choose the mth data for each digit 

Xdt=zeros(ndig,dim);
Xdtest=zeros(ndig,dim);
Xdtrain=zeros(N,dim);


%
% figure of all digits
%
%
figure; 
colormap('gray'); 
title('Denosing using linear PCA'); tic


% which number of eigenvalues of kpca
npcs = [2.^(0:7) 190];
lpcs = length(npcs);

errordt = zeros(1,lpcs); 
errordtest = zeros(1,lpcs); 
errordtrain= zeros(1,lpcs); 


for k=1:lpcs;
 nb_pcs=npcs(k); 
 disp(['nb_pcs = ', num2str(nb_pcs)]); 
 Ud=U(:,(1:nb_pcs)); lamd=lam(1:nb_pcs);
    
 for i=1:ndig
   dig=digs(i);
   fprintf('digit %d : ', dig)
   xt=Xnt(i,:);
   xtest = Xtest(i,:);
   if k==1 
     % plot the original clean digits
     %
     subplot(2+lpcs, ndig, i);
     pcolor(1:15,16:-1:1,reshape(Xtest1(i,:), 15, 16)'); shading interp; 
     set(gca,'xticklabel',[]);set(gca,'yticklabel',[]);        
     
     if i==1, ylabel('original'), end 
     
     % plot the noisy digits 
     %
     subplot(2+lpcs, ndig, i+ndig); 
     pcolor(1:15,16:-1:1,reshape(xt, 15, 16)'); shading interp; 
     set(gca,'xticklabel',[]);set(gca,'yticklabel',[]);        
     if i==1, ylabel('noisy'), end
     drawnow
   end    
   Xdt(i,:) = preimage_rbf(Xtr,sig2,Ud,xt,'denoise');
   Xdtest(i,:) = preimage_rbf(Xtr,sig2,Ud,xtest,'denoise');
   
   errordt(1,k) = errordt(1,k)+mae(Xdt(i,:)-Xtest1(i,:));
   errordtest(1,k) = errordtest(1,k)+mae(Xdtest(i,:)-Xtest2(i,:));
   
   subplot(2+lpcs, ndig, i+(2+k-1)*ndig);
   pcolor(1:15,16:-1:1,reshape(Xdt(i,:), 15, 16)'); shading interp; 
   set(gca,'xticklabel',[]);set(gca,'yticklabel',[]);           
   if i==1, ylabel(['n=',num2str(nb_pcs)]); end
   drawnow    
 end % for i
 for j = 1:N
     xtrain=Xn(j,:);
     Xdtrain(j,:) = preimage_rbf(Xtr,sig2,Ud,xtrain,'denoise');
     errordtrain(1,k) = errordtrain(1,k)+mae(Xdtrain(j,:)-X(j,:));
 end
 
 
end % for k
saveas(gcf,strcat('plotsPCA/denoisy_test1_HW_kPCAdigits_sigFactor_',num2str(sigmafactor),'.jpg'))
savefig(strcat('plotsPCA/denoisy_test1_HW_kPCAdigits_sigFactor_',num2str(sigmafactor),'.fig'))
close all


mae_training_kPCA = (errordtrain)/N;
mae_test1_kPCA = (errordt)/ndig;
mae_test2_kPCA = (errordtest)/ndig;
%
% denosing using Linear PCA for comparison
%

% which number of eigenvalues of pca
npcs = [2.^(0:7) 190];
lpcs = length(npcs);

errordt_lineal = zeros(1,lpcs); 
errordtest_lineal = zeros(1,lpcs); 
errordtrain_lineal= zeros(1,lpcs);

figure; colormap('gray');title('Denosing using linear PCA');

for k=1:lpcs;
 nb_pcs=npcs(k); 
 Ud=U_lin(:,(1:nb_pcs)); lamd=lam(1:nb_pcs);
    
 for i=1:ndig
    dig=digs(i);
    xt=Xnt(i,:);
    xtest = Xtest(i,:);
    
    proj_lin=xt*Ud; % projections of linear PCA
    proj_lin_test=xtest*Ud; % projections of linear PCA
    
    if k==1 
        % plot the original clean digits
        %
        subplot(2+lpcs, ndig, i);
        pcolor(1:15,16:-1:1,reshape(Xtest2(i,:), 15, 16)'); shading interp; 
        set(gca,'xticklabel',[]);set(gca,'yticklabel',[]);                
        if i==1, ylabel('original'), end  
        
        % plot the noisy digits 
        %
        subplot(2+lpcs, ndig, i+ndig); 
        pcolor(1:15,16:-1:1,reshape(xtest, 15, 16)'); shading interp; 
        set(gca,'xticklabel',[]);set(gca,'yticklabel',[]);        
        if i==1, ylabel('noisy'), end
    end
    Xdt_lin(i,:) = proj_lin*Ud';
    Xdtest_lin(i,:) = proj_lin_test*Ud';

    errordt_lineal(1,k) = errordt_lineal(1,k)+mae(Xdt_lin(i,:)-Xtest1(i,:));
    errordtest_lineal(1,k) = errordtest_lineal(1,k)+mae(Xdtest_lin(i,:)-Xtest2(i,:));
    
    subplot(2+lpcs, ndig, i+(2+k-1)*ndig);
    pcolor(1:15,16:-1:1,reshape(Xdtest_lin(i,:), 15, 16)'); shading interp; 
    set(gca,'xticklabel',[]);set(gca,'yticklabel',[]);        
    
    if i==1, ylabel(['n=',num2str(nb_pcs)]), end
 end % for i
 
 for j = 1:N
     xtrain=Xn(j,:);
     proj_lin_train=xtrain*Ud;
     Xdtrain(j,:) = proj_lin_train*Ud';
     
     errordtrain_lineal(1,k) = errordtrain_lineal(1,k)+mae(Xdtrain(j,:)-X(j,:));
 end
 
 
end % for k
% saveas(gcf,strcat('plotsPCA/denoisy_test2_HW_linearPCAdigits_sigFactor_',num2str(sigmafactor),'.jpg'))
% savefig(strcat('plotsPCA/denoisy_test2_HW_linearPCAdigits_sigFactor_',num2str(sigmafactor),'.fig'))
close all

mae_training_linearPCA = (errordtrain_lineal)/N;
mae_test1_linearPCA = (errordt_lineal)/ndig;
mae_test2_linearPCA = (errordtest_lineal)/ndig;

