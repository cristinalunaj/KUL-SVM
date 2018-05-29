clear;clc;

load two3drings;        % load the toy example

[N,d]=size(X);

perm=randperm(N);   % shuffle the data
X=X(perm,:);
%for sig2 = [0.001, 0.005,0.01, 0.2, 0.5, 1]
sig2=50;              % set the kernel parameters


K=kernel_matrix(X,'RBF_kernel',sig2);   %compute the RBF kernel (affinity) matrix

D=diag(sum(K));         % compute the degree matrix (sum of the columns of K)

[U,lambda]=eigs(inv(D)*K,3);  % Compute the 3 largest eigenvalues/vectors using Lanczos
                              % The largest eigenvector does not contain
                              % clustering information. For binary clustering,
                              % the solution is the second largest eigenvector.
                              
clust=sign(U(:,2)); % Threshold the eigenvector solution to obtain binary cluster indicators

[y,order]=sort(clust,'descend');    % Sort the data using the cluster information
Xsorted=X(order,:);

Ksorted=kernel_matrix(Xsorted,'RBF_kernel',sig2);   % Compute the kernel matrix of the
                                                    % sorted data.


proj=K*U(:,2:3);    % Compute the projections onto the subspace spanned by the second,
                    % and third largest eigenvectors.
                                                    
                                                   
%%%% PLOTTING SECTION %%%%                                                 
subplot(1,2,1)
scatter3(X(:,1),X(:,2),X(:,3),15);
title('Two interlaced rings in a 3D space');

subplot(1,2,2);
scatter3(X(:,1),X(:,2),X(:,3),30,clust);
title('Clustering results');
% saveas(gcf,strcat('clusterPlots/initialData_sig',num2str(sig2),'.jpg'))
% savefig(strcat('clusterPlots/initialData_sig',num2str(sig2),'.fig'))
disp('<<<<<<<<<<<<Press any key>>>>>>>>>>>>>>');
pause;


figure;
subplot(1,2,1);
imshow(K);
title('Kernel matrix of the original data');

subplot(1,2,2);
imshow(Ksorted);
title('Kernel matrix after sorting the data using the cluster information');
% 
% saveas(gcf,strcat('clusterPlots/Kmatrix_sig',num2str(sig2),'.jpg'))
% savefig(strcat('clusterPlots/Kmatrix_sig',num2str(sig2),'.fig'))



figure;
scatter(proj(:,1),proj(:,2),15,clust);
title('Projections onto subspace spanned by the 2nd and 3rd largest eigenvectors');
% saveas(gcf,strcat('clusterPlots/project_sig',num2str(sig2),'.jpg'))
% savefig(strcat('clusterPlots/project_sig',num2str(sig2),'.fig'))
% close all
%end






                              
                  
                             
                      
