clear
close all


X = 3.*randn(100,2);
ssize = 8;
sig2 = 1;
indexCandidates = ones(1,ssize)*-1;

for sig2 =  [1]%[0.001, 0.005,0.01,0.1,1, 10]
subset = zeros(ssize,2);
for t = 1:100,

  %
  % new candidate subset
  %
  r = ceil(rand*ssize);
  candidate = [subset([1:r-1 r+1:end],:); X(t,:)];
  
  %
  % is this candidate better than the previous?
  %
  if kentropy(candidate, 'RBF_kernel',sig2)>...
        kentropy(subset, 'RBF_kernel',sig2),
    subset = candidate;
    indexCandidates= [indexCandidates([1:r-1 r+1:end]),t];
  end
  
  %
  % make a figure
  %
  figure
  plot(X(:,1),X(:,2),'b*'); hold on;
  plot(subset(:,1),subset(:,2),'ro','linewidth',6); hold off; 
  pause(1)
  
  close all
  
if(size(find(indexCandidates==-1),2)<ssize&&size(find(indexCandidates~=-1),2)>=3)
      % make a figure
%
figure
subplot(1,2,1);
plot(X(:,1),             X(:,2),'b*'); hold on;
plot(subset(:,1),subset(:,2),'ro','linewidth',6); hold off; 
title('original space')

%
% transform the data in feature space
%

nonZeroCandidates = indexCandidates(find(indexCandidates~=-1))
newsubset = subset(end-size(nonZeroCandidates,2)+1:end,:)

       
features = AFEm(newsubset,'RBF_kernel',sig2,X);
subplot(1,2,2);
plot3(features(:,1),             features(:,2),             features(:,3),'k*'); hold on;
plot3(features(nonZeroCandidates,1),features(nonZeroCandidates,2),features(nonZeroCandidates,3),'ro','linewidth',6); hold off;
title('feature space')
pause(1)

end
end
  
  
end
% saveas(gcf,strcat('plotsFixedLSSVM/results_sig',num2str(sig2),'.jpg'))
% savefig(strcat('plotsFixedLSSVM/results_sig',num2str(sig2),'.fig'))
% close all
% end