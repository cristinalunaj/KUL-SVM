close all
class1 = find(Ytrain==1)
class0 = find(Ytrain==-1)

figure
scatter(Xtrain(class1,1),Xtrain(class1,2),'r');
hold on
scatter(Xtrain(class0,1),Xtrain(class0,2),'b');
hold off

class1 = find(Yval==1)
class0 = find(Yval==-1)
figure
scatter(Xval(class1,1),Xval(class1,2),'r');
hold on
scatter(Xval(class0,1),Xval(class0,2),'b');

gam=10
sig2=100
[alpha,b] = trainlssvm({Xtrain,Ytrain,'c',gam,sig2,'RBF_kernel'});
figure
plotlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel'},{alpha,b});


