%data = load('breast_cancer_wisconsin_data.mat','-ascii'); function_type = 'c';
clc;clear;close all
data = load('shuttle.dat','-ascii'); function_type = 'c';  data = data(1:end,:);

addpath('../LSSVMlab')

X = data(1:43500,1:end-1);
Y = data(1:43500,end);
testX = data(43501:end,1:end-1);
testY = data(43501:end,end);

%% class1 
load('SHUTTLE_RESULTS/CLASS1/k3RBFindexClass1Detected.mat', '-ascii');
indexOriginal = 1:14500;
testWithIndex = [indexOriginal',testY];
classifiedAs1 = testWithIndex(k3RBFindexClass1Detected,:)
wrongClassifiedAs1 = classifiedAs1(find(classifiedAs1(:,2)~=1),:)

%% class 4 
load('SHUTTLE_RESULTS/CLASS4/k3RBFindexClass4Detected.mat', '-ascii');
classDetected = 4; 
indexOriginal = 1:14500;
testWithIndex = [indexOriginal',testY];
classifiedAs4 = testWithIndex(k3RBFindexClass4Detected,:)
wrongClassifiedAs5 = classifiedAs4(find(classifiedAs4(:,2)~=classDetected),:)

%% class 5
load('SHUTTLE_RESULTS/CLASS5/k3RBFindexClass5Detected.mat', '-ascii');
classDetected = 5; 
indexOriginal = 1:14500;
testWithIndex = [indexOriginal',testY];
classifiedAs5 = testWithIndex(k3RBFindexClass5Detected,:)
wrongClassifiedAs5 = classifiedAs5(find(classifiedAs5(:,2)~=classDetected),:)

%% class 3
load('SHUTTLE_RESULTS/CLASS3/k3RBFindexClass3Detected.mat', '-ascii');
classDetected = 3; 
indexOriginal = 1:14500;
testWithIndex = [indexOriginal',testY];
classifiedAs3 = testWithIndex(k3RBFindexClass3Detected,:)
wrongClassifiedAs3 = classifiedAs3(find(classifiedAs3(:,2)~=classDetected),:)

%% class 2
load('SHUTTLE_RESULTS/CLASS2/k3RBFindexClass2Detected.mat', '-ascii');
classDetected = 2; 
indexOriginal = 1:14500;
testWithIndex = [indexOriginal',testY];
classifiedAs2 = testWithIndex(k3RBFindexClass2Detected,:)
wrongClassifiedAs2 = classifiedAs2(find(classifiedAs2(:,2)~=classDetected),:)

%% class 7

load('SHUTTLE_RESULTS/CLASS7/k3RBFindexClass7Detected.mat', '-ascii');
classDetected =7; 
indexOriginal = 1:14500;
testWithIndex = [indexOriginal',testY];
classifiedAs7 = testWithIndex(k3RBFindexClass7Detected,:)
wrongClassifiedAs7 = classifiedAs7(find(classifiedAs7(:,2)~=classDetected),:)

%% class6
load('SHUTTLE_RESULTS/CLASS7/k3RBFindexClass6Detected.mat', '-ascii');
classDetected =6; 
indexOriginal = 1:14500;
testWithIndex = [indexOriginal',testY];
classifiedAs6 = testWithIndex(k3RBFindexClass6Detected,:)
wrongClassifiedAs6 = classifiedAs6(find(classifiedAs6(:,2)~=classDetected),:)

%% total

%data = load('breast_cancer_wisconsin_data.mat','-ascii'); function_type = 'c';
clc;clear;close all
data = load('shuttle.dat','-ascii'); function_type = 'c';  data = data(1:end,:);

addpath('../LSSVMlab')

X = data(1:43500,1:end-1);
Y = data(1:43500,end);
testX = data(43501:end,1:end-1);
testY = data(43501:end,end);



load('SHUTTLE_RESULTS/CLASS1/k3RBFindexClass1Detected.mat', '-ascii');
indexOriginal = 1:14500;
newClass = zeros(1,14500)';
testWithIndex = [indexOriginal',testY,newClass];
classifiedAs1 = testWithIndex(k3RBFindexClass1Detected,:)
classifiedAs1(:,3) = 1;

load('SHUTTLE_RESULTS/CLASS4/k3RBFindexClass4Detected.mat', '-ascii');
classifiedAs4 = testWithIndex(k3RBFindexClass4Detected,:)
classifiedAs4(:,3) = 4;

load('SHUTTLE_RESULTS/CLASS5/k3RBFindexClass5Detected.mat', '-ascii');
classifiedAs5 = testWithIndex(k3RBFindexClass5Detected,:)
classifiedAs5(:,3) = 5;

load('SHUTTLE_RESULTS/CLASS3/k3RBFindexClass3Detected.mat', '-ascii');
classifiedAs3 = testWithIndex(k3RBFindexClass3Detected,:)
classifiedAs3(:,3) = 3;

load('SHUTTLE_RESULTS/CLASS2/k3RBFindexClass2Detected.mat', '-ascii');
classifiedAs2 = testWithIndex(k3RBFindexClass2Detected,:)
classifiedAs2(:,3) = 2;

load('SHUTTLE_RESULTS/CLASS7/k3RBFindexClass7Detected.mat', '-ascii');
classifiedAs7 = testWithIndex(k3RBFindexClass7Detected,:)
classifiedAs7(:,3) = 7;

load('SHUTTLE_RESULTS/CLASS7/k3RBFindexClass6Detected.mat', '-ascii');
classifiedAs6 = testWithIndex(k3RBFindexClass6Detected,:)
classifiedAs6(:,3) = 6;

unionClass = [classifiedAs1;classifiedAs4;classifiedAs5;classifiedAs3;classifiedAs2;classifiedAs7;classifiedAs6];
plotconfusion(unionClass(:,2),unionClass(:,3))
A = 'END'


