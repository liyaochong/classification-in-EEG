%% 清空环境变量
clear;
clc;
close all;

%% 添加当前路径至工作环境
currentFolder = pwd;
addpath(genpath(currentFolder));

%% 导入原始的数据集，并提取出一组样本的16个信道的原始EEG信号进行显示
load('data\EEGdata_newdata.mat') % 数据的路径可修改
data = EEGdata_sleep(:,:,14); % A为数据集中的第一个样本
% 绘制各个信道的EEG信号
figure(1)
main_title = 'EEGdata-sleep';
suptitle(main_title);
for i = 1:size(data,1)
    sub_title = ['Channel ',num2str(i)];
    subplot(8,2,i);
    plot(data(i,:));
    title(sub_title);
end

%% 对16通道的脑电信号进行操作小波包分解(数据已导出)
% 对Awake数据进行特征提取
% n = 4;
% WPE_awake = WPE_obtain(EEGdata_awake,n);
% 对Sleep数据进行特征提取
% WPE_sleep = WPE_obtain(EEGdata_sleep,n);
%% 将所有数据进行归一化处理
load('data\awake_label.mat') 
load('data\sleep_label.mat') 
load('data\WPE_awake.mat')
load('data\WPE_sleep')

input_all = [WPE_awake;WPE_sleep];
[inputn,inputps] = mapminmax(input_all);
% 归一化后的数据送回原变量
awake = inputn((1:230),:);
sleep = inputn((231:380),:);

%% 两类信号的70%的样本用于训练SVM，30%的样本用于测试模型分类能力
%产生随机数，确保实验的无序性
number_1 = randperm(size(awake,1)); 
number_2 = randperm(size(sleep,1));
% 区分awake状态下的样本数据
% 230*0.7=161个awake状态样本用于训练
% 230*0.3=69个awake状态样本用于测试
awake_train_label = zeros(size(awake_label,1)*0.7,size(awake_label,2));
awake_test_label = zeros(size(awake_label,1)*0.3,size(awake_label,2));
awake_train = zeros(size(awake,1)*0.7,size(awake,2));
awake_test = zeros(size(awake,1)*0.3,size(awake,2));
% 分配70%的awake数据至训练集
for i=1:1:(size(awake,1)*0.7)
    awake_train(i,:) = awake(number_1(i),:);
    awake_train_label(i,1) = awake_label(number_1(i),:);
end
% 分配30%的awake数据(剩余的)至测试集
for i=(size(awake,1)*0.7+1):1:size(awake,1)
    awake_test(i-(size(awake,1)*0.7),:) = awake(number_1(i),:); 
    awake_test_label(i-(size(awake,1)*0.7),1) = awake_label(number_1(i),:);  
end
% 区分睡眠状态下的样本数据
% 150*0.7=105个sleep状态样本用于训练
% 150*0.3=45个sleep状态样本用于测试
sleep_train_label = zeros(size(sleep_label,1)*0.7,size(sleep_label,2));
sleep_test_label = zeros(size(sleep_label,1)*0.3,size(sleep_label,2));
sleep_train = zeros(size(sleep,1)*0.7,size(sleep,2));
sleep_test = zeros(size(sleep,1)*0.3,size(sleep,2));
% 分配70%的sleep数据至训练集
for j=1:1:size(sleep,1)*0.7
    sleep_train(j,:) = sleep(number_2(j),:);
    sleep_train_label(j,1) = sleep_label(number_2(j),:);
end
% 分配30%的sleep数据(剩余的)至测试集
for j=(size(sleep,1)*0.7+1):1:size(sleep,1)
    sleep_test(j-(size(sleep,1)*0.7),:)= sleep(number_2(j),:);
    sleep_test_label(j-(size(sleep,1)*0.7),1) = sleep_label(number_2(j),:);
end

%% 汇总数据
% 说明：xTr、yTr对应为全体训练样本的数据、标签
% 说明：xTe、yTe对应为全体测试样本的数据、标签
xTr = [sleep_train;awake_train];
yTr = [sleep_train_label;awake_train_label];
xTe = [sleep_test;awake_test];
yTe = [sleep_test_label;awake_test_label];

%% 训练、分类
model = fitcsvm(xTr,yTr); % 训练数据，得到模型参数
yPre = predict(model,xTe);% 将测试数据送入模型，得到预测标签yPre

%% 实验结果分析
Acc = sum(yTe == yPre)/length(yTe); 
C = confusionmat(yTe, yPre); % C为混淆矩阵
recall = 0;
accuracy = 0;
n = length(C);
for i=1:n
    recall = recall + C(i,i)/sum(C(i,:));
    accuracy = accuracy + C(i,i)/sum(C(:,i));
end
recall = recall/n;
accuracy = accuracy/n;
f1 = 2*(recall*accuracy)/(recall+accuracy);
disp(['Acc =' num2str(Acc)]);
disp(['F1 =' num2str(f1)]);
disp(['Recall =' num2str(recall)]);
disp(['Accuracy =' num2str(accuracy)]);
plotconfusion(yPre',yTe')