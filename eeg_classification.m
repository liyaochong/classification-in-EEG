%% 清空环境变量
clear;
clc;
close all;
%% 添加当前路径至工作环境
currentFolder = pwd;
addpath(genpath(currentFolder));
%% 提取出一组样本的16个信道的原始EEG信号，进行显示
load('data\EEGdata_newdata.mat') %数据的路径可修改
A = EEGdata_awake(:,:,1); %A为数据集中的第一个样本
data = A';
len = 2048;
t1=data(1:len,1); %第1个信道
t2=data(1:len,2); %第2个信道
t3=data(1:len,3); %第3个信道
t4=data(1:len,4); %第4个信道
t5=data(1:len,5); %第5个信道
t6=data(1:len,6); %第6个信道
t7=data(1:len,7); %第7个信道
t8=data(1:len,8); %第8个信道
t9=data(1:len,9); %第9个信道
t10=data(1:len,10); %第10个信道
t11=data(1:len,11); %第11个信道
t12=data(1:len,12); %第12个信道
t13=data(1:len,13); %第13个信道
t14=data(1:len,14); %第14个信道
t15=data(1:len,15); %第15个信道
t16=data(1:len,16); %第16个信道
%% 绘制原始信号的波形图
figure(1)
subplot(8,2,1);plot(t1);title('第1个信道');
subplot(8,2,2);plot(t2);title('第2个信道');
subplot(8,2,3);plot(t3);title('第3个信道');
subplot(8,2,4);plot(t4);title('第4个信道');
subplot(8,2,5);plot(t5);title('第5个信道');
subplot(8,2,6);plot(t6);title('第6个信道');
subplot(8,2,7);plot(t7);title('第7个信道');
subplot(8,2,8);plot(t8);title('第8个信道');
subplot(8,2,9);plot(t9);title('第9个信道');
subplot(8,2,10);plot(t10);title('第10个信道');
subplot(8,2,11);plot(t11);title('第11个信道');
subplot(8,2,12);plot(t12);title('第12个信道');
subplot(8,2,13);plot(t13);title('第13个信道');
subplot(8,2,14);plot(t14);title('第14个信道');
subplot(8,2,15);plot(t15);title('第15个信道');
subplot(8,2,16);plot(t16);title('第16个信道');
%% 对各个信号进行小波变换（16通道）
[c1,l1] = wavedec(t1,3,'db2');
[c2,l2] = wavedec(t2,3,'db2');
[c3,l3] = wavedec(t3,3,'db2');
[c4,l4] = wavedec(t4,3,'db2');
[c5,l5] = wavedec(t5,3,'db2');
[c6,l6] = wavedec(t6,3,'db2');
[c7,l7] = wavedec(t7,3,'db2');
[c8,l8] = wavedec(t8,3,'db2');
[c9,l9] = wavedec(t9,3,'db2');
[c10,l10] = wavedec(t10,3,'db2');
[c11,l11] = wavedec(t11,3,'db2');
[c12,l12] = wavedec(t12,3,'db2');
[c13,l13] = wavedec(t13,3,'db2');
[c14,l14] = wavedec(t14,3,'db2');
[c15,l15] = wavedec(t15,3,'db2');
[c16,l16] = wavedec(t16,3,'db2');
%% 绘制信号经过小波变换后的波形图（16信道）
figure(2)
subplot(8,2,1);plot(c1);title('第1个信道');
subplot(8,2,2);plot(c2);title('第2个信道');
subplot(8,2,3);plot(c3);title('第3个信道');
subplot(8,2,4);plot(c4);title('第4个信道');
subplot(8,2,5);plot(c5);title('第5个信道');
subplot(8,2,6);plot(c6);title('第6个信道');
subplot(8,2,7);plot(c7);title('第7个信道');
subplot(8,2,8);plot(c8);title('第8个信道');
subplot(8,2,9);plot(c9);title('第9个信道');
subplot(8,2,10);plot(c10);title('第10个信道');
subplot(8,2,11);plot(c11);title('第11个信道');
subplot(8,2,12);plot(c12);title('第12个信道');
subplot(8,2,13);plot(c13);title('第13个信道');
subplot(8,2,14);plot(c14);title('第14个信道');
subplot(8,2,15);plot(c15);title('第15个信道');
subplot(8,2,16);plot(c16);title('第16个信道');
%% 导入两类样本数据以及对应标签（从原始数据集中提出）
% 对应标签说明：0代表awake状态  1代表sleep状态
load('data\awake.mat') %数据的路径可修改
load('data\awake_label.mat') %数据的路径可修改
load('data\sleep.mat') %数据的路径可修改
load('data\sleep_label.mat') %数据的路径可修改
%% 小波变换
awake_1 = zeros(230,2055); %分配内存空间
sleep_1 = zeros(150,2055);
% 分别对两类数据进行小波变换
for i=1:1:230
    t = awake(i,:);
    [c,l] = wavedec(t,3,'db2');
    awake_1(i,:) = c;
end
for j=1:1:150
    q = sleep(j,:);
    [c,l] = wavedec(t,3,'db2');
    sleep_1(j,:) = c;
end
% 将进过小波变换的两类数据分别送回原变量
awake = awake_1;
sleep = sleep_1;
%% 将所有数据进行归一化处理
input_train = [awake;sleep];
[inputn,inputps] = mapminmax(input_train);
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
disp(['f1 =' num2str(f1)]);
disp(['recall =' num2str(recall)]);
disp(['accuracy =' num2str(accuracy)]);
plotconfusion(yPre',yTe')