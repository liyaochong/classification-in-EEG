function [ WPE_EEGdata ] = WPE_obtain( EEGdata,n )
% WPE_obtain 函数用来获取EEG信号的小波包熵
WPE_EEGdata = zeros(size(EEGdata,3),size(EEGdata,1));
E = zeros(1,2^n); % 2^n,此参数与小波分解层数有关
P = zeros(1,2^n); % 2^n,此参数与小波分解层数有关
for i = 1:size(EEGdata,3)
    for j = 1:16 % 表示的16通道的脑电信号
        sample = EEGdata(:,:,i); % 提取的是第i个样本
        t = sample(j,:);
        t = t';
        wpt = wpdec(t,4,'haar'); % 小波包分解，4是层数，进行Haar小波分解，参数可改
        for k = 1:2^4 % 4层分解，16组系数
            E(k) = sum(abs(wprcoef(wpt,[4,k-1])).^2); % 能量求和
        end
        for q = 1:length(E)
            P(q)= E(q)/(sum(E));
        end
        WPE_EEGdata(i,j) = -sum(P.*log(P)); % 熵的定义式      
    end   
end

end

