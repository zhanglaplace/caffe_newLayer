data=dlmread('./sphereface_flip_cos_fliplayer.txt');
x=data(:,2);
y=data(:,1);

data1=dlmread('./faceplusplus.txt');
x1=data1(:,2);
y1=data1(:,1);

data2=dlmread('./dahua.txt');
x2=data2(:,2);
y2=data2(:,1);

data3=dlmread('./tencent.txt');
x3=data3(:,2);
y3=data3(:,1);


data4=dlmread('./deepid3.txt');
x4=data4(:,2);
y4=data4(:,1);

data5=dlmread('./baidu.txt');
x5=data5(:,2);
y5=data5(:,1);

data6=dlmread('./dlib.txt');
x6=data6(:,2);
y6=data6(:,1);

data7=dlmread('./deepid.txt');
x7=data7(:,2);
y7=data7(:,1);

data8=dlmread('./sensingtech.txt');
x8=data8(:,2);
y8=data8(:,1);

plot(x,y,x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6,x7,y7,'--',x8,y8,'--');
xlabel('false positive rate');
ylabel('true positive rate');
title( 'ROC');
legend('ours','face++','dahua','tencent','deepid3','baidu','dlib','deepid','sensingtech');