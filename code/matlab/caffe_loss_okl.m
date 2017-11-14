clc;
clear;

% load the log file of caffe model
fid = fopen('./extra/INFO2017-11-14T15-03-29.log', 'r');
tline = fgetl(fid);

%get arrays to draw figures
testIter = []	
top_1 = [];
top_5 = [];
acc_1 = [];
acc_2 = [];
acc_3 = [];
acc_4 = [];
acc_5 = [];
acc_6 = [];
acc_7 = [];
lossTest= [];
lossIter= [];
lossArray = [];

%record the last line
lastLine = '';

%read line
while ischar(tline)
    %%%%%%%%%%%%%% the accuracy line %%%%%%%%%%%%%%
    k = strfind(tline, 'Test net output #');
    if (k)
        % top-1
        k = strfind(tline, '#0');
        if (k) 
            % If the string contain test and accuracy at the same time  
            % The bias from 'accuracy' to the float number  
            indexStart = k + 15; 
            indexEnd = size(tline);
            str = tline(indexStart : indexEnd(2));  
            top_1 = [top_1, str2num(str)];
        end
        
        % top-5
        k = strfind(tline, '#1');
        if (k) 
            % If the string contain test and accuracy at the same time  
            % The bias from 'accuracy' to the float number  
            indexStart = k + 15; 
            indexEnd = size(tline);
            str = tline(indexStart : indexEnd(2));  
            top_5 = [top_5, str2num(str)];
        end
        
        % lossTest
        k = strfind(tline, '#2');
        if (k) 
            % If the string contain test and accuracy at the same time  
            % The bias from 'accuracy' to the float number  
            indexStart = k + 11; 
            indexEnd = size(tline);
            str = tline(indexStart : indexEnd(2)-20);  
            lossTest = [lossTest, str2num(str)];
        end
        
        % acc1
        k = strfind(tline, '#3');
        if (k) 
            % If the string contain test and accuracy at the same time  
            % The bias from 'accuracy' to the float number  
            indexStart = k + 15; 
            indexEnd = size(tline);
            str = tline(indexStart : indexEnd(2));  
            acc_1 = [acc_1, str2num(str)];
        end
        
        %acc2
         k = strfind(tline, '#4');
        if (k) 
            % If the string contain test and accuracy at the same time  
            % The bias from 'accuracy' to the float number  
            indexStart = k + 15; 
            indexEnd = size(tline);
            str = tline(indexStart : indexEnd(2));  
            acc_2 = [acc_2, str2num(str)];
        end
        
        %acc3
          k = strfind(tline, '#5');
        if (k) 
            % If the string contain test and accuracy at the same time  
            % The bias from 'accuracy' to the float number  
            indexStart = k + 15; 
            indexEnd = size(tline);
            str = tline(indexStart : indexEnd(2));  
            acc_3 = [acc_3, str2num(str)];
        end
        
        %acc4
        k = strfind(tline, '#6');
        if (k) 
            % If the string contain test and accuracy at the same time  
            % The bias from 'accuracy' to the float number  
            indexStart = k + 15; 
            indexEnd = size(tline);
            str = tline(indexStart : indexEnd(2));  
            acc_4 = [acc_4, str2num(str)];
        end
        
        %acc5
        k = strfind(tline, '#7');
        if (k) 
            % If the string contain test and accuracy at the same time  
            % The bias from 'accuracy' to the float number  
            indexStart = k + 15; 
            indexEnd = size(tline);
            str = tline(indexStart : indexEnd(2));  
            acc_5 = [acc_5, str2num(str)];
        end
        
        %acc6
        k = strfind(tline, '#8');
        if (k) 
            % If the string contain test and accuracy at the same time  
            % The bias from 'accuracy' to the float number  
            indexStart = k + 15; 
            indexEnd = size(tline);
            str = tline(indexStart : indexEnd(2));  
            acc_6 = [acc_6, str2num(str)];
        end
        
        %acc7
        k = strfind(tline, '#9');
        if (k) 
            % If the string contain test and accuracy at the same time  
            % The bias from 'accuracy' to the float number  
            indexStart = k + 15; 
            indexEnd = size(tline);
            str = tline(indexStart : indexEnd(2));  
            acc_7 = [acc_7, str2num(str)];
        end

        % Get the number of index
        k = strfind(lastLine, 'Iteration');
        if (k)
            indexStart = k + 10;
            indexEnd = strfind(lastLine, ',');
            str2 = lastLine(indexStart : indexEnd - 1);
            testIter  = [testIter, str2num(str2)];
        end
        % Concatenation of two string
    end
    
    %%%%%%%%%%%%%% the loss line %%%%%%%%%%%%%%
    k1 = strfind(tline, 'Iteration');
    if (k1)
       k2 = strfind(tline, 'loss');
       if (k2)
           indexStart = k2 + 7;
           indexEnd = size(tline);
           str1 = tline(indexStart:indexEnd(2));
           indexStart = k1 + 10;
           indexEnd = strfind(tline, '(') - 1;
           str2 = tline(indexStart:indexEnd);
           res_str1 = strcat(str2, '/', str1);
           lossIter  = [lossIter,  str2num(str2)];
           lossArray = [lossArray, str2num(str1)];
       end
    end
    lastLine = tline;
    tline = fgetl(fid);    
end

plot(lossIter(1,:),lossArray(1,:),'-',testIter,lossTest,'--');
xlabel('iters');
ylabel('loss');
title( 'iteration vs loss');
legend('train\_loss','test\_top\_1');


figure(2);
plot(testIter/500,[top_1',top_5',acc_1',acc_2',acc_3',acc_4',acc_5',acc_6',acc_7']);
xlabel('iters');
ylabel('accuracy');
title( 'iteration vs accurancy');
legend('top\_1','top\_5','acc\_1','acc\_2','acc\_3','acc\_4','acc\_5','acc\_6','acc\_7');



