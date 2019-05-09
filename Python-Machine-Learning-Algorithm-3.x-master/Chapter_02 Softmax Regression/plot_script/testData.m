%% ����������ݼ�  
function [ testDataSet, testLabelSet ] = testData( weights, m, n)  
    testDataSet = ones(m,n);%������ȫ1�ľ���  
    testLabelSet = zeros(m,1);  
    for i = 1:m  
        testDataSet(i,2) = rand()*6-3;  
        testDataSet(i,3) = rand()*19 - 4;  
    end  
      
    %% ����������ݵ���������  
    for i = 1:m  
        testResult = testDataSet(i,:)*weights;  
        [C,I] = max(testResult);  
        testLabelSet(i,:) = I;  
    end  
end  