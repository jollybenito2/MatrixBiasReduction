function [Z, dhat, tauhat]=non_lin_shrink(s2)
s2 = strrep(s2,'/','\');
s1 = 'C:\Users\52999\Desktop\MatrixOptim\';
s = strcat(s1,s2);
s3 = '.csv';
sA = strcat(s,s3);
A = readmatrix(sA);
%Preprocess
A(:,1) = []; %Remove first column
A(1,:) = []; %Remove first row
% B = table2array(A);
% B = detrend(b, 'constant');
% X = B{:,:};
[Z, dhat, tauhat] = QuESTimate(A, 1);
s4 = "_NonlinearShrink.csv";
sZ = strcat(s,s4);
csvwrite(sZ ,Z);
end
