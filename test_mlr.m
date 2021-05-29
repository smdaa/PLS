clc;
clear;
close all;

load accidents

%% simple linear regression
X = hwydata(:,5);
Y = hwydata(:,4);
[~, Y_fitted] = MLR(Y, X);
R_2 = R_squared(Y, Y_fitted);
fprintf('simple linear regression : R^2 = %f\n',R_2);
figure(1)
scatter(X, Y)
hold on
plot(X, Y_fitted)

%% multiple linear regression
X = hwydata(:,[10 14 15]);
Y = hwydata(:,4);
[~, Y_fitted] = MLR(Y, X);
R_2 = R_squared(Y, Y_fitted);
fprintf('multiple linear regression : R^2 = %f\n',R_2);
