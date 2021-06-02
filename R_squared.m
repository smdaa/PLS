function [R_squared] = R_squared(Y, Y_fitted)
R_squared = 1 - sum((Y - Y_fitted).^2) / sum((Y - mean(Y)).^2);
end
