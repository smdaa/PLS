function [BetaPLS, Y_fitted] = PLS(Y, X, k)
    BetaPLS = SIMPLS(Y, X, k);
    BetaPLS = [mean(Y, 1) - mean(X, 1) * BetaPLS; BetaPLS];
    Y_fitted = [ones(size(X, 1),1) X] * BetaPLS;
end