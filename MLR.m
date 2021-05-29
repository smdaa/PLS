function [Beta, Y_fitted] = MLR(Y, X)
    % 0n ajoute l'intercept
    X = [ones(length(X),1) X];
    
    % Beta = inv(X' * X) * X' * Y
    Beta = X \ Y;
    
    % fitted values
    Y_fitted = X * Beta;

end

