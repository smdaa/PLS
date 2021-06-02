function e = RMSE(Y, Yh)
  z = Y(:)-Yh(:);
  e = sqrt(z'*z/numel(z));
end