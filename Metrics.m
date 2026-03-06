function M = Metrics(y_true, y_pred)
%REGRESSION_METRICS Compute regression evaluation metrics.
%   Outputs a struct with:
%     RMSE, r (Pearson correlation), MAE, SD, RPD, RPIQ,
%     maxRelErr, minRelErr

y_true = y_true(:);
y_pred = y_pred(:);

n = numel(y_true);
if n == 0
    error('Empty input to regression_metrics.');
end

% RMSE
err = y_true - y_pred;
RMSE = sqrt(mean(err.^2));

% Pearson correlation r (robust, no toolbox dependency)
yt = y_true - mean(y_true);
yp = y_pred - mean(y_pred);
den = sqrt(sum(yt.^2) * sum(yp.^2));
if den < eps
    r = NaN;
else
    r = (yt' * yp) / den;
end

% MAE
MAE = mean(abs(err));

% SD (population std, consistent with your original code)
SD = sqrt(mean((y_true - mean(y_true)).^2));

% RPD and RPIQ
if RMSE < eps
    RPD  = Inf;
    RPIQ = Inf;
else
    RPD = SD / RMSE;
    Q1 = prctile(y_true, 25);
    Q3 = prctile(y_true, 75);
    RPIQ = (Q3 - Q1) / RMSE;
end

% Relative error extrema (protect division by zero)
denom = y_true;
denom(abs(denom) < eps) = eps;
rel_err = (y_pred - y_true) ./ denom;
maxRelErr = max(rel_err);
minRelErr = min(rel_err);

M = struct( ...
    'RMSE', RMSE, ...
    'r', r, ...
    'MAE', MAE, ...
    'SD', SD, ...
    'RPD', RPD, ...
    'RPIQ', RPIQ, ...
    'maxRelErr', maxRelErr, ...
    'minRelErr', minRelErr);

end
