%% Housekeeping
clear; clc; close all;

%% Load data
% Assumption: "data" is stored in data.mat and contains [features ... target].
% The original code used: data = data'; so we keep that behavior.
load data
data = data';                          % ensure samples are in rows (Q x (R+1))

X = data(:, 1:end-1);                  % features (Q x R)
y = data(:, end);                      % target   (Q x 1)
numFeatures = size(X, 2);

%% Hyperparameters
N_neuron = 100;                        % number of hidden neurons
k_count  = 5;                          % K-folds

% Optional: fix randomness for reproducibility (recommended)
rng(1);

%% Prepare K-fold split
Q = size(X, 1);                        % number of samples
indices = crossvalind('Kfold', Q, k_count);

% Metrics per fold:
% RMSE_test
% r_test
% MAE_test
% RPD_test
% RPIQ_test
% max_rel_err_test
% min_rel_err_test
% RMSE_train
% r_train
% MAE_train
% RPD_train
% RPIQ_train

k_metric = zeros(k_count, 12);

% (Optional) store each fold model if you want later inspection
fold_models = cell(k_count, 1);

%% K-fold cross-validation
for k_i = 1:k_count
    fprintf('\n=== Fold %d / %d ===\n', k_i, k_count);

    % Split train/test
    isTest  = (indices == k_i);
    isTrain = ~isTest;

    X_train = X(isTrain, :);
    y_train = y(isTrain, :);
    X_test  = X(isTest,  :);
    y_test  = y(isTest,  :);

    % ----------------------------
    % Normalization (fit on train)
    % ----------------------------
    % Feature normalization: mapminmax expects column-wise variables, so we transpose twice.
    [Xn_train, inputps] = mapminmax(X_train', -1, 1);
    Xn_train = Xn_train';
    Xn_test  = mapminmax('apply', X_test', inputps)';
    
    % Target normalization (single column)
    [yn_train, outputps] = mapminmax(y_train', -1, 1);
    yn_train = yn_train';
    yn_test  = mapminmax('apply', y_test', outputps)';

    % ----------------------------
    % Train ELM once (no repeats)
    % ----------------------------
    % If your elmtrain has a plot switch by "i", we pass i=0 to keep it quiet.
    run_id = 0;
    evalc('[IW,B,LW,TF,TYPE] = eh_elmtrain(Xn_train, yn_train, N_neuron, ''sig'', 0, run_id);');

    % ----------------------------
    % Predict (normalized)
    % ----------------------------
    yn_test_pred  = eh_elmpredict(Xn_test,  IW, B, LW, TF, TYPE);
    yn_train_pred = eh_elmpredict(Xn_train, IW, B, LW, TF, TYPE);

    % ----------------------------
    % Inverse normalization
    % ----------------------------
    y_test_pred  = mapminmax('reverse', yn_test_pred',  outputps)';   % (Qtest x 1)
    y_train_pred = mapminmax('reverse', yn_train_pred', outputps)';   % (Qtrain x 1)

    % ----------------------------
    % Compute metrics
    % ----------------------------
    mTest  = regression_metrics(y_test,  y_test_pred);
    mTrain = regression_metrics(y_train, y_train_pred);

    % Store fold metrics
    k_metric(k_i, :) = [
        mTest.RMSE,  mTest.r,  mTest.MAE,  mTest.RPD,  mTest.RPIQ,  mTest.maxRelErr,  mTest.minRelErr, ...
        mTrain.RMSE, mTrain.r, mTrain.MAE, mTrain.RPD, mTrain.RPIQ, mTrain.maxRelErr, mTrain.minRelErr
    ];

    % (Optional) store fold model
    model = struct();
    model.IW = IW; model.B = B; model.LW = LW;
    model.TF = TF; model.TYPE = TYPE;
    model.inputps = inputps;
    model.outputps = outputps;
    model.fold_index = k_i;
    fold_models{k_i} = model;

    fprintf('Fold %d | Test RMSE=%.4f, r=%.4f | Train RMSE=%.4f, r=%.4f\n', ...
        k_i, mTest.RMSE, mTest.r, mTrain.RMSE, mTrain.r);
end

%% Report K-fold mean results
kmean_metric = mean(k_metric, 1);

E_test      = kmean_metric(1);
r_test      = kmean_metric(2);
MAE_test    = kmean_metric(3);
RPD_test    = kmean_metric(4);
RPIQ_test   = kmean_metric(5);
maxErr_test = kmean_metric(6);
minErr_test = kmean_metric(7);

E_train      = kmean_metric(8);
r_train      = kmean_metric(9);
MAE_train    = kmean_metric(10);
RPD_train    = kmean_metric(11);
RPIQ_train   = kmean_metric(12);


fprintf('\n===== K-fold Mean Results =====\n');
fprintf('Test : RMSE=%.4f, r=%.4f, MAE=%.4f, RPD=%.4f, RPIQ=%.4f\n', E_test,  r_test,  MAE_test,  RPD_test,  RPIQ_test);
fprintf('Train: RMSE=%.4f, r=%.4f, MAE=%.4f, RPD=%.4f, RPIQ=%.4f\n', E_train, r_train, MAE_train, RPD_train, RPIQ_train);

% Optional: save summary
% save('EH_ELM_cv_summary.mat', 'k_metric', 'kmean_metric', 'fold_models');
