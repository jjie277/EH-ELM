function [Yhat_oof, fold_lams] = teacher_oof_predict(H, S_teacher, T, K, CFG)
Q = size(H,1);
Hteach_full = H(:, S_teacher);
Yhat_oof = zeros(size(T));
fold_lams = nan(K,1);
blocks = ceil((1:Q) * K / Q);
for k = 1:K
    val_mask = (blocks == k);
    tr_mask  = ~val_mask;
    Htr = Hteach_full(tr_mask, :);
    Hva = Hteach_full(val_mask, :);
    Ttr = T(tr_mask, :);
    cfgT = CFG; cfgT.debug_level = 0;
    [LW_T, INFO_T] = readout_spectral_shrinkage(Htr, Ttr, cfgT);
    if isfield(INFO_T,'lambda'), fold_lams(k) = INFO_T.lambda; end
    Yhat_oof(val_mask,:) = Hva * LW_T;
end
end
