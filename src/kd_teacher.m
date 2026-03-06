function [Yhat_teacher, info] = kd_teacher(H, T, S0, rls_info, sel_info, CFG) %#ok<INUSD>
L_teacher = numel(S0);
S_teacher = S0(1:L_teacher);
if CFG.distill.use_oof
    [Yhat_teacher, fold_lams] = teacher_oof_predict(H, S_teacher, T, CFG.distill.kfold, CFG); %#ok<ASGLU>
    info = struct('mode','oof','L_teacher',L_teacher);
else
    [Yhat_teacher, infoT] = teacher_fit_predict(H, S_teacher, T, CFG); %#ok<ASGLU>
    info = struct('mode','insample','L_teacher',L_teacher);
end
end
