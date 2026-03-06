function [Yhat_T, INFO_T] = teacher_fit_predict(H, S_teacher, T, CFG)
Hteach = H(:, S_teacher);
cfgT = CFG; cfgT.debug_level = 0;
[LW_T, INFO_T] = readout_spectral_shrinkage(Hteach, T, cfgT);
Yhat_T = Hteach * LW_T;
end
