function LW = diag_scale_inverse(LW_scaled, scale_vec, S_use, CFG)
if isfield(CFG,'diag_scale') && CFG.diag_scale.enable
    dsub = scale_vec(S_use);
    LW = bsxfun(@times, dsub(:), LW_scaled);
else
    LW = LW_scaled;
end
end
