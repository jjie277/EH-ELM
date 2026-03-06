function [H, scale_vec] = diag_scale_apply(H, CFG)
scale_vec = ones(1, size(H,2));
if isfield(CFG,'diag_scale') && CFG.diag_scale.enable
    switch lower(CFG.diag_scale.mode)
        case 'col_l2'
            col_norm = sqrt(sum(H.^2,1));
            col_norm_safe = max(col_norm, CFG.diag_scale.eps);
            scale_vec = 1 ./ col_norm_safe;
            H = H .* repmat(scale_vec, size(H,1), 1);
        case 'diag_gram'
            col_energy = sum(H.^2,1);
            col_l2 = sqrt(col_energy);
            col_l2_safe = max(col_l2, CFG.diag_scale.eps);
            scale_vec = 1 ./ col_l2_safe;
            H = H .* repmat(scale_vec, size(H,1), 1);
    end
end
end
