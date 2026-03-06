function [LW, INFO] = readout_spectral_shrinkage(H, T, CFG)
[Q,~] = size(H);
[U,S,V] = svd(H,'econ'); s = diag(S);
Z = U' * T;
T_perp = T - U * Z; Tperp2 = sum(T_perp(:).^2);

s2 = s.^2 + eps;
lam_min = max(min(s2)*1e-6, CFG.shrinkage.lambda_floor);
lam_max = max(s2)*1e+3;
lam_grid = logspace(log10(lam_min), log10(lam_max), CFG.shrinkage.lambda_grid_pts);

best = Inf; bestlam = lam_grid(1);
for lam = lam_grid
    g = (s.^2) ./ (s.^2 + lam);
    R_in = U * ((1 - g) .* Z);
    R2   = sum(R_in(:).^2) + Tperp2;
    trA  = sum(g);
    GCV  = R2 / max((Q - trA)^2, eps);
    if GCV < best
        best = GCV; bestlam = lam;
    end
end
lambda = bestlam;
LW = V * diag( s ./ max(s.^2 + lambda, eps) ) * Z;
INFO = struct('method','ridge_gcv','lambda',lambda,'GCV',best);
end
