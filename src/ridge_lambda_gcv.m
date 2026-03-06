function lambda = ridge_lambda_gcv(H, T, SHR)
[Q,~] = size(H);
[U,S,~] = svd(H,'econ');
s = diag(S); s2 = s.^2 + eps;
lam_min = max(min(s2)*1e-6, SHR.lambda_floor);
lam_max = max(s2)*1e+3;
lam_grid = logspace(log10(lam_min), log10(lam_max), SHR.lambda_grid_pts);
Z = U' * T;
T_perp = T - U * Z; Tperp2 = sum(T_perp(:).^2);
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
end
