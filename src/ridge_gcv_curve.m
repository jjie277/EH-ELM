function [lam_grid, gcv_curve, df_curve] = ridge_gcv_curve(H, T, SHR)
[Q,~] = size(H);
[U,S,~] = svd(H,'econ'); s = diag(S); s2 = s.^2 + eps;
lam_min = max(min(s2)*1e-6, SHR.lambda_floor);
lam_max = max(s2)*1e+3;
lam_grid = logspace(log10(lam_min), log10(lam_max), SHR.lambda_grid_pts);
Z = U' * T;
T_perp = T - U * Z; Tperp2 = sum(T_perp(:).^2);
gcv_curve = zeros(size(lam_grid));
df_curve  = zeros(size(lam_grid));
for ii=1:numel(lam_grid)
    lam = lam_grid(ii);
    g = (s.^2) ./ (s.^2 + lam);
    R_in = U * ((1 - g) .* Z);
    R2   = sum(R_in(:).^2) + Tperp2;
    trA  = sum(g);
    gcv_curve(ii) = R2 / max((Q - trA)^2, eps);
    df_curve(ii)  = trA;
end
end
