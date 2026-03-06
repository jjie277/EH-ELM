function [S0, info, sc_sorted, cov_curve] = rls_screen(H, T, CFG)
[lambda_rls, lambda_src] = rls_pick_lambda(H, T, CFG);
[scores, dlam] = rls_scores(H, lambda_rls);
[sc_sorted, idx_sorted] = sort(scores, 'descend');
cov_curve = cumsum(sc_sorted) / max(dlam, eps);
[M, ~] = rls_choose_M(sc_sorted, dlam, size(H,1), size(H,2), CFG);

retries = 0;
while (M > CFG.rls.max_frac_N * size(H,2)) && (retries < CFG.rls.max_retries)
    lambda_rls = lambda_rls * CFG.rls.lambda_grow;
    [scores, dlam] = rls_scores(H, lambda_rls);
    [sc_sorted, idx_sorted] = sort(scores, 'descend');
    cov_curve = cumsum(sc_sorted) / max(dlam, eps);
    [M, ~] = rls_choose_M(sc_sorted, dlam, size(H,1), size(H,2), CFG);
    retries = retries + 1;
end

S0 = idx_sorted(1:M);
info = struct('lambda',lambda_rls,'src',lambda_src,'dlam',dlam,'M',M);
end
