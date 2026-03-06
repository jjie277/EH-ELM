function [M, why] = rls_choose_M(sc_sorted, dlam, Q, N, CFG)
cov = cumsum(sc_sorted) / max(dlam, eps);
idx_cov = find(cov >= CFG.rls.coverage, 1, 'first');
if isempty(idx_cov), idx_cov = numel(sc_sorted); end
M_mul = ceil(CFG.rls.multiplier * dlam);
M = min(idx_cov, M_mul);
if ~isempty(CFG.rls.Lstar_guess), M = max(M, CFG.rls.Lstar_guess); end
M = min(M, min(N, floor(CFG.rls.clip_frac_Q * Q)));
why = sprintf('min(coverage=%.2f@%d, multiplier=%d)', CFG.rls.coverage, idx_cov, M_mul);
end
