function [S_use, sel_info] = select_width_subset(H, T, S0, rls_info, CFG)
switch lower(CFG.dopt.lambda_mode)
    case 'from_rls',   lambda_d = rls_info.lambda; src_d = 'rls';
    case 'gcv_on_S0',  lambda_d = ridge_lambda_gcv(H(:,S0), T, CFG.shrinkage); src_d='gcv_on_S0';
    case 'fixed',      lambda_d = CFG.dopt.lambda_fixed; src_d = 'fixed';
    otherwise,         lambda_d = rls_info.lambda; src_d = 'rls';
end

[perm_local, logdet_curve] = dopt_prefix_qr(H(:,S0), lambda_d);
order_greedy = S0(perm_local);

dlam = rls_info.dlam;
M = numel(S0);
Lmin = max(CFG.dopt.min_L, min(M, ceil(0.5*max(1,round(dlam)))));
Lmax = min(M, floor(CFG.rls.clip_frac_Q * size(H,1)));
[~, L_pool] = build_width_pool_two_stage(Lmin, Lmax, dlam, logdet_curve, CFG);

% Three-Judges (compact)
lc = logdet_curve(:)'; dlog = [lc(1), diff(lc)];
C_det = det_candidates_from_dlog(dlog, Lmin, Lmax, CFG.dopt.elbow_drop, CFG.select.add_neighbors, L_pool);
[C_aopt, L_seq, A_curve] = aopt_candidates_from_curve(H, order_greedy, Lmin, Lmax, lambda_d, CFG.select.aopt_drop, CFG.select.add_neighbors, L_pool); %#ok<ASGLU>
[C_gcv, L_gcv, GCV_vals, best_lams] = gcv_candidates_and_map(H, T, order_greedy, L_pool, CFG.shrinkage, CFG.select.gcv_tau); %#ok<ASGLU>

Cup = union(C_det, C_aopt);
Cinter = intersect(C_gcv, Cup);
if ~isempty(Cinter)
    [Lstar, pick_src] = pick_by_gcv_argmin(Cinter, L_gcv, GCV_vals, 'intersect');
else
    [Lstar, pick_src] = pick_by_gcv_argmin(C_gcv, L_gcv, GCV_vals, 'fallback');
end

S_use = order_greedy(1:Lstar);
sel_info = struct('Lstar',Lstar,'lambda_d',lambda_d,'src',src_d,'pick_src',pick_src,'Lmin',Lmin,'Lmax',Lmax);
end
