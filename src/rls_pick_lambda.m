function [lambda_rls, src] = rls_pick_lambda(H, T, CFG)
switch lower(CFG.rls.lambda_mode)
    case 'gcv'
        lambda_rls = ridge_lambda_gcv(H, T, CFG.shrinkage); src='gcv';
    case 'heuristic'
        s = svd(H,'econ'); s2 = s.^2;
        lambda_rls = max(median(s2)/100, 1e-12); src='heuristic';
    case 'fixed'
        lambda_rls = CFG.rls.lambda_fixed; src='fixed';
    otherwise
        lambda_rls = ridge_lambda_gcv(H, T, CFG.shrinkage); src='gcv';
end
end
