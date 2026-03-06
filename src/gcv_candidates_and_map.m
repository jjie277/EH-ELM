function [C_gcv, L_list, GCV_vals, best_lams] = gcv_candidates_and_map(H, T, order_greedy, L_pool, SHR, tau)
L_list = L_pool(:)'; 
GCV_vals = zeros(size(L_list)); 
best_lams = zeros(size(L_list));
for ii = 1:numel(L_list)
    L = L_list(ii);
    Hsub = H(:, order_greedy(1:L));
    [lam, gcv] = ridge_lambda_gcv_val(Hsub, T, SHR);
    best_lams(ii) = lam;
    GCV_vals(ii) = gcv;
end
gmin = min(GCV_vals);
C_gcv = L_list(GCV_vals <= (1+tau)*gmin);
end
