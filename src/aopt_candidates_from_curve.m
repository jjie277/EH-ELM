function [C_aopt, L_seq, A_curve] = aopt_candidates_from_curve(H, order_greedy, Lmin, Lmax, lambda_ref, aopt_drop, add_neighbors, L_pool)
L_seq = Lmin:Lmax;
A_curve = zeros(size(L_seq));
for ii = 1:numel(L_seq)
    L = L_seq(ii);
    Hsub = H(:, order_greedy(1:L));
    s = svd(Hsub,'econ');
    A_curve(ii) = sum(1 ./ (s.^2 + lambda_ref));
end
dA = -diff(A_curve);
mx = max(dA); thr = aopt_drop * mx;
idx = find(dA <= thr, 1, 'first');
if isempty(idx), k0 = L_seq(end-1);
else, k0 = Lmin + idx;
end
a = max(Lmin, k0 - add_neighbors);
b = min(Lmax, k0 + add_neighbors);
C_aopt = intersect(L_pool(:)', (a:b));
end
