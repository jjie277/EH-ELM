function C_det = det_candidates_from_dlog(dlog, Lmin, Lmax, elbow_drop, add_neighbors, L_pool)
C_det = [];
seg = dlog(Lmin:Lmax);
if isempty(seg), return; end
mx  = max(seg); thr = elbow_drop * mx;
elbow_rel = find(seg <= thr, 1, 'first');
if isempty(elbow_rel), return; end
k0 = Lmin + elbow_rel - 1;
a = max(Lmin, k0 - add_neighbors);
b = min(Lmax, k0 + add_neighbors);
C_det = intersect(L_pool(:)', (a:b));
end
