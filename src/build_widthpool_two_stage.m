function [L_coarse, L_pool] = build_widthpool_two_stage(Lmin, Lmax, dlam, logdet_curve, CFG)
lc = logdet_curve(:)'; M = numel(lc);
Lmin = max(1, min(Lmin, M));
Lmax = max(Lmin, min(Lmax, M));

cand = unique([round(0.5*dlam), round(0.75*dlam), round(1.0*dlam), round(1.25*dlam), ...
               round(1.5*dlam), round(1.75*dlam), round(2.0*dlam), Lmax]);
cand = cand(cand>=Lmin & cand<=Lmax);
if isempty(cand), cand = Lmax; end
L_coarse = unique(cand);

dlog = [lc(1), diff(lc)];
seg = dlog(Lmin:Lmax);
mx  = max(seg);
thr = CFG.dopt.elbow_drop * mx;
elbow_rel = find(seg <= thr, 1, 'first');
neigh = [];
if ~isempty(elbow_rel)
    elbow_idx = Lmin + elbow_rel - 1;
    a = max(Lmin, elbow_idx - CFG.dopt.add_neighbors);
    b = min(Lmax, elbow_idx + CFG.dopt.add_neighbors);
    neigh = a:b;
end
L_pool = unique([L_coarse(:); neigh(:)]);
L_pool = L_pool(L_pool>=Lmin & L_pool<=Lmax);
if isempty(L_pool), L_pool = L_coarse; end
end
