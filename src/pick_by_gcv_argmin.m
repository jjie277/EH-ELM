function [Lstar, pick_src] = pick_by_gcv_argmin(S, L_list, GCV_vals, src_name)
if isempty(S)
    [~,i] = min(GCV_vals);
    Lstar = L_list(i);
    pick_src = [src_name,'_global'];
    return;
end
[tf, idx_all] = ismember(S, L_list);
vals = GCV_vals(idx_all(tf));
vmin = min(vals);
cand = S(tf);
cand = cand(vals == vmin);
Lstar = min(cand);
pick_src = src_name;
end
