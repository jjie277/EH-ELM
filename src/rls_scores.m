function [scores, dlam] = rls_scores(H, lambda)
[~,S,V] = svd(H,'econ');
s = diag(S); s2 = s.^2 + eps;
w = (s2 - eps) ./ (s2 + lambda);
VW = V .* sqrt(w(:)');
scores = sum(VW.^2, 2);
dlam = sum(w);
end
