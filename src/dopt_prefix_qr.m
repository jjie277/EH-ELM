function [perm_local, logdet_curve] = dopt_prefix_qr(H0, lambda)
[~, M] = size(H0);
X = [sqrt(max(lambda,0)) * eye(M); H0];
[~, R, p] = qr(X, 0, 'vector');
perm_local = p(:)';
rd = abs(diag(R)); rd(rd<=0) = eps;
logdet_curve = 2*cumsum(log(rd));
end
