function [IW,B,LW,TF,TYPE] = eh_elmtrain(P,T,N,TF,TYPE)
% ELMTRAIN  Create and train an Extreme Learning Machine (samples-in-rows)

fprintf('[elmtrain DEBUG] %s\n', mfilename('fullpath'));

% ---------------- CONFIG ----------------
CFG = eh_elmtrain_default_config();
CFG.plots.enable = true;
sel_shrink = 1;

% ----------------------------------------

if CFG.paper_mode
    CFG.debug_level = 0;
    CFG.ablation.compare_pinv = false;
end

% ---------- arg checks ----------
if nargin < 2, error('ELM:Arguments','Not enough input arguments.'); end
if nargin < 3, N = size(P,1); end
if nargin < 4, TF = 'sig'; end
if nargin < 5, TYPE = 0; end
if size(P,1) ~= size(T,1)
    error('ELM:Arguments','Rows (samples) of P and T must match.');
end
[Q,R] = size(P);

% targets / one-hot
if TYPE==1
    if isrow(T), idx = T; else, idx = T'; end
    T = full(ind2vec(idx))';
end
[Q,S] = size(T); %#ok<ASGLU>

% ============= RANDOM HIDDEN =============
IW = rand(R,N)*2 - 1;     % R x N
B  = rand(1,N);           % 1 x N

% ============= FORWARD H =============
BiasMatrix = repmat(B, Q, 1);      % Q x N
tempH = P * IW + BiasMatrix;       % Q x N
switch TF
    case 'sig',     H = 1 ./ (1 + exp(-tempH));
    case 'sin',     H = sin(tempH);
    case 'hardlim'
        if exist('hardlim','file')==2
            H = hardlim(tempH);
        else
            H = double(tempH >= 0);
        end
    otherwise,      error('ELM:Arguments','Unknown transfer function: %s', TF);
end

% =============  DIAGONAL SCALING =============
scale_vec = ones(1, size(H,2));
if isfield(CFG,'diag_scale') && CFG.diag_scale.enable
    switch lower(CFG.diag_scale.mode)
        case 'col_l2'
            col_norm = sqrt(sum(H.^2,1));
            col_norm_safe = max(col_norm, CFG.diag_scale.eps);
            scale_vec = 1 ./ col_norm_safe;
            H = H .* repmat(scale_vec, size(H,1), 1);
            clipped = sum(col_norm < CFG.diag_scale.eps);
            logf(CFG,1,'[DiagScale] mode=col_l2 eps=%.1e | #clipped=%d\n', CFG.diag_scale.eps, clipped);
        case 'diag_gram'
            col_energy = sum(H.^2,1);           % diag(H'*H)
            col_l2 = sqrt(col_energy);
            col_l2_safe = max(col_l2, CFG.diag_scale.eps);
            scale_vec = 1 ./ col_l2_safe;
            H = H .* repmat(scale_vec, size(H,1), 1);
            clipped = sum(col_l2 < CFG.diag_scale.eps);
            logf(CFG,1,'[DiagScale] mode=diag_gram eps=%.1e | #clipped=%d\n', CFG.diag_scale.eps, clipped);
        otherwise
            logf(CFG,1,'[DiagScale] unknown mode=%s (skip)\n', CFG.diag_scale.mode);
    end
else
    logf(CFG,1,'[DiagScale] disabled\n');
end

S_use = 1:N; idx_sorted = (1:N)'; dlam = NaN; M = N;
if CFG.modules.rls
    [lambda_rls, lambda_src] = rls_pick_lambda(H, T, CFG);
    [scores, dlam] = rls_scores(H, lambda_rls);
    [sc_sorted, idx_sorted] = sort(scores, 'descend');
    cov_curve = cumsum(sc_sorted) / max(dlam, eps);
    [M, whyM] = rls_choose_M(sc_sorted, dlam, Q, N, CFG);

    retries = 0;
    while (M > CFG.rls.max_frac_N * N) && (retries < CFG.rls.max_retries)
        lambda_rls = lambda_rls * CFG.rls.lambda_grow;
        [scores, dlam] = rls_scores(H, lambda_rls);
        [sc_sorted, idx_sorted] = sort(scores, 'descend');
        cov_curve = cumsum(sc_sorted) / max(dlam, eps);
        [M, whyM] = rls_choose_M(sc_sorted, dlam, Q, N, CFG);
        retries = retries + 1;
    end
    S0 = idx_sorted(1:M);
    logf(CFG,1,'[RLS] lambda=%.3e src=%s | d_lambda=%.2f | M=%d/%d | coverage(M)=%.3f\n', ...
        lambda_rls, lambda_src, dlam, M, N, cov_curve(M));
else
    S0 = (1:N)';
end
assert(numel(unique(S0))==numel(S0), '[RLS] S0 has duplicates!');

if isfield(CFG,'plots') && CFG.plots.enable
    try
        plot_rls_scores(sc_sorted, M, lambda_rls, cov_curve(M));
    catch ME
        logf(CFG,1,'[plot RLS] skipped: %s\n', ME.message);
    end
else
    logf(CFG,1,'[plot RLS] skipped by CFG.plots.enable=false\n');
end

if CFG.modules.dopt
    switch lower(CFG.dopt.lambda_mode)
        case 'from_rls',   lambda_d = lambda_rls; src_d = 'rls';
        case 'gcv_on_S0',  lambda_d = ridge_lambda_gcv(H(:,S0), T, CFG.shrinkage); src_d='gcv_on_S0';
        case 'fixed',      lambda_d = CFG.dopt.lambda_fixed; src_d = 'fixed';
        otherwise,         lambda_d = lambda_rls; src_d = 'rls';
    end

    [perm_local, logdet_curve] = dopt_prefix_qr(H(:,S0), lambda_d);
    if numel(unique(perm_local)) ~= numel(perm_local) || any(sort(perm_local) ~= 1:numel(perm_local))
        perm_local = 1:numel(perm_local);
    end
    order_greedy = S0(perm_local);

    Lmin = max(CFG.dopt.min_L, min(M, ceil(0.5*max(1,round(dlam)))));
    Lmax = min(M, floor(CFG.rls.clip_frac_Q * Q));
    [L_coarse, L_pool] = build_width_pool_two_stage(Lmin, Lmax, dlam, logdet_curve, CFG);

    logf(CFG,1,'[Dopt] lambda=%.3e src=%s | Lmin=%d Lmax=%d | |L_pool|=%d\n', lambda_d, src_d, Lmin, Lmax, numel(L_pool));

    % diagnostics
    lc   = logdet_curve(:)';             % 1xM
    dlog = [lc(1), diff(lc)];            % marginal gains
    rd   = exp(0.5 * dlog);
    [rd_sorted, idx_rd] = sort(rd, 'descend'); %#ok<ASGLU>

    if CFG.dopt.save_curve && isfield(CFG,'plots') && CFG.plots.enable
        try
            save_marginal_gain_plot(dlog, CFG.dopt.curve_filename, CFG, Lmin, Lmax);
            logf(CFG,1,'[plot] saved marginal gain curve: %s\n', CFG.dopt.curve_filename);
        catch ME
            logf(CFG,1,'[plot] save failed: %s\n', ME.message);
        end
    end

    % ----- GCV core) -----
    if CFG.select.enable
        logf(CFG,1,'[Select] Three-Judges (GCV core)\n');

        % 1) Det-based candidates from elbow
        C_det = det_candidates_from_dlog(dlog, Lmin, Lmax, CFG.dopt.elbow_drop, CFG.select.add_neighbors, L_pool);

        % 2) A-opt candidates 
        [C_aopt, L_seq, A_curve] = aopt_candidates_from_curve(H, order_greedy, Lmin, Lmax, lambda_d, CFG.select.aopt_drop, CFG.select.add_neighbors, L_pool);

        % 3) C_gcv on L_pool
        [C_gcv, L_gcv, GCV_vals, best_lams] = gcv_candidates_and_map(H, T, order_greedy, L_pool, CFG.shrinkage, CFG.select.gcv_tau); %#ok<ASGLU>

        print_set(CFG, 'C_det',  C_det);
        print_set(CFG, 'C_aopt', C_aopt);
        print_set(CFG, 'C_gcv',  C_gcv);

        % union then intersect with C_gcv
        Cup = union(C_det, C_aopt);
        Cinter = intersect(C_gcv, Cup);
        if ~isempty(Cinter)
            [Lstar, pick_src] = pick_by_gcv_argmin(Cinter, L_gcv, GCV_vals, 'intersect(C_gcv, union(C_det,C_aopt))');
        else
            [Lstar, pick_src] = pick_by_gcv_argmin(C_gcv, L_gcv, GCV_vals, 'fallback C_gcv');
        end
        logf(CFG,1,'[Select] L*=%d via %s\n', Lstar, pick_src);

        if CFG.select.save_plots && isfield(CFG,'plots') && CFG.plots.enable
            try
                if CFG.select.plot_det
                    plot_det_with_marks(dlog, Lmin, Lmax, C_det, Lstar, CFG.select.det_filename);
                end
                if CFG.select.plot_aopt
                    plot_curve_with_marks(L_seq, A_curve, C_aopt, Lstar, 'A-opt vs L (lambda_ref from D-opt)', 'A-opt', CFG.select.aopt_filename);
                end
                if CFG.select.plot_gcv
                    ttl = sprintf('GCV vs L (tau = %.1f%%)', 100*CFG.select.gcv_tau);
                    plot_curve_with_marks(L_gcv, GCV_vals, C_gcv, Lstar, ttl, 'GCV', CFG.select.gcv_filename);
                end
            catch ME
                logf(CFG,1,'[plot] save failed: %s\n', ME.message);
            end
        end
    else
        % fallback: pick from pool
        if CFG.dopt.random_pick
            Lstar = L_pool(randi(numel(L_pool)));
            pick_src = 'random_in_L_pool';
        else
            Lstar = L_pool(end); pick_src = 'fallback_last';
        end
        logf(CFG,1,'[Select] L*=%d via %s\n', Lstar, pick_src);
    end

    % KD
    Yhat_teacher = [];
    if CFG.distill.enable
        L_teacher = Lmax;
        S_teacher = order_greedy(1:L_teacher);
        if CFG.distill.use_oof
            [Yhat_teacher, fold_lams] = teacher_oof_predict(H, S_teacher, T, CFG.distill.kfold, CFG); %#ok<ASGLU>
            logf(CFG,1,'[KD] OOF teacher alpha=%.2f L_teacher=%d K=%d\n', CFG.distill.alpha, L_teacher, CFG.distill.kfold);
        else
            [Yhat_teacher, INFO_teacher] = teacher_fit_predict(H, S_teacher, T, CFG);
            logf(CFG,1,'[KD] in-sample teacher alpha=%.2f L_teacher=%d\n', CFG.distill.alpha, L_teacher);
            if isfield(INFO_teacher,'lambda')
                logf(CFG,1,'[KD] teacher lambda*=%.3e\n', INFO_teacher.lambda);
            end
        end
        try
            if size(T,2)==1
                cc = corrcoef(Yhat_teacher(:), T(:)); cc = cc(1,2);
                logf(CFG,1,'[KD] corr(teacher,T)=%.3f\n', cc);
            end
        catch
        end
    end
    
    S_use = order_greedy(1:Lstar);
    if numel(unique(S_use)) ~= numel(S_use)
        S_use = unique(S_use,'stable');
        Lstar = numel(S_use);
        logf(CFG,1,'[subset] repaired duplicates, new L*=%d\n', Lstar);
    end
else
    S_use = S0;
end

if CFG.debug_level >= 1
    fprintf('[CHECK] L*=%d, |S_use|=%d, unique=%d\n', numel(S_use), numel(S_use), numel(unique(S_use)));
end

% =============  subsetting & sync IW/B =============
H_train = H;
if ~isequal(S_use, 1:N)
    H_train = H(:, S_use);
    IW      = IW(:, S_use);
    B       = B(:, S_use);
    logf(CFG,1,'[SUBSET] IW/B trimmed: R x N = %d x %d (L* = %d)\n', size(IW,1), size(IW,2), numel(S_use));
else
    logf(CFG,1,'[SUBSET] using full H\n');
end

LW_scaled = []; INFO = struct();
if sel_shrink == 1
    if exist('Yhat_teacher','var') && CFG.distill.enable && ~isempty(Yhat_teacher)
        T_alpha = (1-CFG.distill.alpha) * T + CFG.distill.alpha * Yhat_teacher;
        [LW_scaled, INFO] = readout_spectral_shrinkage(H_train, T_alpha, CFG);
        Yhat_student = H_train * LW_scaled;
        rmse_hard = sqrt( mean( sum( (Yhat_student - T).^2, 2) ) );
        rmse_toT  = sqrt( mean( sum( (Yhat_student - Yhat_teacher).^2, 2) ) );
        logf(CFG,1,'[KD Student] lambda*=%.3e | RMSE(T)=%.6f | RMSE(to teacher)=%.6f\n', INFO.lambda, rmse_hard, rmse_toT);
    else
        [LW_scaled, INFO] = readout_spectral_shrinkage(H_train, T, CFG);
    end
else
    if exist('Yhat_teacher','var') && CFG.distill.enable && ~isempty(Yhat_teacher)
        T_alpha = (1-CFG.distill.alpha) * T + CFG.distill.alpha * Yhat_teacher;
        [LW_scaled, INFO] = readout_pinv_baseline(H_train, T_alpha, CFG);
    else
        [LW_scaled, INFO] = readout_pinv_baseline(H_train, T, CFG);
    end
end

if isfield(CFG,'diag_scale') && CFG.diag_scale.enable
    dsub = scale_vec(S_use);
    LW   = bsxfun(@times, dsub(:), LW_scaled);
    logf(CFG,1,'[DiagScale] inverse mapping applied on LW\n');
else
    LW = LW_scaled;
end

if CFG.rls.show_geometry_cmp && exist('INFO','var') && isfield(INFO,'lambda') && ~isequal(S_use,1:N)
    geom_compare(H, H_train, 'lambda*', INFO.lambda, CFG);
end

if isfield(CFG,'plots') && CFG.plots.enable
    try
        %  GCV vs lambda (lambda*)
        [lam_grid, gcv_curve, df_curve] = ridge_gcv_curve(H_train, T, CFG.shrinkage);
        plot_gcv_curve(lam_grid, gcv_curve, INFO.lambda);

        % 
        [df_pinv, df_ridge, df_hard, df_soft, s_full, s_sub, s_eff_ridge, ...
            smin_full, smin_sub, smin_eff, smin_reg, s_reg] = ...
            shrinkage_dof_and_spectrum(H, H_train, T, INFO.lambda, CFG);

        % pinv ——
        plot_df_compare(df_pinv, df_ridge, df_hard, df_soft, INFO.lambda);

        plot_spectrum_compare(s_full, s_sub, s_eff_ridge, INFO.lambda);
        plot_spectrum_compare_reg(s_full, s_sub, s_reg, INFO.lambda);

        plot_sigma_min_compare(smin_full, smin_sub, smin_reg, smin_eff, INFO.lambda);

        cond_full = max(s_full) / max(min(s_full), eps);
        cond_sub  = max(s_sub)  / max(min(s_sub),  eps);
        cond_reg  = max(s_reg)  / max(min(s_reg),  eps);  % sqrt(s^2+lambda*)
        pos_eff   = s_eff_ridge(s_eff_ridge > 0);
        if isempty(pos_eff)
            cond_eff = Inf;
        else
            cond_eff = max(pos_eff) / max(min(pos_eff), eps);
        end
        plot_cond_compare(cond_full, cond_sub, cond_reg, cond_eff, INFO.lambda);
        plot_cond_compare_wueff(cond_full, cond_sub, cond_reg, INFO.lambda);
        plot_df_curve(lam_grid, df_curve, INFO.lambda);
    catch ME
        logf(CFG,1,'[plot post-readout] skipped: %s\n', ME.message);
    end
else
    logf(CFG,1,'[plot post-readout] skipped by CFG.plots.enable=false\n');
end

% ---------- return ----------
end

% ================== helpers ==================
function logf(CFG, lvl, varargin)
    if CFG.debug_level >= lvl
        fprintf(varargin{:});
    end
end

function print_set(CFG, name, Lset)
    logf(CFG,1,'%s = {', name);
    for i=1:numel(Lset)
        logf(CFG,1,'%d', Lset(i));
        if i<numel(Lset), logf(CFG,1,', '); end
    end
    logf(CFG,1,'}  (|%s|=%d)\n', name, numel(Lset));
end

function S = geom_stats(H, lambda_ref)
    [~,Sv,~] = svd(H,'econ'); s = diag(Sv);
    condH  = s(1) / max(s(end), eps);
    sigmin = s(end);
    trA    = sum( (s.^2) ./ (s.^2 + lambda_ref) );
    Aopt   = sum( 1 ./ (s.^2 + lambda_ref) );
    S = struct('condH',condH,'sigmin',sigmin,'trA',trA,'Aopt',Aopt);
end

function geom_compare(H_full, H_sub, tag, lambda_ref, CFG)
    Sf = geom_stats(H_full, lambda_ref);
    Ss = geom_stats(H_sub,  lambda_ref);
    dC = Ss.condH  - Sf.condH;
    dS = Ss.sigmin - Sf.sigmin;
    dT = Ss.trA    - Sf.trA;
    dV = Ss.Aopt   - Sf.Aopt;
    logf(CFG,1,'[Geom @%s] lambda=%.3e\n', tag, lambda_ref);
    logf(CFG,1,'  cond(H) : Full=%.2e | Sub=%.2e | d=%.2e\n', Sf.condH,  Ss.condH,  dC);
    logf(CFG,1,'  sigma_min: Full=%.2e | Sub=%.2e | d=%.2e\n', Sf.sigmin, Ss.sigmin, dS);
    logf(CFG,1,'  tr(A)   : Full=%.2f | Sub=%.2f | d=%.2f\n', Sf.trA, Ss.trA, dT);
    logf(CFG,1,'  A-opt   : Full=%.2f | Sub=%.2f | d=%.2f\n', Sf.Aopt,  Ss.Aopt,  dV);
end

function [lambda_rls, src] = rls_pick_lambda(H, T, CFG)
    switch lower(CFG.rls.lambda_mode)
        case 'gcv'
            lambda_rls = ridge_lambda_gcv(H, T, CFG.shrinkage); src = 'gcv';
        case 'heuristic'
            s = svd(H,'econ'); s2 = s.^2;
            lambda_rls = max(median(s2)/100, 1e-12); src = 'heuristic';
        case 'fixed'
            lambda_rls = CFG.rls.lambda_fixed; src = 'fixed';
        otherwise
            lambda_rls = ridge_lambda_gcv(H, T, CFG.shrinkage); src = 'gcv';
    end
end

function [scores, dlam] = rls_scores(H, lambda)
    [~,S,V] = svd(H,'econ');
    s = diag(S); s2 = s.^2 + eps;
    w = (s2 - eps) ./ (s2 + lambda);
    VW = V .* sqrt(w(:)');             % N x r
    scores = sum(VW.^2, 2);            % N x 1
    dlam = sum(w);                     % effective dimension
end

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

function lambda = ridge_lambda_gcv(H, T, SHR)
    [Q,~] = size(H);
    [U,S,~] = svd(H,'econ');
    s = diag(S); s2 = s.^2 + eps;
    lam_min = max(min(s2)*1e-6, SHR.lambda_floor);
    lam_max = max(s2)*1e+3;
    lam_grid = logspace(log10(lam_min), log10(lam_max), SHR.lambda_grid_pts);

    Z = U' * T;
    T_perp = T - U * Z; Tperp2 = sum(T_perp(:).^2);

    best = Inf; bestlam = lam_grid(1);
    for lam = lam_grid
        g = (s.^2) ./ (s.^2 + lam);
        R_in = U * ((1 - g) .* Z);
        R2   = sum(R_in(:).^2) + Tperp2;
        trA  = sum(g);
        GCV  = R2 / max((Q - trA)^2, eps);
        if GCV < best, best = GCV; bestlam = lam; end
    end
    lambda = bestlam;
end


function [perm_local, logdet_curve] = dopt_prefix_qr(H0, lambda)
    [~, M] = size(H0);
    X = [sqrt(max(lambda,0)) * eye(M); H0];   % (M+Q) x M
    try
        [~, R, p] = qr(X, 0, 'vector');
        perm_local = p(:)';                  % 1 x M
    catch
        [~, R, E] = qr(X, 0);
        [~, p] = max(abs(E), [], 1);
        perm_local = p(:)';
    end
    if numel(unique(perm_local)) ~= numel(perm_local) || any(sort(perm_local) ~= 1:M)
        perm_local = 1:M;
    end
    rd = abs(diag(R)); rd(rd<=0) = eps;
    logdet_curve = 2*cumsum(log(rd));
end


function i = argmin(v)
    [~,i] = min(v);
end

function plot_det_with_marks(dlog, Lmin, Lmax, C_det, Lstar, filename)
    f = figure('Visible','off');
    k = 1:numel(dlog);
    yl = [min(dlog)-1, max(dlog)+1];
    h_patch = patch([Lmin Lmax Lmax Lmin], [yl(1) yl(1) yl(2) yl(2)], [0.9 0.9 0.9], ...
        'EdgeColor','none'); hold on; uistack(h_patch,'bottom');
    plot(k, dlog, '-o', 'LineWidth', 1.5, 'MarkerSize', 4); grid on;
    xlabel('k (prefix length)'); ylabel('Delta logdet');
    title('Delta logdet (det-based candidates)');
    line([Lmin Lmin], yl, 'LineStyle','--', 'Color',[0.3 0.3 0.3]);
    line([Lmax Lmax], yl, 'LineStyle','--', 'Color',[0.3 0.3 0.3]);
    if ~isempty(C_det), scatter(C_det, dlog(C_det), 36, 'filled'); end
    if ~isempty(Lstar)
        scatter(Lstar, dlog(Lstar), 64, 'd', 'filled');
        text(Lstar, dlog(Lstar), '  L*', 'VerticalAlignment','bottom');
    end
    try, saveas(f, filename); catch, end
    close(f);
end
