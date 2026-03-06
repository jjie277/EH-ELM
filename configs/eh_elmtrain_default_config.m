function CFG = eh_elmtrain_default_config()
%ELMTRAIN_DEFAULT_CONFIG Default configuration for EH-ELM training pipeline.

CFG.paper_mode = false;
CFG.debug_level = 1;
CFG.ablation.compare_pinv = false;

% readout shrinkage
CFG.shrinkage.method = 'ridge_gcv';    
CFG.shrinkage.lambda_grid_pts = 60;
CFG.shrinkage.lambda_floor    = 1e-12;
CFG.shrinkage.lambda0         = 1e-3;

% RLS
CFG.modules.rls = true;
CFG.rls.lambda_mode = 'gcv';           
CFG.rls.lambda_fixed = 1e-3;
CFG.rls.coverage = 0.95;
CFG.rls.multiplier = 5;
CFG.rls.max_frac_N = 0.8;
CFG.rls.lambda_grow = 3;
CFG.rls.max_retries = 2;
CFG.rls.clip_frac_Q = 0.9;
CFG.rls.Lstar_guess = [];
CFG.rls.save_selection = false;
CFG.rls.save_tag = '';
CFG.rls.show_geometry_cmp = true;

% D-opt & width pool
CFG.modules.dopt = true;
CFG.dopt.lambda_mode = 'from_rls';     % 'from_rls' 
CFG.dopt.lambda_fixed = 1e-2;
CFG.dopt.elbow_drop = 0.5;
CFG.dopt.min_L = 5;
CFG.dopt.add_neighbors = 3; 
CFG.dopt.random_pick = true;
CFG.dopt.save_curve = false;
CFG.dopt.curve_filename = 'dopt_marginal_gain.png';

% Three-Judges
CFG.select.enable = true;
CFG.select.gcv_tau = 0.03;
CFG.select.aopt_drop = 0.5;
CFG.select.add_neighbors = 1;
CFG.select.save_plots = false;
CFG.select.plot_det = true;
CFG.select.plot_aopt = true;
CFG.select.plot_gcv = true;
CFG.select.det_filename  = 'sel_det.png';
CFG.select.aopt_filename = 'sel_aopt.png';
CFG.select.gcv_filename  = 'sel_gcv.png';

CFG.distill.enable    = true;
CFG.distill.use_oof   = true;
CFG.distill.kfold     = 5;
CFG.distill.alpha     = 0.1;
CFG.distill.log_detail = true;

% Diagonal scaling
CFG.diag_scale.enable = true;
CFG.diag_scale.mode   = 'diag_gram';   %  'diag_gram'
CFG.diag_scale.eps    = 1e-12;

% geometry compare
CFG.geom.trA_prefer = 'down';

% plots
CFG.plots.enable = true;
end
