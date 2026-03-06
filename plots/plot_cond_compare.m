function plot_cond_compare(cond_full, cond_sub, cond_reg, cond_eff, bestlam)
    figure('Name','Comparison chart of condition numbers');
    cats = categorical({'Full','Subset','Shrink','Eff(g\cdot s)'});
    cats = reordercats(cats, cellstr(cats));   
    vals = [cond_full, cond_sub, cond_reg, cond_eff];
    plot(cats, vals, '-^', ...
    'Color',[230 119 0]/255, 'LineWidth',2, ...
    'MarkerSize',7, 'MarkerFaceColor','w');
    grid on; box on;
    set(gca, 'YScale', 'log');                                   
    ylabel('Condition number \kappa_2 (log scale)');
%     title(sprintf('Comparison chart of condition numbers（\\lambda^*=%.2e）', bestlam), 'FontWeight','bold');
    set(gca,'FontSize',11);
end