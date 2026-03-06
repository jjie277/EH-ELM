function plot_spectrum_compare(s_full, s_sub, s_eff_ridge, bestlam)
    figure('Name','Comparison of Singular Spectrum Distribution');
    semilogy(s_full,'-','LineWidth',1.6); hold on;
    semilogy(s_sub,'-','LineWidth',1.6);
    semilogy(s_eff_ridge,'-','LineWidth',1.6);
    grid on; xlabel('Index'); ylabel('Singular value (log sacle)');
    legend({'Full H','Sub H_f','Shrinked H_f^{eff} '},'Location','southwest');
    set(gca,'FontSize',11);
end