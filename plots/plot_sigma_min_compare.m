function plot_sigma_min_compare(smin_full, smin_sub, smin_reg, smin_eff, bestlam)
    figure('Name','Minimum Singular Value Comparison ');
    cats = categorical({'Full H','Sub H_f','Reg(H_f)=\surd(s^2+\lambda)','Eff(g\cdot s)'});
    vals = [smin_full, smin_sub, smin_reg, smin_eff];
    bar(cats, vals); grid on;
    ylabel('\sigma_{min}');
    set(gca,'FontSize',11);
end