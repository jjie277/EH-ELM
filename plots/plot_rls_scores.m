function plot_rls_scores(sc_sorted, M, lambda_rls, coverageM, alpha_bg)
    % alpha_bg （0~1），
    if nargin < 5 || isempty(alpha_bg), alpha_bg = 0.38; end

    figure('Name','RLS scores');

    plot(sc_sorted, '-o', 'Color', [0.75 0.60 0.00], ...
        'LineWidth', 1.8, 'MarkerSize', 0.2, ...
        'MarkerFaceColor', [0.95 0.82 0.35]);

    hold on; grid on;

    xline(M, '--', 'LineWidth', 0.8, 'Color', [0.3 0.3 0.3]);

    yLimits = ylim;
    yMid = mean(yLimits);
    text(M + 0.02 * numel(sc_sorted), yMid, '\it{M}', ...
        'FontSize', 14, 'FontAngle', 'italic', ...
        'VerticalAlignment', 'middle');

    xlabel('Feature Index (sorted)');
    ylabel('RLS Score');
%     title(sprintf('RLS scores（\\lambda=%.2e, coverage=%.2f）', ...
%         lambda_rls, coverageM), 'FontWeight', 'bold');
    set(gca, 'FontSize', 11);
end