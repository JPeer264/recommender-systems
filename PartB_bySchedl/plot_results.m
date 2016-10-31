load results;
figure;  
hold on;
plot(rb(:,3), rb(:,2),  'k', 'LineWidth', 3);
plot(cf(:,3), cf(:,2),  'r', 'LineWidth', 3);
plot(cb(:,3), cb(:,2),  'g', 'LineWidth', 3);
plot(hy(:,3), hy(:,2),  'b', 'LineWidth', 3);
legend('RB', 'CF', 'CB', 'HY');
xlabel('Recall (%)', 'FontSize', 18);
ylabel('Precision (%)', 'FontSize', 18);

saveas(gcf, 'precision_recall_plot.eps', 'eps2c');
system(['epstopdf "precision_recall_plot.eps"']);
