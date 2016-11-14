load results_100u;
figure;  
hold on;
plot(rb(:,3), rb(:,2),  'k', 'LineWidth', 3);
plot(pb(:,3), pb(:,2),  'c', 'LineWidth', 3);
plot(cf(:,3), cf(:,2),  'r', 'LineWidth', 3);
plot(cb(:,3), cb(:,2),  'g', 'LineWidth', 3);
plot(cbcf_scbK(:,3), cbcf_scbK(:,2),  'b', 'LineWidth', 3);
plot(cbcf_scb3(:,3), cbcf_scb3(:,2),  'b:', 'LineWidth', 3);
plot(cbpb_rbK(:,3), cbpb_rbK(:,2),  'm', 'LineWidth', 3);
plot(cbpb_rb3(:,3), cbpb_rb3(:,2),  'm:', 'LineWidth', 3);
legend('RB', 'PB', 'CF', 'CB', 'CB+CF (SCB,K)', 'CB+CF (SCB,3)', 'CB+PB (RB,K)', 'CB+PB (RB,3)');
xlabel('Recall (%)', 'FontSize', 18);
ylabel('Precision (%)', 'FontSize', 18);

saveas(gcf, 'precision_recall_plot.eps', 'eps2c');
system(['epstopdf "precision_recall_plot.eps"']);
