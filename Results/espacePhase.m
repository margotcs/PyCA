
Nit = 1;
legendcells = strings(Nit,1)

figure 
hold on

%% set new colors :
% colormap('jet'); % or any other colormap you prefer
% colors = colormap; % get the colormap
% nColors = size(colors, 1);

% newcolors = zeros(Nit,3);
% for i=1
%     colorIndex = round((i-1) / (Nit-1) * (nColors-1)) + 1; % map i to a color index
%     newcolors(i,:) = colors(colorIndex, :);
% end

% colororder(newcolors)


for i=1
    %lifetimeI = 8+2*i
    %filename = ['mut_False_lifeTime_',num2str(lifetimeI),'.h5']
    filename = ['mut_False_lifeTime_',num2str(19),'.h5']
    gestationTime = h5read(filename,"/gestationTime");
    lifetime = h5read(filename,"/lifeTimeMax");
    totalCells = h5read(filename,"/totalCells"); totalCells = double(totalCells);
    healthyCells = h5read(filename,"/healthyCellsNumber"); healthyCells = double(healthyCells);
    infectedCells = h5read(filename,"/infectedCellsNumber"); infectedCells = double(infectedCells);
    taille = size(infectedCells); taille = taille(1)
    time = linspace(1, taille, taille);
    time = 10*time';
    
    % compute derivatives :
    infectedp = diff(infectedCells./(healthyCells + infectedCells))./diff(time);

    infectedCellsPlot = infectedCells./(healthyCells + infectedCells);
    infectedCellsPlot = infectedCellsPlot(1:end-1)
    %legendcells(i,1) = strcat("t_{life} / t_{repro} = ", num2str(lifetime));
    legendcells(i,1) = num2str(lifetime);

    plot(infectedCellsPlot, infectedp, 'o-', 'LineWidth',2, 'MarkerSize',3);
    test=1;
end
 
title("Infected rodents proportion versus lifetime", 'interpreter', 'tex', 'FontWeight', 'normal');
legend(legendcells);  % legendcell est un array de character ou de strings
xlabel('infected ')
ylabel('infected derivee')