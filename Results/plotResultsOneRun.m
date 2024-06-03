
lifeTime = 20;
    filename = ['mut_False_lifeTime_',num2str(lifeTime),'.h5']
    gestationTime = h5read(filename,"/gestationTime");
    lifetime = h5read(filename,"/lifeTimeMax");
    totalCells = h5read(filename,"/totalCells"); totalCells = double(totalCells);
    healthyCells = h5read(filename,"/healthyCellsNumber"); healthyCells = double(healthyCells);
    infectedCells = h5read(filename,"/infectedCellsNumber"); infectedCells = double(infectedCells);
    taille = size(infectedCells); taille = taille(1)
    time = linspace(1, taille, taille);
    time = 10*time';

    figure
    plot(time, infectedCells./healthyCells, 'r', 'LineWidth', 2, 'MarkerSize',2);
    hold on
    plot(time, (infectedCells + healthyCells)./totalCells, 'k', 'LineWidth',2, 'MarkerSize',12);

 
%title(titre, 'interpreter', 'tex', 'FontWeight', 'normal');
%legend(legendcell);  % legendcell est un array de character ou de strings