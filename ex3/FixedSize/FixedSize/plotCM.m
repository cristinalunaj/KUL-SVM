%% PLOT - paint confussio matrix from M
n = 7
M = [11467,2,3,1,1,2,2;
    1,12,0,0,0,0,0;
    3,0,34, 0,1,0,1;
    0,0,0,2154,0,1,0;
    0,0,0,0,808,1,0;
    0,0,0,0,1,3,0;
    1,0,0,0,0,0,1];
    %11472,14,37,2155,811,7,4,14500];



imagesc(M);
colormap(flipud(gray));
textStrings = num2str(M(:));       % Create strings from the matrix values
textStrings = strtrim(cellstr(textStrings));  % Remove any space padding
[x, y] = meshgrid(1:n);  % Create x and y coordinates for the strings
hStrings = text(x(:), y(:), textStrings(:), ...  % Plot the strings
                'HorizontalAlignment', 'center');
midValue = mean(get(gca, 'CLim'));  % Get the middle value of the color range
textColors = repmat(M(:) > midValue, 1, 3);  % Choose white or black for the
                                               %   text color of the strings so
                                               %   they can be easily seen over
                                               %   the background color
set(hStrings, {'Color'}, num2cell(textColors, 2));  % Change the text colors

set(gca, 'XTick', 1:n, ...                             % Change the axes tick marks
         'XTickLabel', {'1', '2','3', '4', '5', '6','7','Total'}, ...  %   and tick labels '3', '4', '5', '6','7','8','9','spurious'
         'YTick', 1:n, ...
         'YTickLabel', {'1', '2','3', '4', '5', '6','7', 'Total'}, ...
         'TickLength', [0 0]);
xlabel('Predicted class');
ylabel('Real class');

     