function showfigure(img)
    figure
    imshow(img.img_gray);
    hold on
    % plot(img.shape_gt(:, 1), img.shape_gt(:, 2), 'g+');
    rectangle('Position', img.bbox_gt, 'EdgeColor', 'y');
    for i = 1: length(img.shape_gt)
        plot(img.shape_gt(i, 1), img.shape_gt(i, 2), 'r+');
        text(img.shape_gt(i, 1), img.shape_gt(i, 2), num2str(i));
    end
    hold off
end