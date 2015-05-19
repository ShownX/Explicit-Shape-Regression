function pro_shape = projectShape(pts, bbx)
% x, y, w, h
centroid_x = bbx(1) + bbx(3)/2;
centroid_y = bbx(2) + bbx(4)/2;

pro_shape = zeros(size(pts));
pro_shape(:,1) = (pts(:,1) - centroid_x)/(bbx(3)/2);
pro_shape(:,2) = (pts(:,2) - centroid_y)/(bbx(4)/2);
end