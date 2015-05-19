function repro_shape = reprojectShape(pts, bbx)
% x, y, w, h
centroid_x = bbx(1) + bbx(3)/2;
centroid_y = bbx(2) + bbx(4)/2;

repro_shape = zeros(size(pts));
repro_shape(:,1) = pts(:,1)*(bbx(3)/2)+ centroid_x;
repro_shape(:,2) = pts(:,2)*(bbx(4)/2)+ centroid_y;
end