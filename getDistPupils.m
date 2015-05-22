function dist_pupils = getDistPupils(shape_gt)
    num_lm = size(shape_gt, 1);
    switch num_lm
        case 68
            dist_pupils = norm((mean(shape_gt(37:42, :)) - mean(shape_gt(43:48, :))));
        case 51
            dist_pupils = norm((mean(shape_gt(20:25, :)) - mean(shape_gt(26:31, :))));
        case 29
            dist_pupils = norm((mean(shape_gt(9:2:17, :)) - mean(shape_gt(10:2:18, :))));
        otherwise
            disp('The landmark number in shape is not correct(68 or 51 or 29)');
    end
end