function data_aug = augmtdata(data, params)
    data_len = length(data);
    data_aug = cell(params.N_aug*data_len, 1);
    
    
    for i = 1: data_len
        % random select initial shape without replacement
        rand_index_ = randperm(data_len, params.N_aug);
        while ismember(i, rand_index_) % if rand_index contain i, rand select again
            rand_index_ = randperm(data_len, params.N_aug);
        end
        % expand the data
        for j = 1: params.N_aug
            r_index = rand_index_(j);
            % copy the original stuff
            data_index = (i-1)*params.N_aug + j;
            data_aug{data_index}.img_gray = data{i}.img_gray;
            data_aug{data_index}.width_orig  = data{i}.width_orig;
            data_aug{data_index}.height_orig = data{i}.height_orig;    
            data_aug{data_index}.width       = data{i}.width;
            data_aug{data_index}.height      = data{i}.height;
            data_aug{data_index}.shape_gt    = data{i}.shape_gt;
            data_aug{data_index}.bbox_gt        = data{i}.bbox_gt;
            % add the new stuff
            data_aug{data_index}.intermediate_shapes = cell(1, params.T);
            data_aug{data_index}.intermediate_bboxes = cell(1, params.T);
            % scale and translate the sampled shape to ground-truth
            % face rectangle region
            select_shape = resetshape(data{i}.bbox_gt, [data{r_index}.shape_gt]);

            data_aug{data_index}.intermediate_shapes{1} = select_shape;
            data_aug{data_index}.intermediate_bboxes{1} = getbbox(select_shape);

            meanshape_resize = resetshape(data_aug{data_index}.intermediate_bboxes{1}, params.meanshape);

            data_aug{data_index}.tf2meanshape = fitgeotrans(bsxfun(@minus, ...
                data_aug{data_index}.intermediate_shapes{1}, mean(data_aug{data_index}.intermediate_shapes{1})), ...
                bsxfun(@minus, meanshape_resize, mean(meanshape_resize)),...
                'nonreflectivesimilarity');
            data_aug{data_index}.meanshape2tf = fitgeotrans(bsxfun(@minus, meanshape_resize, mean(meanshape_resize)), ...
                bsxfun(@minus, data_aug{data_index}.intermediate_shapes{1}, mean(data_aug{data_index}.intermediate_shapes{1})),...
                'nonreflectivesimilarity');                

            shape_residual = bsxfun(@rdivide, data_aug{data_index}.shape_gt - data_aug{data_index}.intermediate_shapes{1},...
                [data_aug{data_index}.intermediate_bboxes{1}(3) data_aug{data_index}.intermediate_bboxes{1}(4)]);  

            [u, v] = transformPointsForward(data_aug{data_index}.tf2meanshape, shape_residual(:, 1), shape_residual(:, 2));
            data_aug{data_index}.shapes_residual = [u, v];
        end
    end
    
end