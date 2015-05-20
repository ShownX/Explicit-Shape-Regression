function ESR_Test()
    load('../../Data/toyData.mat', 'bbx_aug', 'pts_aug');
    load('../../Data/Testing_Data.mat', 'test_images', 'test_bbx', 'test_pts'); % load the current shapes
    load('../../Data/Model.mat', 'Model');
    params = Test_params;
    params.N_img = size(images_aug, 1);
    
    for i = 1: params.N_img
        image = test_images{i};
        bbx = test_bbx{i};
        Prediction = ShapeRegression(image, bbx, bbx_aug, pts_aug, Model, params);
    end
end

function ShapeRegression(image, bbx, bbx_aug, pts_aug, Model, params)
    % Multiple initializations
    N_train_img = size(pts_aug, 1);
    Index_init = randperm(N_train_img, params.N_init);
    for i = 1: params.N_init
        current_shape = pts_aug{Index_init(i)}.pts_chs;
        current_bbx = bbx_aug{Index_init(i)}.bbx_chs;
        current_shape = projectShape(current_shape, current_bbx);
        current_shape = reprojectShape(current_shape, bbx.bbx_chs);
        
        for t = 1: params.T
            prediction_delta = fernCascadeTest(image, current_shape, Model{t}.fernCascade, params);
            current_shape = prediction_delta + projectShape(current_shape, bbx.bbx_chs);
            current_shape = reprojectShape(current_shape, bbx.bbx_chs);
        end
    end
end

function prediction_delta = fernCascadeTest(image, current_shape, fernCascade, params)
    image.intermediate_bbx = getbbox(current_shape);
    meanshape = reprojectShape(params.mean_shape, image.intermediate_bbx);
    image.tf2meanshape = fitgeotrans(bsxfun(@minus, current_shape, mean(current_shape)), ...
                bsxfun(@minus, meanshape, mean(meanshape)),...
                'nonreflectivesimilarity');
    image.meanshape2tf = fitgeotrans( bsxfun(@minus, meanshape, mean(meanshape)),...
                bsxfun(@minus, current_shape, mean(current_shape)), ...
                'nonreflectivesimilarity');
    
    %extract shape indexed pixels
    candidate_pixel_location = fernCascade.candidate_pixel_location;
    nearest_landmark_index = fernCascade.nearest_landmark_index;
    intensities = zeros(1, params.P);
    for j = 1: params.P
        x = candidate_pixel_location(j, 1)*image.intermediate_bbx(3);
        y = candidate_pixel_location(j, 2)* image.intermediate_bbx(4);
        [project_x, project_y] = transformPointsForward(image.meanshape2tf, x, y);
        index = nearest_landmark_index(j);

        real_x = round(project_x + current_shape(index, 1));
        real_y = round(project_y + current_shape(index, 2));
        real_x = max(1, min(real_x, size(image.faceimg, 2)-1));
        real_y = max(1, min(real_y, size(image.faceimg, 1)-1));
        intensities(j)= image.faceimg(real_y, real_x);
    end
    
    delta_shape = zeros(size(params.mean_shape));
    for i = 1: params.K
        fern = fernCascade.fern{i};
        delta_shape = delta_shape + fernTest(intensities, fern, params);
    end
    
    %convert to the currentshape model
    [u, v] = transformPointsForward(image.meanshape2tf, delta_shape(:, 1), delta_shape(:, 2));
    delta_shape_interm_coord = [u, v];
    prediction_delta = bsxfun(@times, delta_shape_interm_coord, [image.intermediate_bbx(3),image.intermediate_bbx(4)]);

end

function fern_pred = fernTest(intensities, fern, params)
    index = 0;
    for i = 1: params.F
        intensity_1 = intensities(fern.selected_pixel_locations(i, 1));
        intensity_2 = intensities(fern.selected_pixel_locations(i, 2));
        
        if (intensity_1 - intensity_2) >= fern.threshold(i)
            index = index + 2^(i-1);
        end
    end
    index = index + 1;
    fern_pred = fern.bin_output{index};
end