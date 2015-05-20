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
            prediction_delta = fernCascadeTest(image, bbx, current_shape, Model{t}.fernCascade, params);
            current_shape = prediction_delta + projectShape(current_shape, bbx.bbx_chs);
            current_shape = reprojectShape(current_shape, bbx.bbx_chs);
        end
    end
end

function prediction_delta = fernCascadeTest(image, bbx, current_shape, fernCascade, params)
    image.intermediate_bbx = getbbox(current_shape);
    meanshape = reprojectShape(params.mean_shape, image.intermediate_bbx);
    image.tf2meanshape = fitgeotrans(bsxfun(@minus, current_shape, mean(current_shape)), ...
                bsxfun(@minus, meanshape, mean(meanshape)),...
                'nonreflectivesimilarity');
    image.meanshape2tf = fitgeotrans( bsxfun(@minus, meanshape, mean(meanshape)),...
                bsxfun(@minus, current_shape, mean(current_shape)), ...
                'nonreflectivesimilarity');
    result = zeros(size(params.mean_shape));
    
    %extract shape indexed pixels
    
    for i = 1: params.K
        fern = fernCascade{i}.fern;
        result = result + fernTest(image, bbx, current_shape, fern);
    end
    
    %convert to the currentshape model
end

function fernTest(image, bbx, current_shape, fern)
end