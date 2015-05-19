function ESR_Train()
    % load data
    % load('../Data/toyData.mat', 'images', 'bbx', 'pts');
    load('../../Data/toyData', 'images_aug', 'bbx_aug', 'pts_aug', 'current_shapes','ground_truth');
    
    params = Train_params;
    params.N_img = size(images_aug, 1);
    % create paralllel local jobs note
    if isempty(gcp('nocreate'))
        parpool(2);
    end

    Y = cell(params.N_img, 1);
    Error = zeros(1, params.T+1);
    Error(1) = mean(evaluation(current_shapes, ground_truth));
    fprintf('Mean Root Square Error: Initial is %f\n', Error(1));
    Model = cell(params.T, 1);
    
    for t = 1: params.T
        % normalized targets
        parfor i = 1:params.N_img
            images_aug{i}.intermediate_bbx = getbbox(current_shapes{i});
            meanshape_reproject = reprojectShape(params.mean_shape, images_aug{i}.intermediate_bbx);
            images_aug{i}.tf2meanshape = fitgeotrans( bsxfun(@minus, current_shapes{i}, mean(current_shapes{i})), ...
                bsxfun(@minus, meanshape_reproject, mean(meanshape_reproject)),...
                'nonreflectivesimilarity');
            images_aug{i}.meanshape2tf = fitgeotrans( bsxfun(@minus, meanshape_reproject, mean(meanshape_reproject)),...
                bsxfun(@minus, current_shapes{i}, mean(current_shapes{i})), ...
                'nonreflectivesimilarity');
            % regression targets
            current_shape = current_shapes{i};
            gt = ground_truth{i};
            bbx = images_aug{i}.intermediate_bbx;
            shape_residual = bsxfun(@rdivide, gt - current_shape, [bbx(3),bbx(4)]);
            [u, v] = transformPointsForward(images_aug{i}.tf2meanshape, shape_residual(:, 1), shape_residual(:, 2));
            Y{i} = [u, v];
        end
        % learn stage regressors
        fprintf('Start %d th Training...\n', t);
        [prediction_delta, fernCascade] = ShapeRegression(Y, images_aug, current_shapes, params, t);
        % reproject and update the current shape
        parfor i = 1:params.N_img
            % regression targets
            bbx = images_aug{i}.intermediate_bbx;
            delta_shape = prediction_delta{i};
            [u, v] = transformPointsForward(images_aug{i}.meanshape2tf, delta_shape(:, 1), delta_shape(:, 2));
            delta_shape_interm_coord = [u, v];
            shape_residual = bsxfun(@times, delta_shape_interm_coord, [bbx(3),bbx(4)]);
            current_shapes{i} = current_shapes{i} + shape_residual;
        end
        Error(t+1) = mean(evaluation(current_shapes, ground_truth));
        fprintf('Mean Root Square Error in %d iteration is %f\n', t, Error(t+1));
        Model{t}.fernCascade = fernCascade;
    end
    bar(Error);
end

function [prediction, fernCascade]= ShapeRegression(Y, images, current_shapes, params, t)
    %generate local coordinates
    candidate_pixel_location = zeros(params.P, 2);
    nearest_landmark_index = zeros(params.P, 1);
    for i = 1: params.P
        nearest_landmark_index(i) = randi(params.N_fp);
        % sample in mean shape coordinate, [-k, k]
        candidate_pixel_location(i, :) = unifrnd(-params.k(t), params.k(t));
    end
    % extrate shape indexed pixel
    intensities = zeros(params.N_img, params.P);
    for i = 1: params.N_img
        for j = 1: params.P
            x = candidate_pixel_location(j, 1)*images{i}.intermediate_bbx(3);
            y = candidate_pixel_location(j, 2)* images{i}.intermediate_bbx(4);
            [project_x, project_y] = transformPointsForward(images{i}.meanshape2tf, x, y);
            index = nearest_landmark_index(j);
            
            real_x = round(project_x + current_shapes{i}(index, 1));
            real_y = round(project_y + current_shapes{i}(index, 2));
            real_x = max(1, min(real_x, size(images{i}.faceimg, 2)-1));
            real_y = max(1, min(real_y, size(images{i}.faceimg, 1)-1));
            intensities(i, j)= images{i}.faceimg(real_y, real_x);
        end
    end
    % compute pixel-pixel covariance
    covariance = cov(intensities);
    % train internal level boost regression
    regression_targets = Y; % initialization
    prediction = cell(params.N_img, 1);
    parfor i = 1: params.N_img
        prediction{i} = zeros(params.N_fp, 2);
    end
    
    fernCascade = cell(params.K, 1);
    for i = 1: params.K
        %fprintf('Fern Training: second level is %d out of %d\n', i, params.K);
        [prediction_delta, fern] = fernRegression(regression_targets, intensities, ...
            covariance, candidate_pixel_location, nearest_landmark_index,params);
        for j = 1: size(prediction_delta,1)
            prediction{j} = prediction{j}+ prediction_delta{j};
            regression_targets{j} = regression_targets{j} - prediction_delta{j};
        end
        fernCascade{i}.fern = fern;
    end
    
end

function [prediction, fern] = fernRegression(regression_targets, intensities, ...
    covariance, candidate_pixel_locations, nearest_landmark_index, params)
    selected_pixel_index = zeros(params.F, 2);
    selected_pixel_locations = zeros(params.F, 4);
    selected_nearest_landmark_index = zeros(params.F, 2);
    threshold = zeros(params.F,1);
    
    for i = 1: params.F
        v = rand(params.N_fp, 2);
        v = v/norm(v);
        % random projection
        projection_result = zeros(params.N_img, 1);
        for j = 1: params.N_img
            projection_result(j) = sum(sum(regression_targets{j}.*v));
        end
        % compute target-pixel covariance
        Y_prob = zeros(1, params.P);
        for j = 1: params.P
            Y_prob(j) = corr(projection_result, intensities(:, j));
        end
        % compute variance of Y_prob
        % sigma_prob = std(Y_prob);
        
        max_correlation = -1;
        m_f = 1;
        n_f = 1;
        for m = 1: params.P
            for n = 1: params.P
                sigma_mn = covariance(m,m) + covariance(n, n) - 2*covariance(m, n);
                if(abs(sigma_mn)<1e-10)
                    continue;
                end
                flag = 0;
                for p = 1: i
                    if (m == selected_pixel_index(p, 1)) && (n == selected_pixel_index(p, 2))
                        flag = 1;
                        break;
                    elseif (m == selected_pixel_index(p, 2)) && (n == selected_pixel_index(p, 1))
                        flag = 1;
                        break;
                    end
                end
                if(flag)
                    continue; 
                end;
                
                temp = (Y_prob(m) - Y_prob(n))/sqrt(sigma_mn); % it does need sqrt(sigma_mn*sigma_prob)
                if(abs(temp)> max_correlation) % do it need abs
                    max_correlation = temp;
                    m_f = m;
                    n_f = n;
                end
            end
        end
        selected_pixel_index(i, 1) = m_f;
        selected_pixel_index(i, 2) = n_f;
        
        selected_pixel_locations(i,1) = candidate_pixel_locations(m_f,1);
        selected_pixel_locations(i,2) = candidate_pixel_locations(m_f,2);
        selected_pixel_locations(i,3) = candidate_pixel_locations(n_f,1);
        selected_pixel_locations(i,4) = candidate_pixel_locations(n_f,2);
        selected_nearest_landmark_index(i,1) = nearest_landmark_index(m_f); 
        selected_nearest_landmark_index(i,2) = nearest_landmark_index(n_f);
        
        max_diff = -1;
        for j = 1: params.N_img
            temp = intensities(j, m_f) - intensities(j, n_f);
            if (abs(temp) > max_diff)
                max_diff = abs(temp);
            end
        end

        threshold(i) = unifrnd(-0.2*max_diff, 0.2*max_diff);
    end
    
    % determine the bins of each shape
    bin_num = 2^params.F;
    shapes_in_bin = zeros(params.N_img, bin_num);
    index2 = zeros(bin_num, 1);
    for i = 1: params.N_img
        index = 0;
        for j = 1: params.F
            intensity_1 = intensities(i, selected_pixel_index(j, 1));
            intensity_2 = intensities(i, selected_pixel_index(j, 2));
            if(intensity_1 - intensity_2 >= threshold(j))
                index = index + 2^(j-1);
            end
        end
        index2(index+1) = index2(index+1) + 1;
        shapes_in_bin(index2(index+1), index+1)= i;
    end

    % get bin output
    prediction = cell(params.N_img,1);
    bin_output = cell(bin_num, 1);
    for i = 1: bin_num
        bin_size = index2(i);
        temp = zeros(params.N_fp, 2);
        if 0 ~= bin_size
            for j = 1: bin_size
                index = shapes_in_bin(j, i);
                temp = temp + regression_targets{index};
            end
        else
            bin_output{i} = temp;
            continue;
        end
        temp = (1/((1+1000/bin_size)*bin_size))*temp;
        bin_output{i} = temp;
        for j = 1: bin_size
            index = shapes_in_bin(j, i);
            prediction{index} = temp;
        end
    end

    fern.bin_output = bin_output;
    fern.selected_pixel_locations = selected_pixel_locations;
    fern.selected_nearest_landmark_index = selected_nearest_landmark_index;
    fern.threshold = threshold;
end