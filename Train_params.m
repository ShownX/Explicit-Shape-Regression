function params = Train_params()
    params.T = 10; % the iteration stages
    params.P = 40; % default = 400, the pixel number sampled on the images
    params.K = 50; % default = 500, the number of fern on the internal-level boosted regression
    params.k = 0.3; % the local scale of search
    params.F = 5; % the number of features in fern
    
    load('../Data/mean_shape.mat', 'S0');
    params.mean_shape = S0;
    
    params.N_fp = size(params.mean_shape, 1);
    params.N_img = 0;
    
end