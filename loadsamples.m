function data = loadsamples(datapath, imgtype)
    dd = dir(fullfile(datapath,['*.', imgtype]));
    data_len = length(dd);
    data = cell(data_len, 1);

    parfor i = 1: data_len
        
        img = im2uint8(imread(fullfile(datapath, dd(i).name)));
        data{i}.height_orig = size(img, 1);
        data{i}.width_orig = size(img, 2);
        shapepath = strrep(fullfile(datapath, dd(i).name), imgtype, 'pts');
        data{i}.shape_gt = double(loadshape(shapepath));
        data{i}.bbox_gt = getbbox(data{i}.shape_gt);
        
        region = enlargingbbox(data{i}.bbox_gt, 2.0);
    
        region(2) = double(max(region(2), 1));
        region(1) = double(max(region(1), 1));

        bottom_y = double(min(region(2) + region(4) - 1, data{i}.height_orig));
        right_x = double(min(region(1) + region(3) - 1, data{i}.width_orig));

        img_region = img(region(2):bottom_y, region(1):right_x, :);

        % recalculate the location of groundtruth shape and bounding box
        data{i}.shape_gt = bsxfun(@minus, data{i}.shape_gt, double([region(1) region(2)]));
        data{i}.bbox_gt = getbbox(data{i}.shape_gt);

        % only use inner points
        data{i}.shape_gt = data{i}.shape_gt;

        data{i}.isdet = 0;

        if size(img_region, 3) == 1
            data{i}.img_gray = img_region;
        else
            % hsv = rgb2hsv(img_region);
            data{i}.img_gray = rgb2gray(img_region);
        end    

        data{i}.width    = size(img_region, 2);
        data{i}.height   = size(img_region, 1);
    end
end

function shape = loadshape(path)
    % function: load shape from pts file
    file = fopen(path);

    if ~isempty(strfind(path, 'COFW'))
        shape = textscan(file, '%d16 %d16 %d8', 'HeaderLines', 3, 'CollectOutput', 3);
    else
        shape = textscan(file, '%d16 %d16', 'HeaderLines', 3, 'CollectOutput', 2);
    end
    fclose(file);

    shape = shape{1};
end

function region = enlargingbbox(bbox, scale)

    region(1) = floor(bbox(1) - (scale - 1)/2*bbox(3));
    region(2) = floor(bbox(2) - (scale - 1)/2*bbox(4));

    region(3) = floor(scale*bbox(3));
    region(4) = floor(scale*bbox(4));

end
