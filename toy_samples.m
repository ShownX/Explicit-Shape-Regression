function [images_toy, bbx_toy, pts_toy, crt_shapes_toy, gt_shapes_toy] = toy_samples(images, bbx, pts)
    img_num = size(images,1);
    initial_num = 5;
    
    img_num2 = 10;
    Idx = randperm(img_num, img_num2);
    
    images_toy = cell(img_num2*initial_num, 1);
    pts_toy = cell(img_num2*initial_num, 1);
    bbx_toy = cell(img_num2*initial_num, 1);
    crt_shapes_toy = cell(img_num2*initial_num, 1);
    gt_shapes_toy = cell(img_num2*initial_num, 1);
    
    for i = 1: img_num2
        for j = 1: initial_num
            index = Idx(i);
            while index == Idx(i)
                index = randi(img_num);
            end
            images_toy{(i-1)*initial_num + j}.faceimg = images{Idx(i)}.faceimg;
            bbx_toy{(i-1)*initial_num + j}.bbx_chs = bbx{Idx(i)}.bbx_chs;
            pts_toy{(i-1)*initial_num + j}.pts_chs = pts{Idx(i)}.pts_chs;

            temp = projectShape(pts{index}.pts_chs, bbx{index}.bbx_chs);
            temp = reprojectShape(temp, bbx{Idx(i)}.bbx_chs);
            crt_shapes_toy{(i-1)*initial_num + j} = temp;
            gt_shapes_toy{(i-1)*initial_num + j} = pts{Idx(i)}.pts_chs;
        end
    end
end