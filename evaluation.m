function error = evaluation(prediction, groundtruth)
z = size(groundtruth, 1);
[x, y] = size(prediction{1});
detected_points_all = zeros(x, y, z);
ground_truth_all = zeros(x, y, z);
for i = 1: z
    detected_points_all(:, :, i) = prediction{i};
    ground_truth_all(:, :, i) = groundtruth{i};
end

error = compute_error( ground_truth_all, detected_points_all, 1, 4);
end