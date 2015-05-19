function cova = calculate_covariance(density1, density2)
    mean_1 = mean(density1);
    mean_2 = mean(density2);
    density1 = density1 - mean_1;
    density2 = density2 - mean_2;
    cova = mean(density1.*density2);
end