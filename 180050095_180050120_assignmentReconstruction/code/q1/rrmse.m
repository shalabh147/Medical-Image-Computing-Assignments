function value = rrmse(mat1,mat2)
    %value = sqrt(sum(sum((mat1 - mat2).*(mat1 - mat2))))/sqrt(sum(sum(mat1.*mat1)));
      value = norm(mat1 - mat2,'fro')/norm(mat1,'fro');
end