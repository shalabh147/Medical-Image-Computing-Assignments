function ift_R = myFilter(R,filter,theta,L)
    new_size = size(R,1);
    freqs=linspace(-1, 1, new_size).';
    freqs(abs(freqs)>L) = 0;
    
    if(strcmp(filter,'Ram-Lak'))
        filter = abs(freqs);
    elseif(strcmp(filter,'SheppLogan'))
        filter = 2*L*abs(sin(pi*freqs/(2*L)))/pi;
    elseif(strcmp(filter,'Cosine'))
        filter = abs(freqs).*cos(pi*freqs/(2*L));
    end
    
        
    filter = repmat(filter, [1 length(theta)]);
    
    %filter(filter>L) = 0;
    
    ft_R = fftshift(fft(R,[],1),1);
    filteredProj = ft_R .* filter;
    filteredProj = ifftshift(filteredProj,1);
    ift_R = real(ifft(filteredProj,[],1));
    
   
    
  %  for i=1:60
   %     filtered_image(:,i) = filtered_image(:,i).*filter
   % end
    
    
   
   
   %multiplying with abs(w) means multiplying 
   
end