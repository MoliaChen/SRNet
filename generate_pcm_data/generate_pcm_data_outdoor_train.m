load('D:\projects\wireless communication AI competition\paper\data\DATA_Htrainout.mat');

H = zeros(100000,2048);
for i = 1:1:100000
    a = reshape(HT(i,:),[32 32 2]);
    b = a -0.5;
    comp_a = squeeze(b(:,:,1) + b(:,:,2).*1i);
    path_sum = squeeze(sum(power(abs(comp_a),2),2));
    [~,max_index] = sort(path_sum,'descend');
    select_index = sort(max_index(1:14));
    no_select_index = sort(max_index(15:32));
    H(i,:) = reshape(a([select_index.' no_select_index.'],:,:),1,2048);
end

H = single(H);
save('outdoor_H_14_train','H');