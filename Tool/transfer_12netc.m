clear;
addpath ./matlab;
model = './8_5_model/12net-cc-v1/deploy.prototxt';
caffe.set_mode_cpu();
net = caffe.Net(model,'test');
net_t = load('./8_5_model/12net-cc-v1/f12net_c.mat');

net.layers('conv1').params(1).set_data(net_t.layers{1,1}.weights{1});%set weights
net.layers('conv1').params(2).set_data(net_t.layers{1,1}.weights{2}');%set bias

net.layers('conv2').params(1).set_data(net_t.layers{1,5}.weights{1});%set weights
net.layers('conv2').params(2).set_data(net_t.layers{1,5}.weights{2}');%set bias

net.layers('conv3').params(1).set_data(net_t.layers{1,8}.weights{1});%set weights
net.layers('conv3').params(2).set_data(net_t.layers{1,8}.weights{2}');%set bias

bw1 = reshape(net_t.layers{1,2}.weights{1},1,1,16,1);
bb1 = reshape(net_t.layers{1,2}.weights{2},1,1,16,1);
net.layers('bnorm1').params(1).set_data(bw1);%set weights
net.layers('bnorm1').params(2).set_data(bb1);%set bias

bw2 = reshape(net_t.layers{1,6}.weights{1},1,1,128,1);
bb2 = reshape(net_t.layers{1,6}.weights{2},1,1,128,1);
net.layers('bnorm2').params(1).set_data(bw2);%set weights
net.layers('bnorm2').params(2).set_data(bb2);%set bias

bw3 = reshape(net_t.layers{1,9}.weights{1},1,1,45,1);
bb3 = reshape(net_t.layers{1,9}.weights{2},1,1,45,1);
net.layers('bnorm3').params(1).set_data(bw3);%set weights
net.layers('bnorm3').params(2).set_data(bb3);%set bias

net.save('f12netc.caffemodel');

%------------------test----------
im = im2single(imread('1.png'));
im = imresize(im,[12 12]);
%net.blobs('data').set_data(im);
%data = net.blobs('data');
res = net.forward({im});
prob = res{1};
