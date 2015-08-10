# Batch-Normal-For-Caffe

There are some codes for new vision Caffe to extend Batch Normalization Layer.
To add this layer, you only have to modify 'common_layers.hpp', 'upgrade_proto.cpp' and 'caffe.proto'. 
And add 'bn_layer.cpp', 'bn_layer.cu'. It isn't necessary to modify 'layer_factory.cpp' any more.

The original code is https://github.com/ChenglongChen/batch_normalization, which I modified from.
And all their rights belong to the orignial author.

In the tool dir, I include some codes to convert the model generalized from MatConvnet to Caffemodel by using matcaffe.
