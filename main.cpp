//=======================================================================
// Copyright (c) 2017 Adrian Schneider
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <iostream>
#include <assert.h>
#include <math.h>
// export CPLUS_INCLUDE_PATH=/home/cauchy/github/mnist-fashion/include:$CPLUS_INCLUDE_PATH
#include "mnist/mnist_reader.hpp"
// export CPLUS_INCLUDE_PATH=/usr/local/include/opencv4:$CPLUS_INCLUDE_PATH
#include <opencv2/opencv.hpp>
#include "oneapi/dnnl/dnnl.hpp"
#include "example_utils.hpp"

using namespace dnnl;

// #define DEBUG
#define MODIFY
// #define USEREORDER

const std::string MNIST_DATA_LOCATION = "/home/cauchy/github/mnist-fashion/data/mnist";
const std::string MNIST_FASHION_DATA_LOCATION = "/home/cauchy/github/mnist-fashion/data/fashion";
const memory::dim N = 16; // batch_size

// get fasion-mnist
mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
    mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(MNIST_FASHION_DATA_LOCATION, 240, 40);
memory::dim train_t = 0;
memory::dim test_t = 0;

using tag = memory::format_tag;
using dt = memory::data_type;

class Conv2DwithReLu {
public:
    Conv2DwithReLu(dnnl::engine eng, std::vector<primitive> &net,
            std::vector<std::unordered_map<int, memory>> &net_args,
            const memory &src_memory, const memory::dims &src_tz,
            const memory::dims &dst_tz,
            const memory::dims &weights_tz,
            const memory::dims &strides, const memory::dims &padding,
            const float &negative_slope);
    ~Conv2DwithReLu() = default;
    Conv2DwithReLu(const Conv2DwithReLu &obj) = delete; // ban copying to avoid some bugs
    memory dst_memory() { return dst_m; }

private:
    std::vector<float> weights;
    std::vector<float> bias;
    memory dst_m;
    // memory *user_weights_memory_p, *user_bias_memory_p;
    // md *src_md_p, *bias_md_p, *weights_md_p, *dst_md_p;
    // convolution_forward::desc *desc_p;
    // convolution_forward::primitive_desc *pd_p;
    // memory *weights_memory_p, *bias_memory_p;
};

class MaxPooling {
public:
    MaxPooling(dnnl::engine eng, std::vector<primitive> &net,
            std::vector<std::unordered_map<int, memory>> &net_args,
            const memory &src_memory,
            const memory::dims &kernel,
            const memory::dims &dst_tz, const memory::dims &strides,
            const memory::dims &padding, bool trained = true);
    ~MaxPooling() = default;
    MaxPooling(const MaxPooling &obj) = delete; // ban copying to avoid some bugs
    memory dst_memory() { return dst_m; }
private:
    bool iftrain;
    memory dst_m;
};

class Dense {
public:
    Dense(dnnl::engine eng, std::vector<primitive> &net,
            std::vector<std::unordered_map<int, memory>> &net_args,
            const memory &src_memory, const memory::dims &src_tz,
            const memory::dims &dst_tz,
            const memory::dims &weights_tz);
    ~Dense() = default;
    Dense(const Dense &obj) = delete;
    memory dst_memory() { return dst_m; }

private:
    std::vector<float> weights;
    std::vector<float> bias;
    memory dst_m;
};

class ReLU {
public:
    ReLU(dnnl::engine eng, std::vector<primitive> &net,
            std::vector<std::unordered_map<int, memory>> &net_args,
            const memory &src_memory, const float &negative_slope)
    {
        auto desc = eltwise_forward::desc(prop_kind::forward,
                algorithm::eltwise_relu, src_memory.get_desc(),
                negative_slope);
        auto pd = eltwise_forward::primitive_desc(desc, eng);

        // create relu dst memory
        auto dst_memory = memory(pd.dst_desc(), eng);

        net.push_back(eltwise_forward(pd));
        net_args.push_back({{DNNL_ARG_SRC, src_memory},
                {DNNL_ARG_DST, dst_memory}});
        dst_m = dst_memory;
    }

    ~ReLU() = default;
    ReLU(const ReLU &obj) = delete;
    memory dst_memory() { return dst_m; }

private:
    memory dst_m;
};

class CrossEntropyLoss {
public:
    CrossEntropyLoss(dnnl::engine eng, std::vector<primitive> &net,
                            std::vector<std::unordered_map<int, memory>> &net_args,
                            const memory &y_hat, const memory &y_true,
                            const memory::dims &y_tz);
    ~CrossEntropyLoss() = default;
    CrossEntropyLoss(const CrossEntropyLoss &obj);
private:

};

void VGG11(engine::kind engine_kind) {


    auto eng = engine(engine_kind, 0);
    stream s(eng);

    // Vector of primitives and their execute arguments
    std::vector<primitive> net_fwd, net_bwd;
    std::vector<std::unordered_map<int, memory>> net_fwd_args, net_bwd_args;

    // Vectors of input data and expected output
    std::vector<float> net_src(N * 3 * 224 * 224);
    std::vector<float> net_dst(N * 10); // 10 classes

    const int IS = 224*224; // input size

    for (size_t i = 0; i < N * 10; ++i)
        net_dst[i] = (float)0;

    // read src and dst data from fasion-mnist
    for (size_t i = 0; i < N; ++i) {
        std::vector<uint8_t> pic = dataset.training_images[train_t];
        size_t ans = dataset.training_labels[train_t];
        ++train_t;

        // resize imagine (28, 28) -> (224, 224, 3)
        cv::Mat img = cv::Mat(28, 28, CV_8U);
        for (size_t i = 0; i < 28; ++i)
            for (size_t j = 0; j < 28; ++j)
                img.at<uint8_t>(i, j) = (uint8_t)pic[i*28 + j];

        cv::Mat img_rgb(28, 28, CV_8UC3);
        cv::merge(std::vector<cv::Mat>{img, img, img}, img_rgb);

        cv::Mat img_res(224, 224, CV_8UC3);
        cv::resize(img_rgb, img_res, cv::Size() , 8, 8, cv::INTER_LINEAR); //INTER_CUBIC slower

        auto data = img_res.data;

        int fpi = i * 224 * 224 * 3; // first pixel index

        // write data into src while doing normalization (divided by 255)
        for (size_t c = 0; c < 3; ++c) // channel
            for (size_t w = 0; w < 224; ++w)
                for (size_t h = 0; h < 224; ++h)
                    net_src[fpi + c * IS + w * 224 + h] = ((float)(*(data + w * 672 + h * 3 + c))) / 255.0;

        // write data into dst
        net_dst[i*10 + ans] = 1;
    }

    auto net_dst_memory = memory({{memory::dims{N, 10}}, dt::f32, tag::nc}, eng);
    write_to_dnnl_memory(net_dst.data(), net_dst_memory);

    const float negative_slope = 0.0f;

    // VGG11: block 1-1: conv1
    // {batch, 3, 224, 224} (x) {64, 3, 3, 3} -> {batch, 64, 224, 224}
    // kernel: {3,3}; strides: {1, 1}; padding: {1, 1}
    memory::dims conv1_src_tz = {N, 3, 224, 224};
    memory::dims conv1_weights_tz = {64, 3, 3, 3};
    memory::dims conv1_dst_tz = {N, 64, 224, 224};
    memory::dims conv1_strides = {1, 1};
    memory::dims conv1_padding = {1, 1};

    auto conv1_src_memory
            = memory({{conv1_src_tz}, dt::f32, tag::nchw}, eng);
    write_to_dnnl_memory(net_src.data(), conv1_src_memory);

    Conv2DwithReLu conv1(eng, net_fwd, net_fwd_args,
                conv1_src_memory, conv1_src_tz,
                conv1_dst_tz, conv1_weights_tz,
                conv1_strides, conv1_padding, negative_slope);
    memory conv1_dst_memory = conv1.dst_memory();

    // VGG11: block 1-2: max_pooling1
    // {batch, 64, 224, 224} -> {batch, 64, 112, 112}
    // kernel: {2, 2}
    // strides: {2, 2}
    memory::dims pool1_dst_tz = {N, 64, 112, 112};
    memory::dims pool1_kernel = {2, 2};
    memory::dims pool1_strides = {2, 2};
    memory::dims pool1_padding = {0, 0};
    MaxPooling pool1(eng, net_fwd, net_fwd_args,
                conv1_dst_memory,
                pool1_kernel, pool1_dst_tz,
                pool1_strides, pool1_padding);
    memory pool1_dst_memory = pool1.dst_memory();

    // VGG11: block 2-1: conv2
    // {batch, 64, 112, 112} -> {batch, 128, 112, 112}
    // kernel: {3, 3}; strides: {1, 1}; padding: {1, 1}
    memory::dims conv2_src_tz = {N, 64, 112, 112};
    memory::dims conv2_weights_tz = {128, 64, 3, 3};
    memory::dims conv2_dst_tz = {N, 128, 112, 112};
    memory::dims conv2_strides = {1, 1};
    memory::dims conv2_padding = {1, 1};
    Conv2DwithReLu conv2(eng, net_fwd, net_fwd_args,
                pool1_dst_memory, conv2_src_tz,
                conv2_dst_tz, conv2_weights_tz,
                conv2_strides, conv2_padding, negative_slope);
    memory conv2_dst_memory = conv2.dst_memory();

    // VGG11: block 2-2 max_pooling2
    // {batch, 128, 112, 112} -> {batch, 128, 56, 56}
    // kernel: {2, 2}; strides: {2, 2}; padding: {0, 0}
    memory::dims pool2_dst_tz = {N, 128, 56, 56};
    memory::dims pool2_kernel = {2, 2};
    memory::dims pool2_strides = {2, 2};
    memory::dims pool2_padding = {0, 0};
    MaxPooling pool2(eng, net_fwd, net_fwd_args,
                conv2_dst_memory,
                pool2_kernel, pool2_dst_tz,
                pool2_strides, pool2_padding);
    memory pool2_dst_memory = pool2.dst_memory();

    // VGG11: block 3-1: conv3
    // {batch, 128, 56, 56} -> {batch, 256, 56, 56}
    // kernel: {3, 3}; strides: {1, 1}; padding: {1, 1}
    memory::dims conv3_src_tz = {N, 128, 56, 56};
    memory::dims conv3_weights_tz = {256, 128, 3, 3};
    memory::dims conv3_dst_tz = {N, 256, 56, 56};
    memory::dims conv3_strides = {1, 1};
    memory::dims conv3_padding = {1, 1};
    Conv2DwithReLu conv3(eng, net_fwd, net_fwd_args,
                pool2_dst_memory, conv3_src_tz,
                conv3_dst_tz, conv3_weights_tz,
                conv3_strides, conv3_padding, negative_slope);
    memory conv3_dst_memory = conv3.dst_memory();

    // VGG11: block 3-2: conv4
    // {batch, 256, 56, 56} -> {batch, 256, 56, 56}
    memory::dims conv4_src_tz = {N, 256, 56, 56};
    memory::dims conv4_weights_tz = {256, 256, 3, 3};
    memory::dims conv4_dst_tz = {N, 256, 56, 56};
    memory::dims conv4_strides = {1, 1};
    memory::dims conv4_padding = {1, 1};
    Conv2DwithReLu conv4(eng, net_fwd, net_fwd_args,
                conv3_dst_memory, conv4_src_tz,
                conv4_dst_tz, conv4_weights_tz,
                conv4_strides, conv4_padding, negative_slope);
    memory conv4_dst_memory = conv4.dst_memory();

    // VGG11: block 3-3: max_pooling3
    // {batch, 256, 56, 56} -> {batch, 256, 28, 28}
    // kernel: {2, 2}; strides: {2, 2}; padding: {1, 1}
    memory::dims pool3_dst_tz = {N, 256, 28, 28};
    memory::dims pool3_kernel = {2, 2};
    memory::dims pool3_strides = {2, 2};
    memory::dims pool3_padding = {0, 0};
    MaxPooling pool3(eng, net_fwd, net_fwd_args,
                conv4_dst_memory,
                pool3_kernel, pool3_dst_tz,
                pool3_strides, pool3_padding);
    memory pool3_dst_memory = pool3.dst_memory();

    // VGG11: block 4-1: conv5
    // {batch, 256, 28, 28} -> {batch, 512, 28, 28}
    // kernel: {3, 3}; strides: {1, 1}; padding: {1, 1}
    memory::dims conv5_src_tz = {N, 256, 28, 28};
    memory::dims conv5_weights_tz = {512, 256, 3, 3};
    memory::dims conv5_dst_tz = {N, 512, 28, 28};
    memory::dims conv5_strides = {1, 1};
    memory::dims conv5_padding = {1, 1};
    Conv2DwithReLu conv5(eng, net_fwd, net_fwd_args,
                pool3_dst_memory, conv5_src_tz,
                conv5_dst_tz, conv5_weights_tz,
                conv5_strides, conv5_padding, negative_slope);
    memory conv5_dst_memory = conv5.dst_memory();

    // VGG11: block 4-2: conv6
    // {batch, 512, 28, 28} -> {batch, 512, 28, 28}
    // kernel: {3, 3}; strides: {1, 1}; padding: {1, 1}
    memory::dims conv6_src_tz = {N, 512, 28, 28};
    memory::dims conv6_weights_tz = {512, 512, 3, 3};
    memory::dims conv6_dst_tz = {N, 512, 28, 28};
    memory::dims conv6_strides = {1, 1};
    memory::dims conv6_padding = {1, 1};
    Conv2DwithReLu conv6(eng, net_fwd, net_fwd_args,
                conv5_dst_memory, conv6_src_tz,
                conv6_dst_tz, conv6_weights_tz,
                conv6_strides, conv6_padding, negative_slope);
    memory conv6_dst_memory = conv6.dst_memory();

    // VGG11: block 4-3: max_pooling4
    // {batch, 512, 28, 28} -> {batch, 512, 14, 14}
    // kernel: {2, 2}; strides: {2, 2}; padding: {1, 1}
    memory::dims pool4_dst_tz = {N, 512, 14, 14};
    memory::dims pool4_kernel = {2, 2};
    memory::dims pool4_strides = {2, 2};
    memory::dims pool4_padding = {0, 0};
    MaxPooling pool4(eng, net_fwd, net_fwd_args,
                conv6_dst_memory,
                pool4_kernel, pool4_dst_tz,
                pool4_strides, pool4_padding);
    memory pool4_dst_memory = pool4.dst_memory();

    // VGG11: block 5-1: conv7
    // {batch, 512, 14, 14} -> {batch, 512, 14, 14}
    // kernel: {3, 3}; strides: {1, 1}; padding: {1, 1}
    memory::dims conv7_src_tz = {N, 512, 14, 14};
    memory::dims conv7_weights_tz = {512, 512, 3, 3};
    memory::dims conv7_dst_tz = {N, 512, 14, 14};
    memory::dims conv7_strides = {1, 1};
    memory::dims conv7_padding = {1, 1};
    Conv2DwithReLu conv7(eng, net_fwd, net_fwd_args,
                pool4_dst_memory, conv7_src_tz,
                conv7_dst_tz, conv7_weights_tz,
                conv7_strides, conv7_padding, negative_slope);
    memory conv7_dst_memory = conv7.dst_memory();

    // VGG11: block 5-2: conv8
    // {batch, 512, 14, 14} -> {batch, 512, 14, 14}
    // kernel: {3, 3}; strides: {1, 1}; padding: {1, 1}
    memory::dims conv8_src_tz = {N, 512, 14, 14};
    memory::dims conv8_weights_tz = {512, 512, 3, 3};
    memory::dims conv8_dst_tz = {N, 512, 14, 14};
    memory::dims conv8_strides = {1, 1};
    memory::dims conv8_padding = {1, 1};
    Conv2DwithReLu conv8(eng, net_fwd, net_fwd_args,
                conv7_dst_memory, conv8_src_tz,
                conv8_dst_tz, conv8_weights_tz,
                conv8_strides, conv8_padding, negative_slope);
    memory conv8_dst_memory = conv8.dst_memory();

    // VGG11: block 5-3: max_pooling5
    // {batch, 512, 14, 14} -> {batch, 512, 7, 7}
    // kernel: {2, 2}; strides: {2, 2}; padding: {1, 1}
    memory::dims pool5_dst_tz = {N, 512, 7, 7};
    memory::dims pool5_kernel = {2, 2};
    memory::dims pool5_strides = {2, 2};
    memory::dims pool5_padding = {0, 0};
    MaxPooling pool5(eng, net_fwd, net_fwd_args,
                conv8_dst_memory,
                pool5_kernel, pool5_dst_tz,
                pool5_strides, pool5_padding);
    memory pool5_dst_memory = pool5.dst_memory();

    // VGG11: FC4096*2
    // {batch, 512, 7, 7} -> {batch, 4096} -> {batch, 4096}
    memory::dims fc1_src_tz = {N, 512, 7, 7};
    memory::dims fc1_weights_tz = {4096, 512, 7, 7};
    memory::dims fc1_dst_tz = {N, 4096};
    Dense fc1(eng, net_fwd, net_fwd_args,
                pool5_dst_memory, fc1_src_tz,
                fc1_dst_tz, fc1_weights_tz);
    memory fc1_dst_memory = fc1.dst_memory();

    ReLU fc1_relu(eng, net_fwd, net_fwd_args,
                fc1_dst_memory, negative_slope);
    memory fc1_relu_dst_memory = fc1_relu.dst_memory();

    memory::dims fc2_src_tz = {N, 4096};
    memory::dims fc2_weights_tz = {4096, 4096};
    memory::dims fc2_dst_tz = {N, 4096};
    Dense fc2(eng, net_fwd, net_fwd_args,
                fc1_relu_dst_memory, fc2_src_tz,
                fc2_dst_tz, fc2_weights_tz);
    memory fc2_dst_memory = fc2.dst_memory();

    ReLU fc2_relu(eng, net_fwd, net_fwd_args,
                fc2_dst_memory, negative_slope);
    memory fc2_relu_dst_memory = fc2_relu.dst_memory();

    // VGG11: FC1000
    // {batch, 4096} -> {batch, 1000}
    memory::dims fc3_src_tz = {N, 4096};
    memory::dims fc3_weights_tz = {1000, 4096};
    memory::dims fc3_dst_tz = {N, 1000};
    Dense fc3(eng, net_fwd, net_fwd_args,
                fc2_relu_dst_memory, fc3_src_tz,
                fc3_dst_tz, fc3_weights_tz);
    memory fc3_dst_memory = fc3.dst_memory();

    ReLU fc3_relu(eng, net_fwd, net_fwd_args,
                fc3_dst_memory, negative_slope);
    memory fc3_relu_dst_memory = fc3_relu.dst_memory();

    // VGG11: FC10
    // {batch, 1000} -> {batch, 10}
    memory::dims fc4_src_tz = {N, 1000};
    memory::dims fc4_weights_tz = {10, 1000};
    memory::dims fc4_dst_tz = {N, 10};
    Dense fc4(eng, net_fwd, net_fwd_args,
                fc3_relu_dst_memory, fc4_src_tz,
                fc4_dst_tz, fc4_weights_tz);
    memory fc4_dst_memory = fc4.dst_memory();

    // VGG11: the end, softmax
    memory::dims softmax_src_tz = {N, 10};
    auto softmax_src_md = memory::desc(softmax_src_tz, dt::f32, tag::nc);
    auto softmax_dec = softmax_forward::desc(prop_kind::forward_training, softmax_src_md, 1);
    auto softmax_pd = softmax_forward::primitive_desc(softmax_dec, eng);
    auto softmax_dst_memory = memory(softmax_pd.dst_desc(), eng);

    net_fwd.push_back(softmax_forward(softmax_pd));
    net_fwd_args.push_back({{DNNL_ARG_SRC, fc4_dst_memory},
                            {DNNL_ARG_DST, softmax_dst_memory}});

    memory::dims y_tz = {N, 10};
    CrossEntropyLoss loss(eng, net_fwd, net_fwd_args, softmax_dst_memory, net_dst_memory, y_tz);


    //-----------------------------------------------------------------------
    //----------------- Backpropagation Stream  (Data)-------------------------------------



    return;


#ifndef MODIFY
    std::vector<float> conv1_weights(product(conv1_weights_tz));
    std::vector<float> conv1_bias(product(conv1_bias_tz));

    // initializing non-zero values for weights and bias
    for (size_t i = 0; i < conv1_weights.size(); ++i)
        conv1_weights[i] = sinf((float)i);
    for (size_t i = 0; i < conv1_bias.size(); ++i)
        conv1_bias[i] = sinf((float)i);

    auto conv1_user_weights_memory
            = memory({{conv1_weights_tz}, dt::f32, tag::oihw}, eng);
    write_to_dnnl_memory(conv1_weights.data(), conv1_user_weights_memory);
    auto conv1_user_bias_memory = memory({{conv1_bias_tz}, dt::f32, tag::x}, eng);
    write_to_dnnl_memory(conv1_bias.data(), conv1_user_bias_memory);

    auto conv1_src_md = memory::desc({conv1_src_tz}, dt::f32, tag::any);
    auto conv1_bias_md = memory::desc({conv1_bias_tz}, dt::f32, tag::any);
    auto conv1_weights_md = memory::desc({conv1_weights_tz}, dt::f32, tag::any);
    auto conv1_dst_md = memory::desc({conv1_dst_tz}, dt::f32, tag::any);

    auto conv1_desc = convolution_forward::desc(prop_kind::forward,
            algorithm::convolution_direct, conv1_src_md, conv1_weights_md,
            conv1_bias_md, conv1_dst_md, conv1_strides, conv1_padding,
            conv1_padding);
    auto conv1_pd = convolution_forward::primitive_desc(conv1_desc, eng);

    // create reorder primitives between user input and conv src if needed
    auto conv1_src_memory = conv1_user_src_memory;
    if (conv1_pd.src_desc() != conv1_user_src_memory.get_desc()) {
        conv1_src_memory = memory(conv1_pd.src_desc(), eng);
        net_fwd.push_back(reorder(conv1_user_src_memory, conv1_src_memory));
        net_fwd_args.push_back({{DNNL_ARG_FROM, conv1_user_src_memory},
                {DNNL_ARG_TO, conv1_src_memory}});
    }

    auto conv1_weights_memory = conv1_user_weights_memory;
    if (conv1_pd.weights_desc() != conv1_user_weights_memory.get_desc()) {
        conv1_weights_memory = memory(conv1_pd.weights_desc(), eng);
        net_fwd.push_back(
                reorder(conv1_user_weights_memory, conv1_weights_memory));
        net_fwd_args.push_back({{DNNL_ARG_FROM, conv1_user_weights_memory},
                {DNNL_ARG_TO, conv1_weights_memory}});
    }

    // added by rbj (159 modified as well)
    auto conv1_bias_memory = conv1_user_bias_memory;
    if (conv1_pd.bias_desc() != conv1_user_bias_memory.get_desc()) {
        conv1_bias_memory = memory(conv1_pd.bias_desc(), eng);
        net_fwd.push_back(
            reorder(conv1_user_bias_memory, conv1_bias_memory)
        );
        net_fwd_args.push_back({{DNNL_ARG_FROM, conv1_user_bias_memory}, {DNNL_ARG_TO, conv1_bias_memory}});
    }

    // create memory for conv dst
    auto conv1_dst_memory = memory(conv1_pd.dst_desc(), eng);

    // finally create a convolution primitive
    net_fwd.push_back(convolution_forward(conv1_pd));
    net_fwd_args.push_back({{DNNL_ARG_SRC, conv1_src_memory},
            {DNNL_ARG_WEIGHTS, conv1_weights_memory},
            {DNNL_ARG_BIAS, conv1_bias_memory},
            {DNNL_ARG_DST, conv1_dst_memory}});

    // VGG11: block 1-2: ReLU

    auto relu1_desc = eltwise_forward::desc(prop_kind::forward,
            algorithm::eltwise_relu, conv1_dst_memory.get_desc(),
            negative_slope);
    auto relu1_pd = eltwise_forward::primitive_desc(relu1_desc, eng);

    // create relu dst memory
    auto relu1_dst_memory = memory(relu1_pd.dst_desc(), eng);

    net_fwd.push_back(eltwise_forward(relu1_pd));
    net_fwd_args.push_back({{DNNL_ARG_SRC, conv1_dst_memory},
            {DNNL_ARG_DST, relu1_dst_memory}});

    // VGG11: block 1-3: max_pooling1
    // {batch, 64, 224, 224} -> {batch, 64, 112, 112}
    // kernel: {2, 2}
    // strides: {2, 2}
    memory::dims pool1_dst_tz = {N, 64, 112, 112};
    memory::dims pool1_kernel = {2, 2};
    memory::dims pool1_strides = {2, 2};
    memory::dims pool1_padding = {0, 0};

    auto pool1_dst_md = memory::desc({pool1_dst_tz}, dt::f32, tag::any);

    //[Create pooling primitive]
    auto pool1_desc = pooling_forward::desc(prop_kind::forward,
            algorithm::pooling_max, relu1_dst_memory.get_desc(), pool1_dst_md,
            pool1_strides, pool1_kernel, pool1_padding, pool1_padding);
    auto pool1_pd = pooling_forward::primitive_desc(pool1_desc, eng);
    auto pool1_dst_memory = memory(pool1_pd.dst_desc(), eng);
    //[Create pooling primitive]

    // create pooling workspace memory if training
    auto pool1_workspace_memory = memory(pool1_pd.workspace_desc(), eng);

    net_fwd.push_back(pooling_forward(pool1_pd));
    net_fwd_args.push_back({{DNNL_ARG_SRC, relu1_dst_memory},
            {DNNL_ARG_DST, pool1_dst_memory}, // delay putting DST until reorder (if needed)
            {DNNL_ARG_WORKSPACE, pool1_workspace_memory}});

    // VGG11: block 2-1: conv2
    // {batch, 64, 112, 112} -> {batch, 128, 112, 112}
    // kernel: {3, 3}; strides: {1, 1}; padding: {1, 1}
    memory::dims conv2_src_tz = {N, 64, 112, 112};
    memory::dims conv2_weights_tz = {128, 64, 3, 3};
    memory::dims conv2_bias_tz = {128};
    memory::dims conv2_dst_tz = {N, 128, 112, 112};
    memory::dims conv2_strides = {1, 1};
    memory::dims conv2_padding = {1, 1};

    std::vector<float> conv2_weights(product(conv2_weights_tz));
    std::vector<float> conv2_bias(product(conv2_bias_tz));

    // initializing non-zero values for weights and bias
    for (size_t i = 0; i < conv2_weights.size(); ++i)
        conv2_weights[i] = sinf((float)i);
    for (size_t i = 0; i < conv2_bias.size(); ++i)
        conv2_bias[i] = sinf((float)i);

    // create memory for user data
    auto conv2_user_weights_memory
            = memory({{conv2_weights_tz}, dt::f32, tag::oihw}, eng);
    write_to_dnnl_memory(conv2_weights.data(), conv2_user_weights_memory);
    auto conv2_user_bias_memory = memory({{conv2_bias_tz}, dt::f32, tag::x}, eng);
    write_to_dnnl_memory(conv2_bias.data(), conv2_user_bias_memory);

    auto conv2_src_md = memory::desc({conv2_src_tz}, dt::f32, tag::any);
    auto conv2_bias_md = memory::desc({conv2_bias_tz}, dt::f32, tag::any);
    auto conv2_weights_md = memory::desc({conv2_weights_tz}, dt::f32, tag::any);
    auto conv2_dst_md = memory::desc({conv2_dst_tz}, dt::f32, tag::any);

    auto conv2_desc = convolution_forward::desc(prop_kind::forward,
            algorithm::convolution_direct, conv2_src_md, conv2_weights_md,
            conv2_bias_md, conv2_dst_md, conv2_strides, conv2_padding,
            conv2_padding);
    auto conv2_pd = convolution_forward::primitive_desc(conv2_desc, eng);

    // create reorder primitives between user input and conv src if needed
    auto conv2_src_memory = pool1_dst_memory;
    if (conv2_pd.src_desc() != conv2_src_memory.get_desc()) {
        conv2_src_memory = memory(conv2_pd.src_desc(), eng);
        net_fwd.push_back(reorder(pool1_dst_memory, conv2_src_memory));
        net_fwd_args.push_back({{DNNL_ARG_FROM, pool1_dst_memory},
                {DNNL_ARG_TO, conv2_src_memory}});
    }

    auto conv2_weights_memory = conv2_user_weights_memory;
    if (conv2_pd.weights_desc() != conv2_user_weights_memory.get_desc()) {
        conv2_weights_memory = memory(conv2_pd.weights_desc(), eng);
        net_fwd.push_back(
                reorder(conv2_user_weights_memory, conv2_weights_memory));
        net_fwd_args.push_back({{DNNL_ARG_FROM, conv2_user_weights_memory},
                {DNNL_ARG_TO, conv2_weights_memory}});
    }

    auto conv2_bias_memory = conv2_user_bias_memory;
    if (conv2_pd.bias_desc() != conv2_user_bias_memory.get_desc()) {
        conv2_bias_memory = memory(conv2_pd.bias_desc(), eng);
        net_fwd.push_back(
            reorder(conv2_user_bias_memory, conv2_bias_memory)
        );
        net_fwd_args.push_back({{DNNL_ARG_FROM, conv2_user_bias_memory}, {DNNL_ARG_TO, conv2_bias_memory}});
    }

    // create memory for conv dst
    auto conv2_dst_memory = memory(conv2_pd.dst_desc(), eng);

    // finally create a convolution primitive
    net_fwd.push_back(convolution_forward(conv2_pd));
    net_fwd_args.push_back({{DNNL_ARG_SRC, conv2_src_memory},
            {DNNL_ARG_WEIGHTS, conv2_weights_memory},
            {DNNL_ARG_BIAS, conv2_bias_memory},
            {DNNL_ARG_DST, conv2_dst_memory}});

    // VGG11: block 2-2 ReLU
    auto relu2_desc = eltwise_forward::desc(prop_kind::forward,
            algorithm::eltwise_relu, conv2_dst_memory.get_desc(),
            negative_slope);
    auto relu2_pd = eltwise_forward::primitive_desc(relu2_desc, eng);

    // create relu dst memory
    auto relu2_dst_memory = memory(relu2_pd.dst_desc(), eng);

    net_fwd.push_back(eltwise_forward(relu2_pd));
    net_fwd_args.push_back({{DNNL_ARG_FROM, conv2_dst_memory},
            {DNNL_ARG_TO, relu2_dst_memory}});

    // VGG11: block 2-3 max_pooling2
    // {batch, 128, 112, 112} -> {batch, 128, 56, 56}
    // kernel: {2, 2}; strides: {2, 2}; padding: {0, 0}
    memory::dims pool2_dst_tz = {N, 128, 56, 56};
    memory::dims pool2_kernel = {2, 2};
    memory::dims pool2_strides = {2, 2};
    memory::dims pool2_padding = {0, 0};

    auto pool2_dst_md = memory::desc({pool2_dst_tz}, dt::f32, tag::any);

    //[Create pooling primitive]
    auto pool2_desc = pooling_forward::desc(prop_kind::forward,
            algorithm::pooling_max, relu2_dst_memory.get_desc(), pool2_dst_md,
            pool2_strides, pool2_kernel, pool2_padding, pool2_padding);
    auto pool2_pd = pooling_forward::primitive_desc(pool2_desc, eng);
    auto pool2_dst_memory = memory(pool2_pd.dst_desc(), eng);
    //[Create pooling primitive]

    // create pooling workspace memory if training
    auto pool2_workspace_memory = memory(pool2_pd.workspace_desc(), eng);

    net_fwd.push_back(pooling_forward(pool2_pd));
    net_fwd_args.push_back({{DNNL_ARG_SRC, relu2_dst_memory},
            {DNNL_ARG_DST, pool2_dst_memory}, // delay putting DST until reorder (if needed)
            {DNNL_ARG_WORKSPACE, pool2_workspace_memory}});

    // VGG11: block 3-1: conv3
    // {batch, 128, 56, 56} -> {batch, 256, 56, 56}
    // kernel: {3, 3}; strides: {1, 1}; padding: {1, 1}
    memory::dims conv3_src_tz = {N, 128, 56, 56};
    memory::dims conv3_weights_tz = {256, 128, 3, 3};
    memory::dims conv3_bias_tz = {256};
    memory::dims conv3_dst_tz = {N, 256, 56, 56};
    memory::dims conv3_strides = {1, 1};
    memory::dims conv3_padding = {1, 1};

    std::vector<float> conv3_weights(product(conv3_weights_tz));
    std::vector<float> conv3_bias(product(conv3_bias_tz));

    // initializing non-zero values for weights and bias
    for (size_t i = 0; i < conv3_weights.size(); ++i)
        conv3_weights[i] = sinf((float)i);
    for (size_t i = 0; i < conv3_bias.size(); ++i)
        conv3_bias[i] = sinf((float)i);

    // create memory for user data
    auto conv3_user_weights_memory
            = memory({{conv3_weights_tz}, dt::f32, tag::oihw}, eng);
    write_to_dnnl_memory(conv3_weights.data(), conv3_user_weights_memory);
    auto conv3_user_bias_memory = memory({{conv3_bias_tz}, dt::f32, tag::x}, eng);
    write_to_dnnl_memory(conv3_bias.data(), conv3_user_bias_memory);

    auto conv3_src_md = memory::desc({conv3_src_tz}, dt::f32, tag::any);
    auto conv3_bias_md = memory::desc({conv3_bias_tz}, dt::f32, tag::any);
    auto conv3_weights_md = memory::desc({conv3_weights_tz}, dt::f32, tag::any);
    auto conv3_dst_md = memory::desc({conv3_dst_tz}, dt::f32, tag::any);

    auto conv3_desc = convolution_forward::desc(prop_kind::forward,
            algorithm::convolution_direct, conv3_src_md, conv3_weights_md,
            conv3_bias_md, conv3_dst_md, conv3_strides, conv3_padding,
            conv3_padding);
    auto conv3_pd = convolution_forward::primitive_desc(conv3_desc, eng);

    // create reorder primitives between user input and conv src if needed
    auto conv3_src_memory = pool2_dst_memory;
    if (conv3_pd.src_desc() != conv3_src_memory.get_desc()) {
        conv3_src_memory = memory(conv3_pd.src_desc(), eng);
        net_fwd.push_back(reorder(pool2_dst_memory, conv3_src_memory));
        net_fwd_args.push_back({{DNNL_ARG_FROM, pool2_dst_memory},
                {DNNL_ARG_TO, conv3_src_memory}});
    }

    auto conv3_weights_memory = conv3_user_weights_memory;
    if (conv3_pd.weights_desc() != conv3_user_weights_memory.get_desc()) {
        conv3_weights_memory = memory(conv3_pd.weights_desc(), eng);
        net_fwd.push_back(
                reorder(conv3_user_weights_memory, conv3_weights_memory));
        net_fwd_args.push_back({{DNNL_ARG_FROM, conv3_user_weights_memory},
                {DNNL_ARG_TO, conv3_weights_memory}});
    }

    auto conv3_bias_memory = conv3_user_bias_memory;
    if (conv3_pd.bias_desc() != conv3_user_bias_memory.get_desc()) {
        conv3_bias_memory = memory(conv3_pd.bias_desc(), eng);
        net_fwd.push_back(
            reorder(conv3_user_bias_memory, conv3_bias_memory)
        );
        net_fwd_args.push_back({{DNNL_ARG_FROM, conv3_user_bias_memory},
                {DNNL_ARG_TO, conv3_bias_memory}});
    }

    // create memory for conv dst
    auto conv3_dst_memory = memory(conv3_pd.dst_desc(), eng);

    // finally create a convolution primitive
    net_fwd.push_back(convolution_forward(conv3_pd));
    net_fwd_args.push_back({{DNNL_ARG_SRC, conv3_src_memory},
            {DNNL_ARG_WEIGHTS, conv3_weights_memory},
            {DNNL_ARG_BIAS, conv3_bias_memory},
            {DNNL_ARG_DST, conv3_dst_memory}});

    // VGG11: block3: ReLU3
    auto relu3_desc = eltwise_forward::desc(prop_kind::forward,
            algorithm::eltwise_relu, conv3_dst_memory.get_desc(),
            negative_slope);
    auto relu3_pd = eltwise_forward::primitive_desc(relu3_desc, eng);

    // create relu dst memory
    auto relu3_dst_memory = memory(relu3_pd.dst_desc(), eng);

    net_fwd.push_back(eltwise_forward(relu3_pd));
    net_fwd_args.push_back({{DNNL_ARG_FROM, conv3_dst_memory},
            {DNNL_ARG_TO, relu3_dst_memory}});

    // VGG11: block3: conv4
    // {batch, 256, 56, 56} -> {batch, 256, 56, 56}
    memory::dims conv4_src_tz = {N, 256, 56, 56};
    memory::dims conv4_weights_tz = {256, 256, 3, 3};
    memory::dims conv4_bias_tz = {256};
    memory::dims conv4_dst_tz = {N, 256, 56, 56};
    memory::dims conv4_strides = {1, 1};
    memory::dims conv4_padding = {1, 1};

    std::vector<float> conv4_weights(product(conv4_weights_tz));
    std::vector<float> conv4_bias(product(conv4_bias_tz));

    // initializing non-zero values for weights and bias
    for (size_t i = 0; i < conv4_weights.size(); ++i)
        conv4_weights[i] = sinf((float)i);
    for (size_t i = 0; i < conv4_bias.size(); ++i)
        conv4_bias[i] = sinf((float)i);

    // create memory for user data
    auto conv4_user_weights_memory
            = memory({{conv4_weights_tz}, dt::f32, tag::oihw}, eng);
    write_to_dnnl_memory(conv4_weights.data(), conv4_user_weights_memory);
    auto conv4_user_bias_memory = memory({{conv4_bias_tz}, dt::f32, tag::x}, eng);
    write_to_dnnl_memory(conv4_bias.data(), conv4_user_bias_memory);

    auto conv4_src_md = memory::desc({conv4_src_tz}, dt::f32, tag::any);
    auto conv4_bias_md = memory::desc({conv4_bias_tz}, dt::f32, tag::any);
    auto conv4_weights_md = memory::desc({conv4_weights_tz}, dt::f32, tag::any);
    auto conv4_dst_md = memory::desc({conv4_dst_tz}, dt::f32, tag::any);

    auto conv4_desc = convolution_forward::desc(prop_kind::forward,
            algorithm::convolution_direct, conv4_src_md, conv4_weights_md,
            conv4_bias_md, conv4_dst_md, conv4_strides, conv4_padding,
            conv4_padding);
    auto conv4_pd = convolution_forward::primitive_desc(conv4_desc, eng);

    // create reorder primitives between user input and conv src if needed
    auto conv4_src_memory = relu3_dst_memory;
    if (conv4_pd.src_desc() != conv4_src_memory.get_desc()) {
        conv4_src_memory = memory(conv4_pd.src_desc(), eng);
        net_fwd.push_back(reorder(relu3_dst_memory, conv4_src_memory));
        net_fwd_args.push_back({{DNNL_ARG_FROM, relu3_dst_memory},
                {DNNL_ARG_TO, conv4_src_memory}});
    }

    auto conv4_weights_memory = conv4_user_weights_memory;
    if (conv4_pd.weights_desc() != conv4_user_weights_memory.get_desc()) {
        conv4_weights_memory = memory(conv4_pd.weights_desc(), eng);
        net_fwd.push_back(
                reorder(conv4_user_weights_memory, conv4_weights_memory));
        net_fwd_args.push_back({{DNNL_ARG_FROM, conv4_user_weights_memory},
                {DNNL_ARG_TO, conv4_weights_memory}});
    }

    auto conv4_bias_memory = conv4_user_bias_memory;
    if (conv4_pd.bias_desc() != conv4_user_bias_memory.get_desc()) {
        conv4_bias_memory = memory(conv4_pd.bias_desc(), eng);
        net_fwd.push_back(
            reorder(conv4_user_bias_memory, conv4_bias_memory)
        );
        net_fwd_args.push_back({{DNNL_ARG_FROM, conv4_user_bias_memory},
                {DNNL_ARG_TO, conv4_bias_memory}});
    }

    // create memory for conv dst
    auto conv4_dst_memory = memory(conv4_pd.dst_desc(), eng);

    // finally create a convolution primitive
    net_fwd.push_back(convolution_forward(conv4_pd));
    net_fwd_args.push_back({{DNNL_ARG_SRC, conv4_src_memory},
            {DNNL_ARG_WEIGHTS, conv4_weights_memory},
            {DNNL_ARG_BIAS, conv4_bias_memory},
            {DNNL_ARG_DST, conv4_dst_memory}});

    // VGG11: block3: ReLU4
    auto relu4_desc = eltwise_forward::desc(prop_kind::forward,
            algorithm::eltwise_relu, conv4_dst_memory.get_desc(),
            negative_slope);
    auto relu4_pd = eltwise_forward::primitive_desc(relu4_desc, eng);

    // create relu dst memory
    auto relu4_dst_memory = memory(relu4_pd.dst_desc(), eng);

    net_fwd.push_back(eltwise_forward(relu4_pd));
    net_fwd_args.push_back({{DNNL_ARG_FROM, conv4_dst_memory},
            {DNNL_ARG_TO, relu4_dst_memory}});

    // VGG11: block3: max_pooling3
    // {batch, 256, 56, 56} -> {batch, 256, 28, 28}
    // kernel: {2, 2}; strides: {2, 2}; padding: {1, 1}
    memory::dims pool3_dst_tz = {N, 256, 28, 28};
    memory::dims pool3_kernel = {2, 2};
    memory::dims pool3_strides = {2, 2};
    memory::dims pool3_padding = {0, 0};

    auto pool3_dst_md = memory::desc({pool3_dst_tz}, dt::f32, tag::any);

    //[Create pooling primitive]
    auto pool3_desc = pooling_forward::desc(prop_kind::forward,
            algorithm::pooling_max, relu4_dst_memory.get_desc(), pool3_dst_md,
            pool3_strides, pool3_kernel, pool3_padding, pool3_padding);
    auto pool3_pd = pooling_forward::primitive_desc(pool3_desc, eng);
    auto pool3_dst_memory = memory(pool3_pd.dst_desc(), eng);
    //[Create pooling primitive]

    // create pooling workspace memory if training
    auto pool3_workspace_memory = memory(pool3_pd.workspace_desc(), eng);

    net_fwd.push_back(pooling_forward(pool3_pd));
    net_fwd_args.push_back({{DNNL_ARG_SRC, relu4_dst_memory},
            {DNNL_ARG_DST, pool3_dst_memory}, // delay putting DST until reorder (if needed)
            {DNNL_ARG_WORKSPACE, pool3_workspace_memory}});

    // VGG11: block4: conv5
    // {batch, 256, 28, 28} -> {batch, 512, 28, 28}
    // kernel: {3, 3}; strides: {1, 1}; padding: {1, 1}
    memory::dims conv5_src_tz = {N, 256, 28, 28};
    memory::dims conv5_weights_tz = {512, 256, 3, 3};
    memory::dims conv5_bias_tz = {512};
    memory::dims conv5_dst_tz = {N, 512, 28, 28};
    memory::dims conv5_strides = {1, 1};
    memory::dims conv5_padding = {1, 1};

    std::vector<float> conv5_weights(product(conv5_weights_tz));
    std::vector<float> conv5_bias(product(conv5_bias_tz));

    // initializing non-zero values for weights and bias
    for (size_t i = 0; i < conv5_weights.size(); ++i)
        conv5_weights[i] = sinf((float)i);
    for (size_t i = 0; i < conv5_bias.size(); ++i)
        conv5_bias[i] = sinf((float)i);

    // create memory for user data
    auto conv5_user_weights_memory
            = memory({{conv5_weights_tz}, dt::f32, tag::oihw}, eng);
    write_to_dnnl_memory(conv5_weights.data(), conv5_user_weights_memory);
    auto conv5_user_bias_memory = memory({{conv5_bias_tz}, dt::f32, tag::x}, eng);
    write_to_dnnl_memory(conv5_bias.data(), conv5_user_bias_memory);

    auto conv5_src_md = memory::desc({conv5_src_tz}, dt::f32, tag::any);
    auto conv5_bias_md = memory::desc({conv5_bias_tz}, dt::f32, tag::any);
    auto conv5_weights_md = memory::desc({conv5_weights_tz}, dt::f32, tag::any);
    auto conv5_dst_md = memory::desc({conv5_dst_tz}, dt::f32, tag::any);

    auto conv5_desc = convolution_forward::desc(prop_kind::forward,
            algorithm::convolution_direct, conv5_src_md, conv5_weights_md,
            conv5_bias_md, conv5_dst_md, conv5_strides, conv5_padding,
            conv5_padding);
    auto conv5_pd = convolution_forward::primitive_desc(conv5_desc, eng);

    // create reorder primitives between user input and conv src if needed
    auto conv5_src_memory = pool3_dst_memory;
    if (conv5_pd.src_desc() != conv5_src_memory.get_desc()) {
        conv5_src_memory = memory(conv5_pd.src_desc(), eng);
        net_fwd.push_back(reorder(pool3_dst_memory, conv5_src_memory));
        net_fwd_args.push_back({{DNNL_ARG_FROM, pool3_dst_memory},
                {DNNL_ARG_TO, conv5_src_memory}});
    }

    auto conv5_weights_memory = conv5_user_weights_memory;
    if (conv5_pd.weights_desc() != conv5_user_weights_memory.get_desc()) {
        conv5_weights_memory = memory(conv5_pd.weights_desc(), eng);
        net_fwd.push_back(
                reorder(conv5_user_weights_memory, conv5_weights_memory));
        net_fwd_args.push_back({{DNNL_ARG_FROM, conv5_user_weights_memory},
                {DNNL_ARG_TO, conv5_weights_memory}});
    }

    auto conv5_bias_memory = conv5_user_bias_memory;
    if (conv5_pd.bias_desc() != conv5_user_bias_memory.get_desc()) {
        conv5_bias_memory = memory(conv5_pd.bias_desc(), eng);
        net_fwd.push_back(
            reorder(conv5_user_bias_memory, conv5_bias_memory)
        );
        net_fwd_args.push_back({{DNNL_ARG_FROM, conv5_user_bias_memory},
                {DNNL_ARG_TO, conv5_bias_memory}});
    }

    // create memory for conv dst
    auto conv5_dst_memory = memory(conv5_pd.dst_desc(), eng);

    // finally create a convolution primitive
    net_fwd.push_back(convolution_forward(conv5_pd));
    net_fwd_args.push_back({{DNNL_ARG_SRC, conv5_src_memory},
            {DNNL_ARG_WEIGHTS, conv5_weights_memory},
            {DNNL_ARG_BIAS, conv5_bias_memory},
            {DNNL_ARG_DST, conv5_dst_memory}});
#endif


}


int main(int argc, char* argv[]) {

#ifdef DEBUG
    // MNIST_DATA_LOCATION set by MNIST cmake config
    std::cout << "MNIST data directory: " << MNIST_DATA_LOCATION << std::endl;
    std::cout << "MNIST fashion data directory: " << MNIST_FASHION_DATA_LOCATION << std::endl;

    // Load MNIST data
    // {
    //     mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
    //         mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(MNIST_DATA_LOCATION);

    //     std::cout << "Nbr of training images = " << dataset.training_images.size() << std::endl;
    //     std::cout << "Nbr of training labels = " << dataset.training_labels.size() << std::endl;
    //     std::cout << "Nbr of test images = " << dataset.test_images.size() << std::endl;
    //     std::cout << "Nbr of test labels = " << dataset.test_labels.size() << std::endl;
    // }

    // Load fashion MNIST data
    mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
        mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(MNIST_FASHION_DATA_LOCATION, 240, 40);

// #ifdef DEBUG
//     std::cout << "Nbr of training images = " << dataset.training_images.size() << std::endl;
//     std::cout << "Nbr of training labels = " << dataset.training_labels.size() << std::endl;
//     std::cout << "Nbr of test images = " << dataset.test_images.size() << std::endl;
//     std::cout << "Nbr of test labels = " << dataset.test_labels.size() << std::endl;
// #endif

    auto a = img_res.data;
    for (size_t i = 25242; i < 25242+33; ++i)
        std::cout << (float)*(a++);

    return 0;
#endif

#ifndef DEBUG
    return handle_example_errors(VGG11, parse_engine_kind(argc, argv));
#endif

}

Conv2DwithReLu::Conv2DwithReLu(dnnl::engine eng, std::vector<primitive> &net,
            std::vector<std::unordered_map<int, memory>> &net_args,
            const memory &src_memory, const memory::dims &src_tz,
            const memory::dims &dst_tz, const memory::dims &weights_tz,
            const memory::dims &strides, const memory::dims &padding,
            const float &negative_slope):
            weights(product(weights_tz)),
            bias(weights_tz.at(0))
{
    // initializing non-zero values for weights and bias
    for (size_t i = 0; i < weights.size(); ++i)
        weights[i] = sinf((float)i);
    for (size_t i = 0; i < bias.size(); ++i)
        bias[i] = sinf((float)i);

    memory::dims bias_tz = {weights_tz[0]};

#ifdef USEREORDER
    auto user_weights_memory
            = memory({{weights_tz}, dt::f32, tag::oihw}, eng);
    write_to_dnnl_memory(weights.data(), user_weights_memory);
    auto user_bias_memory = memory({{bias_tz}, dt::f32, tag::x}, eng);
    write_to_dnnl_memory(bias.data(), user_bias_memory);
#endif

#ifndef USEREORDER
    auto weights_memory
            = memory({{weights_tz}, dt::f32, tag::oihw}, eng);
    write_to_dnnl_memory(weights.data(), weights_memory);
    auto bias_memory = memory({{bias_tz}, dt::f32, tag::x}, eng);
    write_to_dnnl_memory(bias.data(), bias_memory);
#endif

    auto src_md = memory::desc({src_tz}, dt::f32, tag::any);
    auto bias_md = memory::desc({bias_tz}, dt::f32, tag::any);
    auto weights_md = memory::desc({weights_tz}, dt::f32, tag::any);
    auto dst_md = memory::desc({dst_tz}, dt::f32, tag::any);

    auto desc = convolution_forward::desc(prop_kind::forward,
            algorithm::convolution_direct, src_md, weights_md,
            bias_md, dst_md, strides, padding,
            padding);
    auto pd = convolution_forward::primitive_desc(desc, eng);

#ifdef USEREORDER
    // create reorder primitives between user input and conv src if needed
    auto weights_memory = user_weights_memory;
    if (pd.weights_desc() != user_weights_memory.get_desc()) {
        weights_memory = memory(pd.weights_desc(), eng);
        net.push_back(
                reorder(user_weights_memory, weights_memory));
        net_args.push_back({{DNNL_ARG_FROM, user_weights_memory},
                {DNNL_ARG_TO, weights_memory}});
    }

    // added by rbj (159 modified as well)
    auto bias_memory = user_bias_memory;
    if (pd.bias_desc() != user_bias_memory.get_desc()) {
        bias_memory = memory(pd.bias_desc(), eng);
        net.push_back(
            reorder(user_bias_memory, bias_memory)
        );
        net_args.push_back({{DNNL_ARG_FROM, user_bias_memory}, {DNNL_ARG_TO, bias_memory}});
    }
#endif

    // create memory for conv dst
    auto conv_dst_memory = memory(pd.dst_desc(), eng);

    // finally create a convolution primitive
    net.push_back(convolution_forward(pd));
    net_args.push_back({{DNNL_ARG_SRC, src_memory},
            {DNNL_ARG_WEIGHTS, weights_memory},
            {DNNL_ARG_BIAS, bias_memory},
            {DNNL_ARG_DST, conv_dst_memory}});

    // ReLU
    auto relu_desc = eltwise_forward::desc(prop_kind::forward_training,
            algorithm::eltwise_relu, conv_dst_memory.get_desc(),
            negative_slope);
    auto relu_pd = eltwise_forward::primitive_desc(relu_desc, eng);

    // create relu dst memory
    auto relu_dst_memory = memory(relu_pd.dst_desc(), eng);

    net.push_back(eltwise_forward(relu_pd));
    net_args.push_back({{DNNL_ARG_SRC, conv_dst_memory},
            {DNNL_ARG_DST, relu_dst_memory}});
    dst_m = relu_dst_memory;
}



MaxPooling::MaxPooling(dnnl::engine eng, std::vector<primitive> &net,
            std::vector<std::unordered_map<int, memory>> &net_args,
            const memory &src_memory,
            const memory::dims &kernel,
            const memory::dims &dst_tz, const memory::dims &strides,
            const memory::dims &padding, bool trained): iftrain(trained)
{
    auto dst_md = memory::desc({dst_tz}, dt::f32, tag::any);

    //[Create pooling primitive]
    auto desc = pooling_forward::desc(prop_kind::forward_training,
            algorithm::pooling_max, src_memory.get_desc(), dst_md,
            strides, kernel, padding, padding);
    auto pd = pooling_forward::primitive_desc(desc, eng);
    auto dst_memory = memory(pd.dst_desc(), eng);
    //[Create pooling primitive]


    net.push_back(pooling_forward(pd));
    net_args.push_back({{DNNL_ARG_SRC, src_memory},
            {DNNL_ARG_DST, dst_memory}});
            // {DNNL_ARG_WORKSPACE, workspace_memory}

    // create pooling workspace memory if training
    if (trained) {
        auto workspace_memory = memory(pd.workspace_desc(), eng);
        net_args.back().insert({DNNL_ARG_WORKSPACE, workspace_memory});
    }

    dst_m = dst_memory;
}

Dense::Dense(dnnl::engine eng, std::vector<primitive> &net,
            std::vector<std::unordered_map<int, memory>> &net_args,
            const memory &src_memory, const memory::dims &src_tz,
            const memory::dims &dst_tz,
            const memory::dims &weights_tz):
            weights(product(weights_tz)),
            bias(weights_tz.at(0))
{
    // initializing non-zero values for weights and bias
    for (size_t i = 0; i < weights.size(); ++i)
        weights[i] = sinf((float)i);
    for (size_t i = 0; i < bias.size(); ++i)
        bias[i] = sinf((float)i);

    memory::dims bias_tz = {weights_tz[0]};

    // create memory for user data
    auto weights_memory
            = memory({{weights_tz}, dt::f32, (weights_tz.size() == 2 ? tag::oi : tag::oihw)}, eng);
    write_to_dnnl_memory(weights.data(), weights_memory);
    auto bias_memory = memory({{bias_tz}, dt::f32, tag::x}, eng);
    write_to_dnnl_memory(bias.data(), bias_memory);

    // create memory descriptors for convolution data w/ no specified format
    auto src_md = memory::desc({src_tz}, dt::f32, tag::any);
    auto bias_md = memory::desc({bias_tz}, dt::f32, tag::any);
    auto weights_md = memory::desc({weights_tz}, dt::f32, tag::any);
    auto dst_md = memory::desc({dst_tz}, dt::f32, tag::any);

    // create a inner_product
    auto desc = inner_product_forward::desc(prop_kind::forward_training,
            src_md, weights_md, bias_md, dst_md);
    auto pd = inner_product_forward::primitive_desc(desc, eng);

    auto dst_memory = memory(pd.dst_desc(), eng);

    // create convolution primitive and add it to net
    net.push_back(inner_product_forward(pd));
    net_args.push_back({{DNNL_ARG_SRC, src_memory},
            {DNNL_ARG_WEIGHTS, weights_memory},
            {DNNL_ARG_BIAS, bias_memory},
            {DNNL_ARG_DST, dst_memory}});

    dst_m = dst_memory;
}

CrossEntropyLoss::CrossEntropyLoss(dnnl::engine eng, std::vector<primitive> &net,
                            std::vector<std::unordered_map<int, memory>> &net_args,
                            const memory &y_hat_memory, const memory &y_true_memory,
                            const memory::dims &y_tz)
{

    // 0) Clip y_hat to avoid performing log(0)

    float lower = 1e-7; // alpha
    float upper = 1-1e-7; // beta

    auto y_md = memory::desc({y_tz}, dt::f32, tag::nc);
    auto y_hat_cliped_memory = memory(y_md, eng);

    auto clip_desc = eltwise_forward::desc(prop_kind::forward_training, algorithm::eltwise_clip,
                                                y_md, lower, upper);
    auto clip_pd = eltwise_forward::primitive_desc(clip_desc, eng);

    net.push_back(eltwise_forward(clip_pd));
    net_args.push_back({{DNNL_ARG_SRC, y_hat_memory},
                        {DNNL_ARG_DST, y_hat_cliped_memory}});

    // 1) Perform elementwise log on y_hat_cliped
    auto y_hat_logged_memory = memory(y_md, eng);

    auto log_desc = eltwise_forward::desc(prop_kind::forward_training, algorithm::eltwise_log, y_md);
    auto log_pd = eltwise_forward::primitive_desc(log_desc, eng);

    net.push_back(eltwise_forward(log_pd));
    net_args.push_back({{DNNL_ARG_SRC, y_hat_cliped_memory},
                        {DNNL_ARG_DST, y_hat_logged_memory}});


    return;
}