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

const std::string MNIST_DATA_LOCATION = "/home/cauchy/github/mnist-fashion/data/mnist";
const std::string MNIST_FASHION_DATA_LOCATION = "/home/cauchy/github/mnist-fashion/data/fashion";
const memory::dim N = 16; // batch_size

// get fasion-mnist
mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
    mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(MNIST_FASHION_DATA_LOCATION, 240, 40);
memory::dim train_t = 0;
memory::dim test_t = 0;


void VGG11(engine::kind engine_kind) {
    using tag = memory::format_tag;
    using dt = memory::data_type;

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

    // VGG11: block 1-1: conv1
    // {batch, 3, 224, 224} (x) {64, 3, 3, 3} -> {batch, 64, 224, 224}
    // kernel: {3,3}; strides: {1, 1}; padding: {1, 1}
    memory::dims conv1_src_tz = {N, 3, 224, 224};
    memory::dims conv1_weights_tz = {64, 3, 3, 3};
    memory::dims conv1_bias_tz = {64};
    memory::dims conv1_dst_tz = {N, 64, 224, 224};
    memory::dims conv1_strides = {1, 1};
    memory::dims conv1_padding = {1, 1};

    std::vector<float> conv1_weights(product(conv1_weights_tz));
    std::vector<float> conv1_bias(product(conv1_bias_tz));

    // initializing non-zero values for weights and bias
    for (size_t i = 0; i < conv1_weights.size(); ++i)
        conv1_weights[i] = sinf((float)i);
    for (size_t i = 0; i < conv1_bias.size(); ++i)
        conv1_bias[i] = sinf((float)i);

    auto conv1_user_src_memory
            = memory({{conv1_src_tz}, dt::f32, tag::nchw}, eng);
    write_to_dnnl_memory(net_src.data(), conv1_user_src_memory);
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
    const float negative_slope = 0.0f;

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
        net_fwd.push_back(reorder(conv2_src_memory, conv2_src_memory));
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

    return;
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
