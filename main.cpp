#include <assert.h>
#include <math.h>
#include <iostream>
// export CPLUS_INCLUDE_PATH=/home/cauchy/github/mnist-fashion/include:$CPLUS_INCLUDE_PATH
#include "mnist/mnist_reader.hpp"
// export CPLUS_INCLUDE_PATH=/usr/local/include/opencv4:$CPLUS_INCLUDE_PATH
#include <opencv2/opencv.hpp>
#include "my_layers.hpp"

using namespace dnnl;

// #define DEBUG
#define MODIFY
// #define USEREORDER

const std::string MNIST_DATA_LOCATION =
    "/home/cauchy/github/mnist-fashion/data/mnist";
const std::string MNIST_FASHION_DATA_LOCATION =
    "/home/cauchy/github/mnist-fashion/data/fashion";
const memory::dim N = 16;  // batch_size

// get fasion-mnist
mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
    mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(
        MNIST_FASHION_DATA_LOCATION, 240, 40);
memory::dim train_t = 0;
memory::dim test_t = 0;

using tag = memory::format_tag;
using dt = memory::data_type;

void VGG11(engine::kind engine_kind) {

    auto eng = engine(engine_kind, 0);
    stream s(eng);

    // Vector of primitives and their execute arguments
    std::vector<primitive> net_fwd, net_bwd;
    std::vector<std::unordered_map<int, memory>> net_fwd_args, net_bwd_args;

    // Vectors of input data and expected output
    std::vector<float> net_src(N * 3 * 224 * 224);
    std::vector<float> net_dst(N * 10);  // 10 classes

    const int IS = 224 * 224;  // input size

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
                img.at<uint8_t>(i, j) = (uint8_t)pic[i * 28 + j];

        cv::Mat img_rgb(28, 28, CV_8UC3);
        cv::merge(std::vector<cv::Mat>{img, img, img}, img_rgb);

        cv::Mat img_res(224, 224, CV_8UC3);
        cv::resize(img_rgb, img_res, cv::Size(), 8, 8,
                   cv::INTER_LINEAR);  //INTER_CUBIC slower

        auto data = img_res.data;

        int fpi = i * 224 * 224 * 3;  // first pixel index

        // write data into src while doing normalization (divided by 255)
        for (size_t c = 0; c < 3; ++c)  // channel
            for (size_t w = 0; w < 224; ++w)
                for (size_t h = 0; h < 224; ++h)
                    net_src[fpi + c * IS + w * 224 + h] =
                        ((float)(*(data + w * 672 + h * 3 + c))) / 255.0;

        // write data into dst
        net_dst[i * 10 + ans] = 1;
    }

    auto net_dst_memory =
        memory({{memory::dims{N, 10}}, dt::f32, tag::nc}, eng);
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

    auto conv1_src_memory = memory({{conv1_src_tz}, dt::f32, tag::nchw}, eng);
    write_to_dnnl_memory(net_src.data(), conv1_src_memory);

    Conv2DwithReLu conv1(eng, net_fwd, net_fwd_args, conv1_src_memory,
                         conv1_src_tz, conv1_dst_tz, conv1_weights_tz,
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
    MaxPooling pool1(eng, net_fwd, net_fwd_args, conv1_dst_memory, pool1_kernel,
                     pool1_dst_tz, pool1_strides, pool1_padding);
    memory pool1_dst_memory = pool1.dst_memory();

    // VGG11: block 2-1: conv2
    // {batch, 64, 112, 112} -> {batch, 128, 112, 112}
    // kernel: {3, 3}; strides: {1, 1}; padding: {1, 1}
    memory::dims conv2_src_tz = {N, 64, 112, 112};
    memory::dims conv2_weights_tz = {128, 64, 3, 3};
    memory::dims conv2_dst_tz = {N, 128, 112, 112};
    memory::dims conv2_strides = {1, 1};
    memory::dims conv2_padding = {1, 1};
    Conv2DwithReLu conv2(eng, net_fwd, net_fwd_args, pool1_dst_memory,
                         conv2_src_tz, conv2_dst_tz, conv2_weights_tz,
                         conv2_strides, conv2_padding, negative_slope);
    memory conv2_dst_memory = conv2.dst_memory();

    // VGG11: block 2-2 max_pooling2
    // {batch, 128, 112, 112} -> {batch, 128, 56, 56}
    // kernel: {2, 2}; strides: {2, 2}; padding: {0, 0}
    memory::dims pool2_dst_tz = {N, 128, 56, 56};
    memory::dims pool2_kernel = {2, 2};
    memory::dims pool2_strides = {2, 2};
    memory::dims pool2_padding = {0, 0};
    MaxPooling pool2(eng, net_fwd, net_fwd_args, conv2_dst_memory, pool2_kernel,
                     pool2_dst_tz, pool2_strides, pool2_padding);
    memory pool2_dst_memory = pool2.dst_memory();

    // VGG11: block 3-1: conv3
    // {batch, 128, 56, 56} -> {batch, 256, 56, 56}
    // kernel: {3, 3}; strides: {1, 1}; padding: {1, 1}
    memory::dims conv3_src_tz = {N, 128, 56, 56};
    memory::dims conv3_weights_tz = {256, 128, 3, 3};
    memory::dims conv3_dst_tz = {N, 256, 56, 56};
    memory::dims conv3_strides = {1, 1};
    memory::dims conv3_padding = {1, 1};
    Conv2DwithReLu conv3(eng, net_fwd, net_fwd_args, pool2_dst_memory,
                         conv3_src_tz, conv3_dst_tz, conv3_weights_tz,
                         conv3_strides, conv3_padding, negative_slope);
    memory conv3_dst_memory = conv3.dst_memory();

    // VGG11: block 3-2: conv4
    // {batch, 256, 56, 56} -> {batch, 256, 56, 56}
    memory::dims conv4_src_tz = {N, 256, 56, 56};
    memory::dims conv4_weights_tz = {256, 256, 3, 3};
    memory::dims conv4_dst_tz = {N, 256, 56, 56};
    memory::dims conv4_strides = {1, 1};
    memory::dims conv4_padding = {1, 1};
    Conv2DwithReLu conv4(eng, net_fwd, net_fwd_args, conv3_dst_memory,
                         conv4_src_tz, conv4_dst_tz, conv4_weights_tz,
                         conv4_strides, conv4_padding, negative_slope);
    memory conv4_dst_memory = conv4.dst_memory();

    // VGG11: block 3-3: max_pooling3
    // {batch, 256, 56, 56} -> {batch, 256, 28, 28}
    // kernel: {2, 2}; strides: {2, 2}; padding: {1, 1}
    memory::dims pool3_dst_tz = {N, 256, 28, 28};
    memory::dims pool3_kernel = {2, 2};
    memory::dims pool3_strides = {2, 2};
    memory::dims pool3_padding = {0, 0};
    MaxPooling pool3(eng, net_fwd, net_fwd_args, conv4_dst_memory, pool3_kernel,
                     pool3_dst_tz, pool3_strides, pool3_padding);
    memory pool3_dst_memory = pool3.dst_memory();

    // VGG11: block 4-1: conv5
    // {batch, 256, 28, 28} -> {batch, 512, 28, 28}
    // kernel: {3, 3}; strides: {1, 1}; padding: {1, 1}
    memory::dims conv5_src_tz = {N, 256, 28, 28};
    memory::dims conv5_weights_tz = {512, 256, 3, 3};
    memory::dims conv5_dst_tz = {N, 512, 28, 28};
    memory::dims conv5_strides = {1, 1};
    memory::dims conv5_padding = {1, 1};
    Conv2DwithReLu conv5(eng, net_fwd, net_fwd_args, pool3_dst_memory,
                         conv5_src_tz, conv5_dst_tz, conv5_weights_tz,
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
    Conv2DwithReLu conv6(eng, net_fwd, net_fwd_args, conv5_dst_memory,
                         conv6_src_tz, conv6_dst_tz, conv6_weights_tz,
                         conv6_strides, conv6_padding, negative_slope);
    memory conv6_dst_memory = conv6.dst_memory();

    // VGG11: block 4-3: max_pooling4
    // {batch, 512, 28, 28} -> {batch, 512, 14, 14}
    // kernel: {2, 2}; strides: {2, 2}; padding: {1, 1}
    memory::dims pool4_dst_tz = {N, 512, 14, 14};
    memory::dims pool4_kernel = {2, 2};
    memory::dims pool4_strides = {2, 2};
    memory::dims pool4_padding = {0, 0};
    MaxPooling pool4(eng, net_fwd, net_fwd_args, conv6_dst_memory, pool4_kernel,
                     pool4_dst_tz, pool4_strides, pool4_padding);
    memory pool4_dst_memory = pool4.dst_memory();

    // VGG11: block 5-1: conv7
    // {batch, 512, 14, 14} -> {batch, 512, 14, 14}
    // kernel: {3, 3}; strides: {1, 1}; padding: {1, 1}
    memory::dims conv7_src_tz = {N, 512, 14, 14};
    memory::dims conv7_weights_tz = {512, 512, 3, 3};
    memory::dims conv7_dst_tz = {N, 512, 14, 14};
    memory::dims conv7_strides = {1, 1};
    memory::dims conv7_padding = {1, 1};
    Conv2DwithReLu conv7(eng, net_fwd, net_fwd_args, pool4_dst_memory,
                         conv7_src_tz, conv7_dst_tz, conv7_weights_tz,
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
    Conv2DwithReLu conv8(eng, net_fwd, net_fwd_args, conv7_dst_memory,
                         conv8_src_tz, conv8_dst_tz, conv8_weights_tz,
                         conv8_strides, conv8_padding, negative_slope);
    memory conv8_dst_memory = conv8.dst_memory();

    // VGG11: block 5-3: max_pooling5
    // {batch, 512, 14, 14} -> {batch, 512, 7, 7}
    // kernel: {2, 2}; strides: {2, 2}; padding: {1, 1}
    memory::dims pool5_dst_tz = {N, 512, 7, 7};
    memory::dims pool5_kernel = {2, 2};
    memory::dims pool5_strides = {2, 2};
    memory::dims pool5_padding = {0, 0};
    MaxPooling pool5(eng, net_fwd, net_fwd_args, conv8_dst_memory, pool5_kernel,
                     pool5_dst_tz, pool5_strides, pool5_padding);
    memory pool5_dst_memory = pool5.dst_memory();

    // VGG11: FC4096*2
    // {batch, 512, 7, 7} -> {batch, 4096} -> {batch, 4096}
    memory::dims fc1_src_tz = {N, 512, 7, 7};
    memory::dims fc1_weights_tz = {4096, 512, 7, 7};
    memory::dims fc1_dst_tz = {N, 4096};
    Dense fc1(eng, net_fwd, net_fwd_args, pool5_dst_memory, fc1_src_tz,
              fc1_dst_tz, fc1_weights_tz);
    memory fc1_dst_memory = fc1.dst_memory();

    ReLU fc1_relu(eng, net_fwd, net_fwd_args, fc1_dst_memory, negative_slope);
    memory fc1_relu_dst_memory = fc1_relu.dst_memory();

    memory::dims fc2_src_tz = {N, 4096};
    memory::dims fc2_weights_tz = {4096, 4096};
    memory::dims fc2_dst_tz = {N, 4096};
    Dense fc2(eng, net_fwd, net_fwd_args, fc1_relu_dst_memory, fc2_src_tz,
              fc2_dst_tz, fc2_weights_tz);
    memory fc2_dst_memory = fc2.dst_memory();

    ReLU fc2_relu(eng, net_fwd, net_fwd_args, fc2_dst_memory, negative_slope);
    memory fc2_relu_dst_memory = fc2_relu.dst_memory();

    // VGG11: FC1000
    // {batch, 4096} -> {batch, 1000}
    memory::dims fc3_src_tz = {N, 4096};
    memory::dims fc3_weights_tz = {1000, 4096};
    memory::dims fc3_dst_tz = {N, 1000};
    Dense fc3(eng, net_fwd, net_fwd_args, fc2_relu_dst_memory, fc3_src_tz,
              fc3_dst_tz, fc3_weights_tz);
    memory fc3_dst_memory = fc3.dst_memory();

    ReLU fc3_relu(eng, net_fwd, net_fwd_args, fc3_dst_memory, negative_slope);
    memory fc3_relu_dst_memory = fc3_relu.dst_memory();

    // VGG11: FC10
    // {batch, 1000} -> {batch, 10}
    memory::dims fc4_src_tz = {N, 1000};
    memory::dims fc4_weights_tz = {10, 1000};
    memory::dims fc4_dst_tz = {N, 10};
    Dense fc4(eng, net_fwd, net_fwd_args, fc3_relu_dst_memory, fc4_src_tz,
              fc4_dst_tz, fc4_weights_tz);
    memory fc4_dst_memory = fc4.dst_memory();

    // VGG11: the end, softmax
    memory::dims softmax_src_tz = {N, 10};
    auto softmax_src_md = memory::desc(softmax_src_tz, dt::f32, tag::nc);
    auto softmax_dec =
        softmax_forward::desc(prop_kind::forward_training, softmax_src_md, 1);
    auto softmax_pd = softmax_forward::primitive_desc(softmax_dec, eng);
    auto softmax_dst_memory = memory(softmax_pd.dst_desc(), eng);

    net_fwd.push_back(softmax_forward(softmax_pd));
    net_fwd_args.push_back(
        {{DNNL_ARG_SRC, fc4_dst_memory}, {DNNL_ARG_DST, softmax_dst_memory}});

    memory::dims y_tz = {N, 10};
    // CrossEntropyLoss loss(eng, net_fwd, net_fwd_args, softmax_dst_memory, net_dst_memory, y_tz);

    // 0) Clip y_hat to avoid performing log(0)

    float lower = 1e-7;      // alpha
    float upper = 1 - 1e-7;  // beta

    auto y_md = memory::desc({y_tz}, dt::f32, tag::nc);
    auto y_hat_cliped_memory = memory(y_md, eng);

    auto clip_desc =
        eltwise_forward::desc(prop_kind::forward_training,
                              algorithm::eltwise_clip, y_md, lower, upper);
    auto clip_pd = eltwise_forward::primitive_desc(clip_desc, eng);

    net_fwd.push_back(eltwise_forward(clip_pd));
    net_fwd_args.push_back({{DNNL_ARG_SRC, softmax_dst_memory},
                            {DNNL_ARG_DST, y_hat_cliped_memory}});

    // 1) Perform elementwise log on y_hat_cliped
    auto y_hat_logged_memory = memory(y_md, eng);

    auto log_desc = eltwise_forward::desc(prop_kind::forward_training,
                                          algorithm::eltwise_log, y_md);
    auto log_pd = eltwise_forward::primitive_desc(log_desc, eng);

    net_fwd.push_back(eltwise_forward(log_pd));
    net_fwd_args.push_back({{DNNL_ARG_SRC, y_hat_cliped_memory},
                            {DNNL_ARG_DST, y_hat_logged_memory}});

    // using log(y_hat) and y_true to calculate cross entropy
    // wait until training
    memory::dims loss_tz = {N, 1};
    auto loss_md = memory::desc({loss_tz}, dt::f32, tag::nc);
    auto loss_memory = memory(loss_md, eng);

    //-----------------------------------------------------------------------
    //----------------- Backpropagation Stream  (Data)-------------------------------------

    // use loss and y_hat to calculate loss_diff({N, 10})
    // wait until training
    auto loss_diff_md = memory::desc({y_tz}, dt::f32, tag::nc);
    auto loss_diff_memory = memory(loss_diff_md, eng);

    // softmax back
    auto softmax_back_desc =
        softmax_backward::desc(loss_diff_md, softmax_src_md, 1);
    auto softmax_back_pd =
        softmax_backward::primitive_desc(softmax_back_desc, eng, softmax_pd);
    auto softmax_diff_src_memory = memory(softmax_src_md, eng);

    net_bwd.push_back(softmax_backward(softmax_back_pd));
    net_bwd_args.push_back({{DNNL_ARG_DIFF_DST, loss_diff_memory},
                            {DNNL_ARG_DST, softmax_dst_memory},
                            {DNNL_ARG_DIFF_SRC, softmax_diff_src_memory}});

    // fc4 back
    Dense_back fc4_back(eng, net_bwd, net_bwd_args, softmax_diff_src_memory,
                        fc3_relu_dst_memory, fc4_weights_tz, fc4);
    // fc3 ReLU back
    ReLU_back fc3_relu_back(eng, net_bwd, net_bwd_args,
                            fc4_back.diff_src_memory, fc3_dst_memory, fc3_relu);

    // fc3 back
    Dense_back fc3_back(eng, net_bwd, net_bwd_args,
                        fc3_relu_back.diff_src_memory, fc2_relu_dst_memory,
                        fc3_weights_tz, fc3);

    // fc2 ReLU back
    ReLU_back fc2_relu_back(eng, net_bwd, net_bwd_args,
                            fc3_back.diff_src_memory, fc2_dst_memory, fc2_relu);

    // fc2 back
    Dense_back fc2_back(eng, net_bwd, net_bwd_args,
                        fc2_relu_back.diff_src_memory, fc1_relu_dst_memory,
                        fc2_weights_tz, fc2);

    // fc1 ReLU back
    ReLU_back fc1_relu_back(eng, net_bwd, net_bwd_args,
                            fc2_back.diff_src_memory, fc1_dst_memory, fc1_relu);

    // fc1 back
    Dense_back fc1_back(eng, net_bwd, net_bwd_args,
                        fc1_relu_back.diff_src_memory, pool5_dst_memory,
                        fc1_weights_tz, fc1);

    // pool5 back
    MaxPooling_back pool5_back(
        eng, net_bwd, net_bwd_args, pool5_kernel, pool5_strides, pool5_padding,
        fc1_back.diff_src_memory, conv8_dst_memory, pool5);

    // conv8 back
    Conv2DwithReLu_back conv8_back(eng, net_bwd, net_bwd_args, conv8_weights_tz, conv8_strides, conv8_padding
                , pool5_back.diff_src_memory, conv7_dst_memory, conv8);


    return;
}

int main(int argc, char* argv[]) {

#ifdef DEBUG
    // MNIST_DATA_LOCATION set by MNIST cmake config
    std::cout << "MNIST data directory: " << MNIST_DATA_LOCATION << std::endl;
    std::cout << "MNIST fashion data directory: " << MNIST_FASHION_DATA_LOCATION
              << std::endl;

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
        mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(
            MNIST_FASHION_DATA_LOCATION, 240, 40);

    // #ifdef DEBUG
    //     std::cout << "Nbr of training images = " << dataset.training_images.size() << std::endl;
    //     std::cout << "Nbr of training labels = " << dataset.training_labels.size() << std::endl;
    //     std::cout << "Nbr of test images = " << dataset.test_images.size() << std::endl;
    //     std::cout << "Nbr of test labels = " << dataset.test_labels.size() << std::endl;
    // #endif

    auto a = img_res.data;
    for (size_t i = 25242; i < 25242 + 33; ++i)
        std::cout << (float)*(a++);

    return 0;
#endif

#ifndef DEBUG
    return handle_example_errors(VGG11, parse_engine_kind(argc, argv));
#endif
}