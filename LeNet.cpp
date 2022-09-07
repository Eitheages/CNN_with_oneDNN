#include <opencv2/opencv.hpp>
#include "mnist/mnist_reader.hpp"
#include "my_layers.hpp"

// #define DEBUG

const std::string MNIST_DATA_LOCATION =
    "/home/cauchy/github/mnist-fashion/data/mnist";
const std::string MNIST_FASHION_DATA_LOCATION =
    "/home/cauchy/github/mnist-fashion/data/fashion";

// get fasion-mnist
mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
    mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(
        MNIST_FASHION_DATA_LOCATION, 0, 40);

// hyper parameter
const int N = 32;  // batch_size
const int step = 20;
const float LearningRate = 0.005;
const int epoch = 10;

void LeNet(engine::kind engine_kind) {
    // Vectors of input data and expected output
    std::vector<float> net_src(N * 1 * 28 * 28);
    std::vector<float> net_dst(N * 10);  // 10 classes

    const int IS = 28 * 28;

    auto eng = engine(engine_kind, 0);
    stream s(eng);

    // Vector of primitives and their execute arguments
    std::vector<primitive> net_fwd, net_bwd, net_sgd;
    std::vector<std::unordered_map<int, memory>> net_fwd_args, net_bwd_args,
        net_sgd_args;

    // LeNet conv1
    // {batch, 1, 28, 28} (x) {6, 1, 5, 5} -> {batch, 6, 28, 28}
    // padding = "same"
    memory::dims conv1_src_tz = {N, 1, 28, 28};
    memory::dims conv1_weights_tz = {6, 1, 5, 5};
    memory::dims conv1_dst_tz = {N, 6, 28, 28};
    memory::dims conv1_strides = {1, 1};
    memory::dims conv1_padding = {2, 2};

    auto conv1_src_memory = memory({{conv1_src_tz}, dt::f32, tag::nchw}, eng);

    Conv2DwithActi conv1(eng, net_fwd, net_fwd_args, conv1_src_memory,
                         conv1_src_tz, conv1_dst_tz, conv1_weights_tz,
                         conv1_strides, conv1_padding,
                         algorithm::eltwise_logistic);
    memory conv1_dst_memory = conv1.dst_memory();

    // LeNet pool1
    // {batch, 6, 28, 28} -> {batch, 6, 14, 14}
    // kernel = {2, 2}, strides = {2, 2}, padding = "valid"
    memory::dims pool1_dst_tz = {N, 6, 14, 14};
    memory::dims pool1_kernel = {2, 2};
    memory::dims pool1_strides = {2, 2};
    memory::dims pool1_padding = {0, 0};

    MeanPooling pool1(eng, net_fwd, net_fwd_args, conv1_dst_memory,
                      pool1_kernel, pool1_dst_tz, pool1_strides, pool1_padding,
                      algorithm::pooling_avg_include_padding);
    memory pool1_dst_memory = pool1.dst_memory();

    // LeNet conv2
    // {batch, 6, 14, 14} (x) {16, 6, 5, 5} -> {batch, 16, 10, 10}
    // padding = "valid"
    memory::dims conv2_src_tz = {N, 6, 14, 14};
    memory::dims conv2_weights_tz = {16, 6, 5, 5};
    memory::dims conv2_dst_tz = {N, 16, 10, 10};
    memory::dims conv2_strides = {1, 1};
    memory::dims conv2_padding = {0, 0};

    Conv2DwithActi conv2(eng, net_fwd, net_fwd_args, pool1_dst_memory,
                         conv2_src_tz, conv2_dst_tz, conv2_weights_tz,
                         conv2_strides, conv2_padding,
                         algorithm::eltwise_logistic);
    memory conv2_dst_memory = conv2.dst_memory();

    // LeNet pool2
    // {batch, 16, 10, 10} -> {batch, 16, 5, 5}
    // kernel = {2, 2}, strides = {2, 2}, padding = "valid"
    memory::dims pool2_dst_tz = {N, 16, 5, 5};
    memory::dims pool2_kernel = {2, 2};
    memory::dims pool2_strides = {2, 2};
    memory::dims pool2_padding = {0, 0};

    MeanPooling pool2(eng, net_fwd, net_fwd_args, conv2_dst_memory,
                      pool2_kernel, pool2_dst_tz, pool2_strides, pool2_padding,
                      algorithm::pooling_avg_include_padding);
    memory pool2_dst_memory = pool2.dst_memory();

    // LeNet fc1
    // {batch, 16, 5, 5} -> {batch, 120}
    memory::dims fc1_src_tz = {N, 16, 5, 5};
    memory::dims fc1_weights_tz = {120, 16, 5, 5};
    memory::dims fc1_dst_tz = {N, 120};
    Dense fc1(eng, net_fwd, net_fwd_args, pool2_dst_memory, fc1_src_tz,
              fc1_dst_tz, fc1_weights_tz);
    memory fc1_dst_memory = fc1.dst_memory();

    Eltwise fc1_sig(eng, net_fwd, net_fwd_args, fc1_dst_memory,
                    algorithm::eltwise_logistic);
    memory fc1_sig_dst_memory = fc1_sig.dst_memory();

    // LeNet fc2
    // {batch, 120} -> {batch, 84}
    memory::dims fc2_src_tz = {N, 120};
    memory::dims fc2_weights_tz = {84, 120};
    memory::dims fc2_dst_tz = {N, 84};
    Dense fc2(eng, net_fwd, net_fwd_args, fc1_sig_dst_memory, fc2_src_tz,
              fc2_dst_tz, fc2_weights_tz);
    memory fc2_dst_memory = fc2.dst_memory();

    Eltwise fc2_sig(eng, net_fwd, net_fwd_args, fc2_dst_memory,
                    algorithm::eltwise_logistic);
    memory fc2_sig_dst_memory = fc2_sig.dst_memory();

    // LeNet fc3
    // {batch, 84} -> {batch, 10}
    memory::dims fc3_src_tz = {N, 84};
    memory::dims fc3_weights_tz = {10, 84};
    memory::dims fc3_dst_tz = {N, 10};
    Dense fc3(eng, net_fwd, net_fwd_args, fc2_sig_dst_memory, fc3_src_tz,
              fc3_dst_tz, fc3_weights_tz);
    memory fc3_dst_memory = fc3.dst_memory();

    // LeNet softmax
    memory::dims softmax_src_tz = {N, 10};
    auto softmax_src_md = memory::desc(softmax_src_tz, dt::f32, tag::nc);
    auto softmax_dec =
        softmax_forward::desc(prop_kind::forward_training, softmax_src_md, 1);
    auto softmax_pd = softmax_forward::primitive_desc(softmax_dec, eng);
    auto softmax_dst_memory = memory(softmax_pd.dst_desc(), eng);

    net_fwd.push_back(softmax_forward(softmax_pd));
    net_fwd_args.push_back(
        {{DNNL_ARG_SRC, fc3_dst_memory}, {DNNL_ARG_DST, softmax_dst_memory}});

    memory::dims y_tz = {N, 10};

    // 0) Clip y_hat to avoid performing log(0)

    float lower = 1e-7;      // alpha
    float upper = 1 - 1e-7;  // beta

    auto y_md = memory::desc({y_tz}, dt::f32, tag::nc);
    auto y_hat_clipped_memory = memory(y_md, eng);

    auto clip_desc =
        eltwise_forward::desc(prop_kind::forward_training,
                              algorithm::eltwise_clip, y_md, lower, upper);
    auto clip_pd = eltwise_forward::primitive_desc(clip_desc, eng);

    net_fwd.push_back(eltwise_forward(clip_pd));
    net_fwd_args.push_back({{DNNL_ARG_SRC, softmax_dst_memory},
                            {DNNL_ARG_DST, y_hat_clipped_memory}});

    // 1) Perform elementwise log on y_hat_cliped
    auto y_hat_logged_memory = memory(y_md, eng);

    auto log_desc = eltwise_forward::desc(prop_kind::forward_training,
                                          algorithm::eltwise_log, y_md);
    auto log_pd = eltwise_forward::primitive_desc(log_desc, eng);

    net_fwd.push_back(eltwise_forward(log_pd));
    net_fwd_args.push_back({{DNNL_ARG_SRC, y_hat_clipped_memory},
                            {DNNL_ARG_DST, y_hat_logged_memory}});
    std::vector<float> y_hat_logged(product(y_tz));
    std::vector<float> y_hat_clipped(product(y_tz));

    // using log(y_hat) and y_true to calculate cross entropy
    // wait until training

    //-----------------------------------------------------------------------
    //----------------- Backpropagation Stream-------------------------------------
    std::vector<float> diff_y_hat(product(y_tz));
    auto diff_y_hat_md = memory::desc({y_tz}, dt::f32, tag::nc);
    auto diff_y_hat_memory = memory(diff_y_hat_md, eng);

    // softmax back
    auto softmax_back_desc =
        softmax_backward::desc(diff_y_hat_md, softmax_src_md, 1);
    auto softmax_back_pd =
        softmax_backward::primitive_desc(softmax_back_desc, eng, softmax_pd);
    auto softmax_diff_src_memory = memory(softmax_src_md, eng);

    net_bwd.push_back(softmax_backward(softmax_back_pd));
    net_bwd_args.push_back({{DNNL_ARG_DIFF_DST, diff_y_hat_memory},
                            {DNNL_ARG_DST, softmax_dst_memory},
                            {DNNL_ARG_DIFF_SRC, softmax_diff_src_memory}});

    // fc3 back
    Dense_back fc3_back(eng, net_bwd, net_bwd_args, softmax_diff_src_memory,
                        fc2_sig_dst_memory, fc3_weights_tz, fc3);

    // fc2 sigmoid back
    Eltwise_back fc2_sig_back(eng, net_bwd, net_bwd_args,
                              fc3_back.diff_src_memory, fc2_dst_memory, fc2_sig,
                              algorithm::eltwise_logistic);

    // fc2 back
    Dense_back fc2_back(eng, net_bwd, net_bwd_args,
                        fc2_sig_back.diff_src_memory, fc1_sig_dst_memory,
                        fc2_weights_tz, fc2);

    // fc1 sigmoid back
    Eltwise_back fc1_sig_back(eng, net_bwd, net_bwd_args,
                              fc2_back.diff_src_memory, fc1_dst_memory, fc1_sig,
                              algorithm::eltwise_logistic);

    // fc1 back
    Dense_back fc1_back(eng, net_bwd, net_bwd_args,
                        fc1_sig_back.diff_src_memory, pool2_dst_memory,
                        fc1_weights_tz, fc1);

    // pool2 back
    MeanPooling_back pool2_back(eng, net_bwd, net_bwd_args, pool2_kernel,
                                pool2_strides, pool2_padding,
                                fc1_back.diff_src_memory, conv2_dst_memory,
                                pool2, algorithm::pooling_avg_include_padding);

    // conv2 back
    Conv2DwithActi_back conv2_back(eng, net_bwd, net_bwd_args, conv2_weights_tz,
                                   conv2_strides, conv2_padding,
                                   pool2_back.diff_src_memory, pool1_dst_memory,
                                   conv2, algorithm::eltwise_logistic);

    // pool1 back
    MeanPooling_back pool1_back(eng, net_bwd, net_bwd_args, pool1_kernel,
                                pool1_strides, pool1_padding,
                                fc1_back.diff_src_memory, conv2_dst_memory,
                                pool1, algorithm::pooling_avg_include_padding);

    // conv1 back
    Conv2DwithActi_back conv1_back(eng, net_bwd, net_bwd_args, conv1_weights_tz,
                                   conv1_strides, conv1_padding,
                                   pool1_back.diff_src_memory, conv1_src_memory,
                                   conv1, algorithm::eltwise_logistic);

    //-----------------------------------------------------------------------
    //----------------- Weights update -------------------------------------
    updateWeights_SGD(conv1.weights_memory, conv1_back.diff_weights_memory,
                      LearningRate, net_sgd, net_sgd_args, eng);
    updateWeights_SGD(conv2.weights_memory, conv2_back.diff_weights_memory,
                      LearningRate, net_sgd, net_sgd_args, eng);
    updateWeights_SGD(fc1.weights_memory, fc1_back.diff_weights_memory,
                      LearningRate, net_sgd, net_sgd_args, eng);
    updateWeights_SGD(fc2.weights_memory, fc2_back.diff_weights_memory,
                      LearningRate, net_sgd, net_sgd_args, eng);
    updateWeights_SGD(fc3.weights_memory, fc3_back.diff_weights_memory,
                      LearningRate, net_sgd, net_sgd_args, eng);

    updateWeights_SGD(conv1.bias_memory, conv1_back.diff_bias_memory,
                      LearningRate, net_sgd, net_sgd_args, eng);
    updateWeights_SGD(conv2.bias_memory, conv2_back.diff_bias_memory,
                      LearningRate, net_sgd, net_sgd_args, eng);
    updateWeights_SGD(fc1.bias_memory, fc1_back.diff_bias_memory, LearningRate,
                      net_sgd, net_sgd_args, eng);
    updateWeights_SGD(fc2.bias_memory, fc2_back.diff_bias_memory, LearningRate,
                      net_sgd, net_sgd_args, eng);
    updateWeights_SGD(fc3.bias_memory, fc3_back.diff_bias_memory, LearningRate,
                      net_sgd, net_sgd_args, eng);

    // data index

    // --------------------- training ----------------------------
    std::vector<float> entropy(step);
    std::vector<float> loss(epoch);

    for (size_t k = 0; k < epoch; ++k) {
        std::cout << k + 1 << " th training..." << std::endl;
        int train_t = 0;
        int test_t = 0;
        for (size_t g = 0; g < step; ++g) {
            //---------------- get input and output----------------

            for (size_t i = 0; i < N * 10; ++i)
                net_dst[i] = (float)0;

            // read src and dst data from fasion-mnist
            for (size_t i = 0; i < N; ++i) {
                std::vector<uint8_t> pic = dataset.training_images[train_t];
                size_t ans = dataset.training_labels[train_t];
                ++train_t;

                int fpi = i * 1 * IS;  // first pixel index

                // write data into src while doing normalization (divided by 255)
                for (size_t j = 0; j < IS; ++j)
                    net_src[fpi + j] = ((float)pic[j]) / 255.0;

                // write data into dst
                net_dst[i * 10 + ans] = 1;
            }

            write_to_dnnl_memory(net_src.data(), conv1_src_memory);

            // -----------------------------------------------------

            // forward calculate
            for (size_t i = 0; i < net_fwd.size(); ++i) {
                net_fwd.at(i).execute(s, net_fwd_args.at(i));
                // std::cout << "Forward primitive " << i << " executed!"
                //   << std::endl;
            }

            // calculate the loss function -- cross entropy
            read_from_dnnl_memory(y_hat_clipped.data(), y_hat_clipped_memory);
            read_from_dnnl_memory(y_hat_logged.data(), y_hat_logged_memory);
            float crossEntropy = 0;
            for (size_t j = 0; j < y_hat_logged.size(); ++j) {
                crossEntropy += y_hat_logged[j] * net_dst[j];
                diff_y_hat[j] = -net_dst[j] / ((float)N * y_hat_clipped[j]);
            }
            crossEntropy /= (float)(-N);
            entropy[g] = crossEntropy;
            write_to_dnnl_memory(diff_y_hat.data(), diff_y_hat_memory);

            // backward calculate
            for (size_t i = 0; i < net_bwd.size(); ++i) {
                net_bwd.at(i).execute(s, net_bwd_args.at(i));
                // std::cout << "Backward primitive " << i << " executed!"
                //           << std::endl;
            }

            // finally update weights and bias
            for (size_t i = 0; i < net_sgd.size(); ++i)
                net_sgd.at(i).execute(s, net_sgd_args.at(i));
        }
        float loss_sum = 0;
        for (size_t g = 0; g < step; ++g)
            loss_sum += entropy[g];
        loss[k] = loss_sum / step;
        std::cout << k + 1 << " th training, loss: " << loss[k] << std::endl;
    }
    // for (size_t k = 0; k < epoch; ++k)
    //     std::cout << k << "th training, loss: " << loss[k] << std::endl;
    s.wait();
}

int main(int argc, char** argv) {
    if (dataset.training_images.size() < N * step) {
        std::cout << "there are not so much data!" << std::endl;
        std::cout << dataset.training_images.size() << std::endl;
        return 0;
    }
    return handle_example_errors(LeNet, parse_engine_kind(argc, argv));
}