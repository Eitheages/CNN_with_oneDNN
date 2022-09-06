#ifndef MY_LAYERS
#define MY_LAYERS

#include "oneapi/dnnl/dnnl.hpp"
#include "example_utils.hpp"
#include <math.h>

using namespace dnnl;

class Conv2DwithReLu {
public:
    Conv2DwithReLu(engine eng, std::vector<primitive>& net,
                   std::vector<std::unordered_map<int, memory>>& net_args,
                   const memory& src_memory, const memory::dims& src_tz,
                   const memory::dims& dst_tz, const memory::dims& weights_tz,
                   const memory::dims& strides, const memory::dims& padding,
                   const float& negative_slope);
    ~Conv2DwithReLu() = default;
    Conv2DwithReLu(const Conv2DwithReLu& obj) =
        delete;  // ban copying to avoid some bugs
    memory dst_memory() const { return dst_m; }

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
    MaxPooling(engine eng, std::vector<primitive>& net,
               std::vector<std::unordered_map<int, memory>>& net_args,
               const memory& src_memory, const memory::dims& kernel,
               const memory::dims& dst_tz, const memory::dims& strides,
               const memory::dims& padding, bool trained = true);
    ~MaxPooling() = default;
    MaxPooling(const MaxPooling& obj) =
        delete;  // ban copying to avoid some bugs
    memory dst_memory() const { return dst_m; }

private:
    bool iftrain;
    memory dst_m;
};

class Dense {
public:
    Dense(engine eng, std::vector<primitive>& net,
          std::vector<std::unordered_map<int, memory>>& net_args,
          const memory& src_memory, const memory::dims& src_tz, const memory::dims& dst_tz,
          const memory::dims& weights_tz);
    ~Dense() = default;
    Dense(const Dense& obj) = delete;
    memory dst_memory() const { return dst_m; }
    dnnl::inner_product_forward::primitive_desc prim_desc() const { return pd_m; }

private:
    std::vector<float> weights;
    std::vector<float> bias;
    memory dst_m;
    dnnl::inner_product_forward::primitive_desc pd_m;
};

class ReLU {
public:
    ReLU(engine eng, std::vector<primitive>& net,
         std::vector<std::unordered_map<int, memory>>& net_args,
         const memory& src_memory, const float& negative_slope) {
        auto desc = dnnl::eltwise_forward::desc(
            dnnl::prop_kind::forward_training, algorithm::eltwise_relu,
            src_memory.get_desc(), negative_slope);
        auto pd = dnnl::eltwise_forward::primitive_desc(desc, eng);

        // create relu dst memory
        auto dst_memory = memory(pd.dst_desc(), eng);

        net.push_back(dnnl::eltwise_forward(pd));
        net_args.push_back(
            {{DNNL_ARG_SRC, src_memory}, {DNNL_ARG_DST, dst_memory}});
        dst_m = dst_memory;
    }

    ~ReLU() = default;
    ReLU(const ReLU& obj) = delete;
    memory dst_memory() const { return dst_m; }

private:
    memory dst_m;
};

class CrossEntropyLoss {
public:
    CrossEntropyLoss(engine eng, std::vector<primitive>& net,
                     std::vector<std::unordered_map<int, memory>>& net_args,
                     const memory& y_hat, const memory& y_true, const memory::dims& y_tz);
    ~CrossEntropyLoss() = default;
    CrossEntropyLoss(const CrossEntropyLoss& obj);

private:
};

class Eltwise {
public:
    memory arg_src, arg_dst;
    /**
     * @brief Construct a new Eltwise object
     *
     * @param activation algorithm objects which defines the element-wise operation
     * @param alpha Alpha parameter (algorithm dependent)
     * @param beta Beta Paremeter (algorithm dependent)
     * @param input dnnl:memory object containing the input
     * @param net This is the vector of primitives to which we will append the FC layer primitive
     * @param net_args This is the associated map to which we will add the arguments of the primitive
     * @param eng oneAPI engine that will host the primitive
     */
    Eltwise(algorithm activation, float alpha, float beta, memory input,
            std::vector<primitive>& net,
            std::vector<std::unordered_map<int, memory>>& net_args, engine eng);

private:
};

class Eltwise_back {
public:
    memory arg_diff_src, arg_src, arg_diff_dst;
    /**
     * @brief Construct a new Eltwise_back object
     *
     * @param activation
     * @param alpha
     * @param beta
     * @param eltwise_fwd
     * @param diff_dst
     * @param net The pipeline onto which the primitive will be appended
     * @param net_args The arguments
     * @param eng The oneAPI engine
     */
    Eltwise_back(engine eng, std::vector<primitive>& net,
                 std::vector<std::unordered_map<int, memory>>& net_args,
                 algorithm activation, Eltwise eltwise_fwd, memory diff_dst,
                 float alpha = 0.f, float beta = 0.f);

private:
};

class Dense_back {
public:
    Dense_back(engine eng, std::vector<primitive>& net,
               std::vector<std::unordered_map<int, memory>>& net_args,
               const memory& diff_dst_memory, const memory& src_memory,
               const memory::dims& weights_tz, const Dense& dense_fwd);
    ~Dense_back() = default;
    Dense_back(const Dense_back& obj) = delete;
};


using tag = memory::format_tag;
using dt = memory::data_type;

Conv2DwithReLu::Conv2DwithReLu(
    dnnl::engine eng, std::vector<primitive>& net,
    std::vector<std::unordered_map<int, memory>>& net_args,
    const memory& src_memory, const memory::dims& src_tz,
    const memory::dims& dst_tz, const memory::dims& weights_tz,
    const memory::dims& strides, const memory::dims& padding,
    const float& negative_slope)
    : weights(product(weights_tz)), bias(weights_tz.at(0)) {
    // initializing non-zero values for weights and bias
    for (size_t i = 0; i < weights.size(); ++i)
        weights[i] = sinf((float)i);
    for (size_t i = 0; i < bias.size(); ++i)
        bias[i] = sinf((float)i);

    memory::dims bias_tz = {weights_tz[0]};

#ifdef USEREORDER
    auto user_weights_memory = memory({{weights_tz}, dt::f32, tag::oihw}, eng);
    write_to_dnnl_memory(weights.data(), user_weights_memory);
    auto user_bias_memory = memory({{bias_tz}, dt::f32, tag::x}, eng);
    write_to_dnnl_memory(bias.data(), user_bias_memory);
#endif

#ifndef USEREORDER
    auto weights_memory = memory({{weights_tz}, dt::f32, tag::oihw}, eng);
    write_to_dnnl_memory(weights.data(), weights_memory);
    auto bias_memory = memory({{bias_tz}, dt::f32, tag::x}, eng);
    write_to_dnnl_memory(bias.data(), bias_memory);
#endif

    auto src_md = memory::desc({src_tz}, dt::f32, tag::any);
    auto bias_md = memory::desc({bias_tz}, dt::f32, tag::any);
    auto weights_md = memory::desc({weights_tz}, dt::f32, tag::any);
    auto dst_md = memory::desc({dst_tz}, dt::f32, tag::any);

    auto desc = convolution_forward::desc(
        prop_kind::forward, algorithm::convolution_direct, src_md, weights_md,
        bias_md, dst_md, strides, padding, padding);
    auto pd = convolution_forward::primitive_desc(desc, eng);

#ifdef USEREORDER
    // create reorder primitives between user input and conv src if needed
    auto weights_memory = user_weights_memory;
    if (pd.weights_desc() != user_weights_memory.get_desc()) {
        weights_memory = memory(pd.weights_desc(), eng);
        net.push_back(reorder(user_weights_memory, weights_memory));
        net_args.push_back({{DNNL_ARG_FROM, user_weights_memory},
                            {DNNL_ARG_TO, weights_memory}});
    }

    // added by rbj (159 modified as well)
    auto bias_memory = user_bias_memory;
    if (pd.bias_desc() != user_bias_memory.get_desc()) {
        bias_memory = memory(pd.bias_desc(), eng);
        net.push_back(reorder(user_bias_memory, bias_memory));
        net_args.push_back(
            {{DNNL_ARG_FROM, user_bias_memory}, {DNNL_ARG_TO, bias_memory}});
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
    auto relu_desc = eltwise_forward::desc(
        prop_kind::forward_training, algorithm::eltwise_relu,
        conv_dst_memory.get_desc(), negative_slope);
    auto relu_pd = eltwise_forward::primitive_desc(relu_desc, eng);

    // create relu dst memory
    auto relu_dst_memory = memory(relu_pd.dst_desc(), eng);

    net.push_back(eltwise_forward(relu_pd));
    net_args.push_back(
        {{DNNL_ARG_SRC, conv_dst_memory}, {DNNL_ARG_DST, relu_dst_memory}});
    dst_m = relu_dst_memory;
}

MaxPooling::MaxPooling(dnnl::engine eng, std::vector<primitive>& net,
                       std::vector<std::unordered_map<int, memory>>& net_args,
                       const memory& src_memory, const memory::dims& kernel,
                       const memory::dims& dst_tz, const memory::dims& strides,
                       const memory::dims& padding, bool trained)
    : iftrain(trained) {
    auto dst_md = memory::desc({dst_tz}, dt::f32, tag::any);

    //[Create pooling primitive]
    auto desc = pooling_forward::desc(
        prop_kind::forward_training, algorithm::pooling_max,
        src_memory.get_desc(), dst_md, strides, kernel, padding, padding);
    auto pd = pooling_forward::primitive_desc(desc, eng);
    auto dst_memory = memory(pd.dst_desc(), eng);
    //[Create pooling primitive]

    net.push_back(pooling_forward(pd));
    net_args.push_back(
        {{DNNL_ARG_SRC, src_memory}, {DNNL_ARG_DST, dst_memory}});
    // {DNNL_ARG_WORKSPACE, workspace_memory}

    // create pooling workspace memory if training
    if (trained) {
        auto workspace_memory = memory(pd.workspace_desc(), eng);
        net_args.back().insert({DNNL_ARG_WORKSPACE, workspace_memory});
    }

    dst_m = dst_memory;
}

Dense::Dense(dnnl::engine eng, std::vector<primitive>& net,
             std::vector<std::unordered_map<int, memory>>& net_args,
             const memory& src_memory, const memory::dims& src_tz,
             const memory::dims& dst_tz, const memory::dims& weights_tz)
    : weights(product(weights_tz)), bias(weights_tz.at(0)) {
    // initializing non-zero values for weights and bias
    for (size_t i = 0; i < weights.size(); ++i)
        weights[i] = sinf((float)i);
    for (size_t i = 0; i < bias.size(); ++i)
        bias[i] = sinf((float)i);

    memory::dims bias_tz = {weights_tz[0]};

    // create memory for user data
    auto weights_memory = memory(
        {{weights_tz}, dt::f32, (weights_tz.size() == 2 ? tag::oi : tag::oihw)},
        eng);
    write_to_dnnl_memory(weights.data(), weights_memory);
    auto bias_memory = memory({{bias_tz}, dt::f32, tag::x}, eng);
    write_to_dnnl_memory(bias.data(), bias_memory);

    // create memory descriptors for convolution data w/ no specified format
    auto src_md = memory::desc({src_tz}, dt::f32, tag::any);
    auto bias_md = memory::desc({bias_tz}, dt::f32, tag::any);
    auto weights_md = memory::desc({weights_tz}, dt::f32, tag::any);
    auto dst_md = memory::desc({dst_tz}, dt::f32, tag::any);

    // create a inner_product
    auto desc = inner_product_forward::desc(prop_kind::forward_training, src_md,
                                            weights_md, bias_md, dst_md);
    auto pd = inner_product_forward::primitive_desc(desc, eng);

    auto dst_memory = memory(pd.dst_desc(), eng);

    // create convolution primitive and add it to net
    net.push_back(inner_product_forward(pd));
    net_args.push_back({{DNNL_ARG_SRC, src_memory},
                        {DNNL_ARG_WEIGHTS, weights_memory},
                        {DNNL_ARG_BIAS, bias_memory},
                        {DNNL_ARG_DST, dst_memory}});

    dst_m = dst_memory;
    pd_m = pd;
}

CrossEntropyLoss::CrossEntropyLoss(
    dnnl::engine eng, std::vector<primitive>& net,
    std::vector<std::unordered_map<int, memory>>& net_args,
    const memory& y_hat_memory, const memory& y_true_memory,
    const memory::dims& y_tz) {

    // 0) Clip y_hat to avoid performing log(0)

    float lower = 1e-7;      // alpha
    float upper = 1 - 1e-7;  // beta

    auto y_md = memory::desc({y_tz}, dt::f32, tag::nc);
    auto y_hat_cliped_memory = memory(y_md, eng);

    auto clip_desc =
        eltwise_forward::desc(prop_kind::forward_training,
                              algorithm::eltwise_clip, y_md, lower, upper);
    auto clip_pd = eltwise_forward::primitive_desc(clip_desc, eng);

    net.push_back(eltwise_forward(clip_pd));
    net_args.push_back(
        {{DNNL_ARG_SRC, y_hat_memory}, {DNNL_ARG_DST, y_hat_cliped_memory}});

    // 1) Perform elementwise log on y_hat_cliped
    auto y_hat_logged_memory = memory(y_md, eng);

    auto log_desc = eltwise_forward::desc(prop_kind::forward_training,
                                          algorithm::eltwise_log, y_md);
    auto log_pd = eltwise_forward::primitive_desc(log_desc, eng);

    net.push_back(eltwise_forward(log_pd));
    net_args.push_back({{DNNL_ARG_SRC, y_hat_cliped_memory},
                        {DNNL_ARG_DST, y_hat_logged_memory}});

    return;
}

Eltwise::Eltwise(algorithm activation, float alpha, float beta,
                 memory input, std::vector<primitive>& net,
                 std::vector<std::unordered_map<int, memory>>& net_args,
                 engine eng) {

    auto src_md = input.get_desc();

    auto dst_mem = memory(src_md, eng);
    auto dst_md = dst_mem.get_desc();

    std::cout << "Memory allocated\n";

    auto eltwise_desc = dnnl::eltwise_forward::desc(
        dnnl::prop_kind::forward_training, activation, dst_md, alpha, beta);
    auto eltwise_pd = dnnl::eltwise_forward::primitive_desc(eltwise_desc, eng);

    arg_src = input;
    arg_dst = dst_mem;

    net.push_back(dnnl::eltwise_forward(eltwise_pd));
    net_args.push_back({{DNNL_ARG_SRC, input}, {DNNL_ARG_DST, dst_mem}});
}

Eltwise_back::Eltwise_back(
    dnnl::engine eng, std::vector<primitive>& net,
    std::vector<std::unordered_map<int, memory>>& net_args,
    algorithm activation, Eltwise eltwise_fwd, memory diff_dst,
    float alpha, float beta) {

    auto diff_dst_md = diff_dst.get_desc();
    //auto diff_src_md = memory::desc(diff_dst_md.dims(), dt::f32, tag::any);

    auto diff_src_md = diff_dst_md;

    auto diff_src_mem = memory(diff_src_md, eng);

    auto src_mem = eltwise_fwd.arg_src;
    auto src_md = src_mem.get_desc();

    // Recreate forward descriptor for hint
    auto eltwise_fwd_desc = dnnl::eltwise_forward::desc(
        dnnl::prop_kind::forward_training, activation,
        eltwise_fwd.arg_dst.get_desc(), alpha, beta);
    auto eltwise_fwd_pd =
        dnnl::eltwise_forward::primitive_desc(eltwise_fwd_desc, eng);

    // We use diff_dst_md as diff_data_md because it is an input and the cnn_trainin_f32.cpp examples
    // does the same thing, however there is no clear explanation in the documentation...
    // https://oneapi-src.github.io/oneDNN/structdnnl_1_1eltwise__backward_1_1desc.html

    auto eltwise_bwd_desc = dnnl::eltwise_backward::desc(
        activation, diff_dst_md, src_md, alpha, beta);

    auto eltwise_bwd_pd = dnnl::eltwise_backward::primitive_desc(
        eltwise_bwd_desc, eng, eltwise_fwd_pd);

    arg_diff_dst = diff_dst;
    arg_src = src_mem;
    arg_diff_src = diff_src_mem;

    net.push_back(dnnl::eltwise_backward(eltwise_bwd_pd));
    net_args.push_back({{DNNL_ARG_DIFF_DST, diff_dst},
                        {DNNL_ARG_SRC, src_mem},
                        {DNNL_ARG_DIFF_SRC, diff_src_mem}});
}

Dense_back::Dense_back(engine eng, std::vector<primitive>& net,
                       std::vector<std::unordered_map<int, memory>>& net_args,
                       const memory& diff_dst_memory, const memory& src_memory,
                       const memory::dims& weights_tz, const Dense& dense_fwd) {

    memory::dims bias_tz = {weights_tz[0]};
    std::vector<float> diff_fc_weights(product(weights_tz));
    std::vector<float> diff_fc_bias(product(bias_tz));

    auto diff_weights_memory = memory(
        {{weights_tz}, dt::f32, (weights_tz.size() == 2 ? tag::oi : tag::oihw)},
        eng);
    auto diff_bias_memory = memory({{bias_tz}, dt::f32, tag::x}, eng);

    auto desc = inner_product_backward_weights::desc(
        src_memory.get_desc(), diff_weights_memory.get_desc(),
        diff_bias_memory.get_desc(), diff_dst_memory.get_desc());
    auto pd = inner_product_backward_weights::primitive_desc(
        desc, eng, dense_fwd.prim_desc());

    net.push_back(inner_product_backward_weights(pd));
    net_args.push_back({{DNNL_ARG_DIFF_DST, diff_dst_memory},
                        {DNNL_ARG_SRC, src_memory},
                        {DNNL_ARG_WEIGHTS, diff_weights_memory},
                        {DNNL_ARG_BIAS, diff_bias_memory}});

    return;
}

#endif