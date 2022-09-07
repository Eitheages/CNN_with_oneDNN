#ifndef MY_LAYERS
#define MY_LAYERS

#include <bits/stdc++.h>
#include "example_utils.hpp"
#include "oneapi/dnnl/dnnl.hpp"

using namespace dnnl;
using tag = memory::format_tag;
using dt = memory::data_type;

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
    convolution_forward::primitive_desc conv_pd() const { return pd1_m; }
    eltwise_forward::primitive_desc relu_pd() const { return pd2_m; }

    memory conv_dst_memory, weights_memory, bias_memory;  // for backward

private:
    memory dst_m;
    convolution_forward::primitive_desc pd1_m;
    eltwise_forward::primitive_desc pd2_m;
};

class Conv2DwithReLu_back {
    // firstly backward relu, calc diff_relu_src based on diff_dst and relu_src
    // secondly backward conv: calc diff_weights and diff_bias based on diff_conv_dst and src;
    // calc diff_src based on diff_dst and weights
    // remember there are always (diff_relu_src == diff_conv_dst) and (relu_src == conv_dst)
public:
    Conv2DwithReLu_back(engine eng, std::vector<primitive>& net,
                        std::vector<std::unordered_map<int, memory>>& net_args,
                        const memory::dims& weights_tz,
                        const memory::dims& strides,
                        const memory::dims& padding,
                        const memory& diff_dst_memory, const memory& src_memory,
                        const Conv2DwithReLu& conv_fwd,
                        float negative_slope = 0.0f);
    ~Conv2DwithReLu_back() = default;
    Conv2DwithReLu_back(const Conv2DwithReLu_back&) = delete;

    memory diff_src_memory;
    memory diff_weights_memory, diff_bias_memory;
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
    pooling_forward::primitive_desc prim_desc() const { return pd_m; }
    memory workspace_memory;

private:
    bool iftrain;
    memory dst_m;
    pooling_forward::primitive_desc pd_m;
};

class MaxPooling_back {
    // calc diff_src based on diff_dst and workspace
public:
    MaxPooling_back(engine eng, std::vector<primitive>& net,
                    std::vector<std::unordered_map<int, memory>>& net_args,
                    const memory::dims& kernel, const memory::dims& strides,
                    const memory::dims& padding, const memory& diff_dst_memory,
                    const memory& src_memory, const MaxPooling& pool_bwd);
    ~MaxPooling_back() = default;
    MaxPooling_back(const MaxPooling_back&) = delete;

    memory diff_src_memory;
};

class Dense {
public:
    Dense(engine eng, std::vector<primitive>& net,
          std::vector<std::unordered_map<int, memory>>& net_args,
          const memory& src_memory, const memory::dims& src_tz,
          const memory::dims& dst_tz, const memory::dims& weights_tz);
    ~Dense() = default;
    Dense(const Dense& obj) = delete;
    memory dst_memory() const { return dst_m; }
    dnnl::inner_product_forward::primitive_desc prim_desc() const {
        return pd_m;
    }

    memory weights_memory, bias_memory;

private:
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
        pd_m = pd;
    }

    ~ReLU() = default;
    ReLU(const ReLU& obj) = delete;
    memory dst_memory() const { return dst_m; }
    eltwise_forward::primitive_desc prim_desc() const { return pd_m; }

private:
    eltwise_forward::primitive_desc pd_m;
    memory dst_m;
};

class ReLU_back {
    // calc diff_src based on diff_dst and src
public:
    ReLU_back(engine eng, std::vector<primitive>& net,
              std::vector<std::unordered_map<int, memory>>& net_args,
              const memory& diff_dst_memory, const memory& src_memory,
              const ReLU& relu_fwd, float negative_slope = 0.0f);

    memory diff_src_memory;
};

class Dense_back {
    // calc diff_weights and diff_bias based on diff_dst and src
    // calc diff_src based on diff_dst and weights
public:
    Dense_back(engine eng, std::vector<primitive>& net,
               std::vector<std::unordered_map<int, memory>>& net_args,
               const memory& diff_dst_memory, const memory& src_memory,
               const memory::dims& weights_tz, const Dense& dense_fwd);
    ~Dense_back() = default;
    Dense_back(const Dense_back& obj) = delete;
    memory diff_src_memory, diff_weights_memory, diff_bias_memory;
};

Conv2DwithReLu::Conv2DwithReLu(
    dnnl::engine eng, std::vector<primitive>& net,
    std::vector<std::unordered_map<int, memory>>& net_args,
    const memory& src_memory, const memory::dims& src_tz,
    const memory::dims& dst_tz, const memory::dims& weights_tz,
    const memory::dims& strides, const memory::dims& padding,
    const float& negative_slope) {

    std::vector<float> weights(product(weights_tz));
    std::vector<float> bias(weights_tz.at(0));

    std::default_random_engine generator(155);
    std::normal_distribution<float> norm_dist(0.f, 1.f);

    // initializing non-zero values for weights and bias
    for (size_t i = 0; i < weights.size(); ++i)
        weights[i] = norm_dist(generator);
    for (size_t i = 0; i < bias.size(); ++i)
        bias[i] = norm_dist(generator);

    memory::dims bias_tz = {weights_tz[0]};

#ifdef USEREORDER
    auto user_weights_memory = memory({{weights_tz}, dt::f32, tag::oihw}, eng);
    write_to_dnnl_memory(weights.data(), user_weights_memory);
    auto user_bias_memory = memory({{bias_tz}, dt::f32, tag::x}, eng);
    write_to_dnnl_memory(bias.data(), user_bias_memory);
#endif

#ifndef USEREORDER
    weights_memory = memory({{weights_tz}, dt::f32, tag::oihw}, eng);
    write_to_dnnl_memory(weights.data(), weights_memory);
    bias_memory = memory({{bias_tz}, dt::f32, tag::x}, eng);
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
    conv_dst_memory = memory(pd.dst_desc(), eng);

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
    pd1_m = pd;
    pd2_m = relu_pd;
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
        workspace_memory = memory(pd.workspace_desc(), eng);
        net_args.back().insert({DNNL_ARG_WORKSPACE, workspace_memory});
    }

    dst_m = dst_memory;
    pd_m = pd;
}

Dense::Dense(dnnl::engine eng, std::vector<primitive>& net,
             std::vector<std::unordered_map<int, memory>>& net_args,
             const memory& src_memory, const memory::dims& src_tz,
             const memory::dims& dst_tz, const memory::dims& weights_tz) {

    std::vector<float> weights(product(weights_tz));
    std::vector<float> bias(weights_tz.at(0));

    std::default_random_engine generator(155);
    std::normal_distribution<float> norm_dist(0.f, 1.f);

    // initializing non-zero values for weights and bias
    for (size_t i = 0; i < weights.size(); ++i)
        weights[i] = norm_dist(generator);
    for (size_t i = 0; i < bias.size(); ++i)
        bias[i] = norm_dist(generator);

    memory::dims bias_tz = {weights_tz[0]};

    // create memory for user data
    weights_memory = memory(
        {{weights_tz}, dt::f32, (weights_tz.size() == 2 ? tag::oi : tag::oihw)},
        eng);
    write_to_dnnl_memory(weights.data(), weights_memory);
    bias_memory = memory({{bias_tz}, dt::f32, tag::x}, eng);
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

Dense_back::Dense_back(engine eng, std::vector<primitive>& net,
                       std::vector<std::unordered_map<int, memory>>& net_args,
                       const memory& diff_dst_memory, const memory& src_memory,
                       const memory::dims& weights_tz, const Dense& dense_fwd) {

    memory::dims bias_tz = {weights_tz[0]};
    // std::vector<float> diff_fc_weights(product(weights_tz));
    // std::vector<float> diff_fc_bias(product(bias_tz));

    diff_weights_memory = memory(
        {{weights_tz}, dt::f32, (weights_tz.size() == 2 ? tag::oi : tag::oihw)},
        eng);
    diff_bias_memory = memory({{bias_tz}, dt::f32, tag::x}, eng);

    auto src_md = src_memory.get_desc();
    auto diff_dst_md = diff_dst_memory.get_desc();
    auto fwd_pd = dense_fwd.prim_desc();

    auto bwd_weights_desc = inner_product_backward_weights::desc(
        src_md, diff_weights_memory.get_desc(), diff_bias_memory.get_desc(),
        diff_dst_md);
    auto bwd_weights_pd = inner_product_backward_weights::primitive_desc(
        bwd_weights_desc, eng, fwd_pd);

    net.push_back(inner_product_backward_weights(bwd_weights_pd));
    net_args.push_back({{DNNL_ARG_DIFF_DST, diff_dst_memory},
                        {DNNL_ARG_SRC, src_memory},
                        {DNNL_ARG_DIFF_WEIGHTS, diff_weights_memory},
                        {DNNL_ARG_DIFF_BIAS, diff_bias_memory}});

    auto bwd_data_desc = inner_product_backward_data::desc(
        src_md, dense_fwd.weights_memory.get_desc(), diff_dst_md);
    auto bwd_data_pd =
        inner_product_backward_data::primitive_desc(bwd_data_desc, eng, fwd_pd);

    diff_src_memory = memory(src_md, eng);

    net.push_back(inner_product_backward_data(bwd_data_pd));
    net_args.push_back({{DNNL_ARG_DIFF_DST, diff_dst_memory},
                        {DNNL_ARG_WEIGHTS, dense_fwd.weights_memory},
                        {DNNL_ARG_DIFF_SRC, diff_src_memory}});

    return;
}

ReLU_back::ReLU_back(engine eng, std::vector<primitive>& net,
                     std::vector<std::unordered_map<int, memory>>& net_args,
                     const memory& diff_dst_memory, const memory& src_memory,
                     const ReLU& relu_fwd, float negative_slope) {
    auto src_md = src_memory.get_desc();
    diff_src_memory = memory(src_md, eng);

    auto bwd_desc = eltwise_backward::desc(algorithm::eltwise_relu,
                                           diff_src_memory.get_desc(), src_md,
                                           negative_slope);
    auto bwd_pd =
        eltwise_backward::primitive_desc(bwd_desc, eng, relu_fwd.prim_desc());

    net.push_back(eltwise_backward(bwd_pd));
    net_args.push_back({{DNNL_ARG_SRC, src_memory},
                        {DNNL_ARG_DIFF_DST, diff_dst_memory},
                        {DNNL_ARG_DIFF_SRC, diff_src_memory}});
}

MaxPooling_back::MaxPooling_back(
    engine eng, std::vector<primitive>& net,
    std::vector<std::unordered_map<int, memory>>& net_args,
    const memory::dims& kernel, const memory::dims& strides,
    const memory::dims& padding, const memory& diff_dst_memory,
    const memory& src_memory, const MaxPooling& pool_bwd) {
    auto src_md = src_memory.get_desc();
    diff_src_memory = memory(src_md, eng);
    auto bwd_desc = pooling_backward::desc(
        algorithm::pooling_max, diff_src_memory.get_desc(),
        diff_dst_memory.get_desc(), strides, kernel, padding, padding);
    auto bwd_pd =
        pooling_backward::primitive_desc(bwd_desc, eng, pool_bwd.prim_desc());

    net.push_back(pooling_backward(bwd_pd));
    net_args.push_back({{DNNL_ARG_DIFF_DST, diff_dst_memory},
                        {DNNL_ARG_WORKSPACE, pool_bwd.workspace_memory},
                        {DNNL_ARG_DIFF_SRC, diff_src_memory}});
}

Conv2DwithReLu_back::Conv2DwithReLu_back(
    engine eng, std::vector<primitive>& net,
    std::vector<std::unordered_map<int, memory>>& net_args,
    const memory::dims& weights_tz, const memory::dims& strides,
    const memory::dims& padding, const memory& diff_dst_memory,
    const memory& src_memory, const Conv2DwithReLu& conv_fwd,
    float negative_slope) {
    // 1) relu back
    auto relu_src_md = conv_fwd.conv_dst_memory.get_desc();
    auto diff_relu_src_memory = memory(relu_src_md, eng);
    auto diff_relu_src_md = diff_relu_src_memory.get_desc();

    auto relu_bwd_desc = eltwise_backward::desc(algorithm::eltwise_relu,
                                                diff_relu_src_md, relu_src_md);
    auto relu_bwd_pd = eltwise_backward::primitive_desc(
        relu_bwd_desc, eng, conv_fwd.relu_pd(), negative_slope);

    net.push_back(eltwise_backward(relu_bwd_pd));
    net_args.push_back({{DNNL_ARG_SRC, conv_fwd.conv_dst_memory},
                        {DNNL_ARG_DIFF_DST, diff_dst_memory},
                        {DNNL_ARG_DIFF_SRC, diff_relu_src_memory}});

    // 2) convolution back (weights)
    memory::dims bias_tz = {weights_tz[0]};

    diff_weights_memory = memory({{weights_tz}, dt::f32, tag::oihw}, eng);
    diff_bias_memory = memory({{bias_tz}, dt::f32, tag::x}, eng);

    auto weights_md = memory::desc({weights_tz}, dt::f32, tag::any);

    auto src_md = src_memory.get_desc();

    auto conv_weights_bwd_desc = convolution_backward_weights::desc(
        algorithm::convolution_direct, src_md, diff_weights_memory.get_desc(),
        diff_relu_src_md, strides, padding, padding);
    auto conv_weights_bwd_pd = convolution_backward_weights::primitive_desc(
        conv_weights_bwd_desc, eng, conv_fwd.conv_pd());

    net.push_back(convolution_backward_weights(conv_weights_bwd_pd));
    net_args.push_back({{DNNL_ARG_DIFF_DST, diff_relu_src_memory},
                        {DNNL_ARG_SRC, src_memory},
                        {DNNL_ARG_DIFF_WEIGHTS, diff_weights_memory},
                        {DNNL_ARG_DIFF_BIAS, diff_bias_memory}});

    diff_src_memory = memory(src_md, eng);

    auto conv_data_bwd_desc = convolution_backward_data::desc(
        algorithm::convolution_direct, diff_src_memory.get_desc(), weights_md,
        diff_relu_src_md, strides, padding, padding);
    auto conv_data_bwd_pd = convolution_backward_data::primitive_desc(
        conv_data_bwd_desc, eng, conv_fwd.conv_pd());

    net.push_back(convolution_backward_data(conv_data_bwd_pd));
    net_args.push_back({{DNNL_ARG_DIFF_DST, diff_relu_src_memory},
                        {DNNL_ARG_WEIGHTS, conv_fwd.weights_memory},
                        {DNNL_ARG_DIFF_SRC, diff_src_memory}});
}

#endif