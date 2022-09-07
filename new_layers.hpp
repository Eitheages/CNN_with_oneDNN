#ifndef NEW_L
#define NEW_L
// #define USEREORDER

#include <bits/stdc++.h>
#include "example_utils.hpp"
#include "oneapi/dnnl/dnnl.hpp"

using namespace dnnl;
using tag = memory::format_tag;
using dt = memory::data_type;

void print_vector(std::vector<dnnl::memory::dim> const& input) {
    for (int i = 0; i < input.size(); i++) {
        std::cout << input.at(i) << ' ';
    }
    std::cout << "\n";
}

dnnl::memory checkType(
    dnnl::memory::desc md_true_type, dnnl::memory mem_to_check,
    std::vector<dnnl::primitive>& net,
    std::vector<std::unordered_map<int, dnnl::memory>>& net_args,
    dnnl::engine eng) {
    auto mem_reordered = mem_to_check;
    if (md_true_type != mem_to_check.get_desc()) {
        std::cout << "Memory mismatch adding reorder primitive\n";
        auto mem_reordered = dnnl::memory(md_true_type, eng);
        net.push_back(dnnl::reorder(mem_to_check, mem_reordered));
        net_args.push_back(
            {{DNNL_ARG_FROM, mem_to_check}, {DNNL_ARG_TO, mem_reordered}});
    }
    return mem_reordered;
}

class Conv2D {
public:
    dnnl::memory arg_src;      //!< Source memory handler
    dnnl::memory arg_dst;      //!< Destination memory handler
    dnnl::memory arg_bias;     //!< Bias memory handler
    dnnl::memory arg_weights;  //!< Weights memory handler
    /**
         * @brief Construct a new Conv 2 D object
         *
         * @param batch_size Size of the batch
         * @param patch_length Length of the H and W
         * @param n_kernels Number of kernels
         * @param kernel_size Size of the kernel
         * @param stride_length Stride
         * @param padding_length Padding
         * @param dilation Dilation coefficient for the dilated convolution (0 for no dilation as per oneAPI specs)
         * @param input Input memory
         * @param net This is the vector of primitives to which we will append the FC layer primitive
         * @param net_args This is the associated map to which we will add the arguments of the primitive
         * @param eng oneAPI engine that will host the primitive
         */
    Conv2D(int batch_size, int patch_length, int n_kernels, int kernel_size,
           int stride_length, int padding_length, int dilation,
           dnnl::memory input, std::vector<dnnl::primitive>& net,
           std::vector<std::unordered_map<int, dnnl::memory>>& net_args,
           dnnl::engine eng);

private:
};

class MaxPool2D {
public:
    dnnl::memory arg_src, arg_dst, arg_workspace;
    dnnl::pooling_forward::primitive_desc* pooling_fwd_pd;
    /**
         * @brief Construct a new Max Pool 2 D object
         *
         * @param kernel_size the size of the kernel
         * @param stride_length the length of the stride
         * @param input Input memory
         * @param net This is the vector of primitives to which we will append the FC layer primitive
         * @param net_args This is the associated map to which we will add the arguments of the primitive
         * @param eng oneAPI engine that will host the primitive
         */
    MaxPool2D(int kernel_size, int stride_length, dnnl::memory input,
              std::vector<dnnl::primitive>& net,
              std::vector<std::unordered_map<int, dnnl::memory>>& net_args,
              dnnl::engine eng);

private:
};

class Eltwise {
public:
    dnnl::memory arg_src, arg_dst;
    /**
         * @brief Construct a new Eltwise object
         *
         * @param activation dnnl::algorithm objects which defines the element-wise operation
         * @param alpha Alpha parameter (algorithm dependent)
         * @param beta Beta Paremeter (algorithm dependent)
         * @param input dnnl:memory object containing the input
         * @param net This is the vector of primitives to which we will append the FC layer primitive
         * @param net_args This is the associated map to which we will add the arguments of the primitive
         * @param eng oneAPI engine that will host the primitive
         */
    Eltwise(dnnl::algorithm activation, float alpha, float beta,
            dnnl::memory input, std::vector<dnnl::primitive>& net,
            std::vector<std::unordered_map<int, dnnl::memory>>& net_args,
            dnnl::engine eng);

private:
};

class Dense {
public:
    dnnl::memory arg_src;      //!< Source memory handler
    dnnl::memory arg_dst;      //!< Destination memory handler
    dnnl::memory arg_bias;     //!< Bias memory handler
    dnnl::memory arg_weights;  //!< Weights memory handler
    /**
         * @brief Construct a new Dense object
         *
         * @param fc_output_size this is the number of units inside the FC layer
         * @param input Input memory
         * @param net This is the vector of primitives to which we will append the FC layer primitive
         * @param net_args This is the associated map to which we will add the arguments of the primitive
         * @param eng oneAPI engine that will host the primitive
         */
    Dense(int fc_output_size, dnnl::memory input,
          std::vector<dnnl::primitive>& net,
          std::vector<std::unordered_map<int, dnnl::memory>>& net_args,
          dnnl::engine eng);

private:
};

class Eltwise_back {
public:
    dnnl::memory arg_diff_src, arg_src, arg_diff_dst;
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
    Eltwise_back(dnnl::algorithm activation, float alpha, float beta,
                 Eltwise eltwise_fwd, dnnl::memory diff_dst,
                 std::vector<dnnl::primitive>& net,
                 std::vector<std::unordered_map<int, dnnl::memory>>& net_args,
                 dnnl::engine eng);

private:
};

class Dense_back_data {
public:
    dnnl::memory arg_diff_src, arg_diff_dst;
    dnnl::memory arg_weights;
    /**
         * @brief Construct a new Dense_back_data object
         *
         * @param diff_dst The dnnl::memory object containing the gradient of the loss with respect to the output
         * @param dense_fwd The Dense forward layer
         * @param net The pipeline onto which the primitive will be appended
         * @param net_args The arguments
         * @param eng The oneAPI engine
         */
    Dense_back_data(
        dnnl::memory diff_dst, Dense dense_fwd,
        std::vector<dnnl::primitive>& net,
        std::vector<std::unordered_map<int, dnnl::memory>>& net_args,
        dnnl::engine eng);

private:
};

class Conv2D_back_data {
public:
    dnnl::memory
        arg_diff_src;  //<! Gradient of the loss with respect to the input
    dnnl::memory
        arg_diff_dst;  //<! Gradient of the loss with respect to the output
    dnnl::memory arg_weights;  //<! Weights of the convolution primitive
    /**
         * @brief Construct a new Conv2D_back_data object
         *
         * @param diff_dst Gradient of the loss with respect to the output (ie. the gradient coming from the previous layer)
         * @param conv2d_fwd The class containing the forward primitive
         * @param stride_length The stride
         * @param padding_length The padding
         * @param dilation The dilation
         * @param net The pipeline onto which the primitive will be appended
         * @param net_args The arguments
         * @param eng The oneAPI engine
         */
    Conv2D_back_data(
        dnnl::memory diff_dst, Conv2D conv2d_fwd, int stride_length,
        int padding_length, int dilation, std::vector<dnnl::primitive>& net,
        std::vector<std::unordered_map<int, dnnl::memory>>& net_args,
        dnnl::engine eng);

private:
};

class MaxPool2D_back {
public:
    dnnl::memory arg_diff_src, arg_diff_dst;
    /**
         * @brief Construct a new MaxPool2D_back object
         *
         * @param kernel_size the size of the kernel
         * @param stride_length the stride length
         * @param maxpool_fwd the MaxPool2D forward class
         * @param diff_dst_mem The dnnl::memory object containing the gradient of the loss with respect to the output
         * @param net The pipeline onto which the primitive will be appended
         * @param net_args The arguments
         * @param eng The oneAPI engine
         */
    MaxPool2D_back(int kernel_size, int stride_length, MaxPool2D maxpool_fwd,
                   dnnl::memory diff_dst_mem, std::vector<dnnl::primitive>& net,
                   std::vector<std::unordered_map<int, dnnl::memory>>& net_args,
                   dnnl::engine eng);

private:
};

class Conv2D_back_weights {
public:
    dnnl::memory arg_src, arg_diff_dst;
    dnnl::memory arg_diff_weights, arg_diff_bias;
    /**
         * @brief Construct a new Conv2D_back_weights object
         *
         * @param diff_dst Gradient of loss with respect to the output
         * @param conv2d_fwd Forward Conv2D object
         * @param stride_length Stride
         * @param padding_length Padding
         * @param dilation Dilation coefficient
         * @param net This is the vector of primitives to which we will append the FC layer primitive
         * @param net_args This is the associated map to which we will add the arguments of the primitive
         * @param eng oneAPI engine that will host the primitive
         */
    Conv2D_back_weights(
        dnnl::memory diff_dst, Conv2D conv2d_fwd, int stride_length,
        int padding_length, int dilation, std::vector<dnnl::primitive>& net,
        std::vector<std::unordered_map<int, dnnl::memory>>& net_args,
        dnnl::engine eng);

private:
};

/**
 * @brief Primitive which provides backward weights pass for the Dense
 *
 */
class Dense_back_weights {
public:
    dnnl::memory arg_src, arg_diff_dst;
    dnnl::memory arg_diff_weights, arg_diff_bias;
    /**
         * @brief Construct a new Dense_back_weights object
         *
         * @param diff_dst Gradient of loss with respect to the output
         * @param dense_fwd Forward Dense object
         * @param net This is the vector of primitives to which we will append the FC layer primitive
         * @param net_args This is the associated map to which we will add the arguments of the primitive
         * @param eng oneAPI engine that will host the primitive
         */
    Dense_back_weights(
        dnnl::memory diff_dst, Dense dense_fwd,
        std::vector<dnnl::primitive>& net,
        std::vector<std::unordered_map<int, dnnl::memory>>& net_args,
        dnnl::engine eng);

private:
};

// Conv2D, only Glorot initializer implemented
Conv2D::Conv2D(int batch_size, int patch_length, int n_kernels, int kernel_size,
               int stride_length, int padding_length, int dilation,
               dnnl::memory input, std::vector<dnnl::primitive>& net,
               std::vector<std::unordered_map<int, dnnl::memory>>& net_args,
               dnnl::engine eng) {
    // RNG for ALL purposes
    std::default_random_engine generator(155);
    std::normal_distribution<float> norm_dist(0.f, 1.f);

    std::cout << "Creating convolutional layer!\n";
    // N channels = one since we have monochromatic images (WIP)
    //dnnl::memory::dims conv_src_tz = {batch_size, 1, patch_length, patch_length};
    dnnl::memory::dims conv_src_tz = input.get_desc().dims();
    // Get number of "channels" (concept of channel changes after input) from the input
    dnnl::memory::dims conv_weights_tz = {n_kernels, conv_src_tz[1],
                                          kernel_size, kernel_size};
    dnnl::memory::dims conv_bias_tz = {n_kernels};
    dnnl::memory::dims conv_dst_tz = {batch_size, n_kernels, patch_length,
                                      patch_length};
    dnnl::memory::dims conv_strides = {stride_length, stride_length};
    dnnl::memory::dims conv_dilates = {dilation, dilation};
    dnnl::memory::dims conv_padding = {padding_length, padding_length};

    std::vector<float> conv_weights(product(conv_weights_tz));
    std::vector<float> conv_bias(product(conv_bias_tz));

    // Initialize weight and biases
    for (int i = 0; i < conv_weights.size(); i++) {
        conv_weights[i] = norm_dist(generator);
    }

    for (int i = 0; i < conv_bias.size(); i++) {
        conv_bias[i] = norm_dist(generator);
    }

    // Write initialized weights and biases on selected engine
    auto conv_user_weights_memory =
        dnnl::memory({{conv_weights_tz}, dt::f32, tag::oihw}, eng);
    auto conv_user_bias_memory =
        dnnl::memory({{conv_bias_tz}, dt::f32, tag::x}, eng);

    write_to_dnnl_memory(conv_weights.data(), conv_user_weights_memory);
    write_to_dnnl_memory(conv_bias.data(), conv_user_bias_memory);

    // Create memory descriptor for input, weights, biases, destination
    auto conv_src_md = dnnl::memory::desc({conv_src_tz}, dt::f32, tag::any);
    auto conv_weights_md =
        dnnl::memory::desc({conv_weights_tz}, dt::f32, tag::any);
    auto conv_bias_md = dnnl::memory::desc({conv_bias_tz}, dt::f32, tag::any);
    auto conv_dst_md = dnnl::memory::desc({conv_dst_tz}, dt::f32, tag::any);

    // create a (dilated) convolution primitive descriptor
    // The method is overloaded, hence by simply having the correct number of parameters
    // we are choosing a dilated convolution

    std::cout << "Creating primitive descriptor (ma quello desc non quello "
                 "dopo) for convolution\n";

    std::cout << "SRC dims size: " << conv_src_md.dims().size() << "\n";
    std::cout << "Source vector md content: "
              << "\n";
    print_vector(conv_src_md.dims());
    std::cout << "Weights dims size: " << conv_weights_md.dims().size() << "\n";
    std::cout << "Weights vector md content: "
              << "\n";
    print_vector(conv_weights_md.dims());
    std::cout << "Dst dims size: " << conv_dst_md.dims().size() << "\n";
    std::cout << "Dst vector md content: "
              << "\n";
    print_vector(conv_dst_md.dims());
    std::cout << "Bias dims size: " << conv_bias_md.dims().size() << "\n";
    std::cout << "Bias vector md content: "
              << "\n";
    print_vector(conv_bias_md.dims());

    auto conv_desc = dnnl::convolution_forward::desc(
        dnnl::prop_kind::forward, dnnl::algorithm::convolution_direct,
        conv_src_md, conv_weights_md, conv_bias_md, conv_dst_md, conv_strides,
        conv_padding, conv_padding);

    std::cout << "Ho creato il primitive descriptor (ma quello desc non quello "
                 "dopo) for convolution\n";

    // check if f32 convolution is supported on selected engine
    try {
        dnnl::convolution_forward::primitive_desc(conv_desc, eng);
    } catch (dnnl::error& e) {
        if (e.status == dnnl_unimplemented)
            throw example_allows_unimplemented{
                "No f32 convolution implementation is available for this "
                "platform.\n"
                "Please refer to the developer guide for details."};

        // on any other error just re-throw
        throw;
    }

    std::cout << "Creating primitive descriptor for convolution\n";
    auto conv_pd = dnnl::convolution_forward::primitive_desc(conv_desc, eng);

    // Check if the types are proper
    std::cout << "Testing types\n";
    auto conv_weights_memory = checkType(
        conv_pd.weights_desc(), conv_user_weights_memory, net, net_args, eng);
    std::cout << "Weights check OK!\n";
    auto conv_bias_memory = checkType(
        conv_pd.bias_desc(), conv_user_bias_memory, net, net_args, eng);
    std::cout << "Bias check ok!\n";
    auto conv_src_memory =
        checkType(conv_pd.src_desc(), input, net, net_args, eng);
    //auto conv_src_memory = input;
    std::cout << "Source check OK!\n";
    std::cout << "Types tested!\n";

    // Create memory for output (no check needed)
    auto conv_dst_memory = dnnl::memory(conv_pd.dst_desc(), eng);

    arg_src = conv_src_memory;
    arg_weights = conv_weights_memory;
    arg_bias = conv_bias_memory;
    arg_dst = conv_dst_memory;

    // Append primitive to network vector
    net.push_back(dnnl::convolution_forward(conv_pd));
    net_args.push_back({{DNNL_ARG_SRC, conv_src_memory},
                        {DNNL_ARG_WEIGHTS, conv_weights_memory},
                        {DNNL_ARG_BIAS, conv_bias_memory},
                        {DNNL_ARG_DST, conv_dst_memory}});
    std::cout << "Convolutional layer created, new net args size is: "
              << net_args.size() << "\n";
    // Return index to locate the layer
}

MaxPool2D::MaxPool2D(
    int kernel_size, int stride_length, dnnl::memory input,
    std::vector<dnnl::primitive>& net,
    std::vector<std::unordered_map<int, dnnl::memory>>& net_args,
    dnnl::engine eng) {
    auto src_md = input.get_desc();

    long batch = src_md.dims()[0];
    long channels = src_md.dims()[1];
    long input_height = src_md.dims()[2];
    long input_width = src_md.dims()[3];
    long padding = 0;

    const dnnl::memory::dim output_height =
        (input_height - kernel_size + padding + padding) / stride_length + 1;
    const dnnl::memory::dim output_width =
        (input_height - kernel_size + padding + padding) / stride_length + 1;

    // Source (src) and destination (dst) tensors dimensions.
    dnnl::memory::dims src_dims = {batch, channels, input_height, input_width};
    dnnl::memory::dims dst_dims = {batch, channels, output_height,
                                   output_width};

    // Kernel dimensions.
    dnnl::memory::dims kernel_dims = {kernel_size, kernel_size};
    // Strides, padding dimensions.
    dnnl::memory::dims strides_dims = {stride_length, stride_length};
    dnnl::memory::dims padding_dims_l = {padding, padding};
    dnnl::memory::dims padding_dims_r = {padding, padding};

    auto dst_md = dnnl::memory::desc(dst_dims, dt::f32, tag::nchw);
    auto dst_mem = dnnl::memory(dst_md, eng);
    std::cout << "Allocated DST MEM\n";

    // Create descriptor.
    auto pooling_desc = dnnl::pooling_forward::desc(
        dnnl::prop_kind::forward_training, dnnl::algorithm::pooling_max, src_md,
        dst_md, strides_dims, kernel_dims, padding_dims_l, padding_dims_r);
    auto pooling_pd = dnnl::pooling_forward::primitive_desc(pooling_desc, eng);
    std::cout << "Allocated primitive\n";

    auto workspace_mem = dnnl::memory(pooling_pd.workspace_desc(), eng);
    std::cout << "Workspace allocated\n";

    arg_src = input;
    arg_dst = dst_mem;
    arg_workspace = workspace_mem;
    pooling_fwd_pd = &pooling_pd;

    net.push_back(dnnl::pooling_forward(pooling_pd));
    net_args.push_back({{DNNL_ARG_SRC, input},
                        {DNNL_ARG_DST, dst_mem},
                        {DNNL_ARG_WORKSPACE, workspace_mem}});
}

Dense::Dense(int fc_output_size, dnnl::memory input,
             std::vector<dnnl::primitive>& net,
             std::vector<std::unordered_map<int, dnnl::memory>>& net_args,
             dnnl::engine eng) {

    // RNG for ALL purposes
    std::default_random_engine generator;
    generator.seed(155);
    std::normal_distribution<float> norm_dist(0.f, 1.f);

    // 0,1,2,3 are used to grab the dimension we need from the source dims vector

    dnnl::memory::dims weights_dims_fc;
    dnnl::memory::dims bias_dims_fc = {fc_output_size};
    dnnl::memory::dims dst_dims_fc;
    dnnl::memory::desc src_md_fc = input.get_desc();
    dnnl::memory::dims src_dims_fc = src_md_fc.dims();

    // check if we need to flatten ie. use the proper tag according to vector size
    // TODO. check what happens with dimension 3 ;)
    bool from_conv = (src_dims_fc.size() > 3);

    std::cout << "Time to create some memory descriptors!\n";

    if (from_conv) {
        weights_dims_fc = {fc_output_size, src_dims_fc[1], src_dims_fc[2],
                           src_dims_fc[3]};
    } else {
        weights_dims_fc = {fc_output_size, src_dims_fc[1]};
    }

    dst_dims_fc = {src_dims_fc[0], fc_output_size};

    std::cout << "Source MD OK!\n";

    auto bias_md_fc = dnnl::memory::desc(bias_dims_fc, dt::f32, tag::a);
    std::cout << "Bias MD OK!\n";
    auto dst_md_fc = dnnl::memory::desc(dst_dims_fc, dt::f32, tag::nc);
    std::cout << "DST MD OK!\n";
    dnnl::memory::desc weights_md_fc;
    if (from_conv) {
        std::cout << "Set tag from_conv: \n";
        weights_md_fc = dnnl::memory::desc(weights_dims_fc, dt::f32, tag::oihw);
    } else {
        weights_md_fc = dnnl::memory::desc(weights_dims_fc, dt::f32, tag::oi);
    }
    std::cout << "Weights MD OK!\n";
    std::cout << "time to allocate some memory!\n";
    auto src_mem_fc = dnnl::memory(src_md_fc, eng);
    std::cout << "Source allocated!\n";
    auto bias_mem_fc = dnnl::memory(bias_md_fc, eng);
    std::cout << "Weights allocated!\n";
    auto weights_mem_fc = dnnl::memory(weights_md_fc, eng);
    std::cout << "Bias allocated!\n";
    auto dst_mem_fc = dnnl::memory(dst_md_fc, eng);
    std::cout << "Destination allocated!\n";

    // No initialization, will be done in post-op routine
    std::vector<float> fc_weights(product(weights_dims_fc));
    std::vector<float> fc_bias(product(bias_dims_fc));

    std::cout << "Initializing weights: \n";
    for (int i = 0; i < fc_weights.size(); i++) {
        fc_weights[i] = norm_dist(generator);
        //std::cout << fc_weights[i] << " ";
    }
    std::cout << "\n";

    std::cout << "Initializing biases: \n";
    for (int i = 0; i < fc_bias.size(); i++) {
        fc_bias[i] = norm_dist(generator);
        //std::cout << fc_bias[i] << " ";
    }
    std::cout << "\n";

    std::cout << "Write bias to memory: \n";
    write_to_dnnl_memory(fc_bias.data(), bias_mem_fc);
    std::cout << "Write weights to memory: \n";
    write_to_dnnl_memory(fc_weights.data(), weights_mem_fc);

    std::cout << "Dimensions:\n";
    for (int i = 0; i < src_md_fc.dims().size(); i++)
        std::cout << src_md_fc.dims()[i] << " ";
    std::cout << "\n";
    for (int i = 0; i < weights_md_fc.dims().size(); i++)
        std::cout << weights_md_fc.dims()[i] << " ";
    std::cout << "\n";
    for (int i = 0; i < bias_md_fc.dims().size(); i++)
        std::cout << bias_md_fc.dims()[i] << " ";
    std::cout << "\n";
    for (int i = 0; i < dst_md_fc.dims().size(); i++)
        std::cout << dst_md_fc.dims()[i] << " ";
    std::cout << "\n";

    auto fc_desc = dnnl::inner_product_forward::desc(
        dnnl::prop_kind::forward_training, src_md_fc, weights_md_fc, bias_md_fc,
        dst_md_fc);

    auto fc_pd = dnnl::inner_product_forward::primitive_desc(fc_desc, eng);

    // Check if the types are proper
    std::cout << "Start type checking: \n";
    std::cout << "Check source type\n";
    src_mem_fc = checkType(fc_pd.src_desc(), input, net, net_args, eng);
    std::cout << "Check weights type\n";
    weights_mem_fc =
        checkType(fc_pd.weights_desc(), weights_mem_fc, net, net_args, eng);
    std::cout << "Check bias type\n";
    bias_mem_fc = checkType(fc_pd.bias_desc(), bias_mem_fc, net, net_args, eng);

    // Set dnnl::memory pointers inside class
    arg_src = src_mem_fc;
    arg_dst = dst_mem_fc;
    arg_weights = weights_mem_fc;
    arg_bias = bias_mem_fc;

    // Append primitive to network vector
    net.push_back(dnnl::inner_product_forward(fc_pd));
    net_args.push_back({{DNNL_ARG_SRC, src_mem_fc},
                        {DNNL_ARG_WEIGHTS, weights_mem_fc},
                        {DNNL_ARG_BIAS, bias_mem_fc},
                        {DNNL_ARG_DST, dst_mem_fc}});
}

Conv2D_back_data::Conv2D_back_data(
    dnnl::memory diff_dst, Conv2D conv2d_fwd, int stride_length,
    int padding_length, int dilation, std::vector<dnnl::primitive>& net,
    std::vector<std::unordered_map<int, dnnl::memory>>& net_args,
    dnnl::engine eng) {

    auto conv_diff_src_md = conv2d_fwd.arg_src.get_desc();
    auto conv_diff_src_memory = dnnl::memory(conv_diff_src_md, eng);

    std::cout << "Allocating memory for backward convolution\n";
    // Create memory area for backward pass (get types from conv2d_fwd)
    auto conv_weights = conv2d_fwd.arg_weights;
    auto conv_weights_md = conv_weights.get_desc();

    auto conv_bias_md = conv2d_fwd.arg_bias.get_desc();

    std::cout << "Obtaining memory descriptors for backward convolution\n";
    // create memory descriptors for f32 convolution data
    auto conv_bwd_src_md = conv2d_fwd.arg_src.get_desc();
    // Get dst descriptor to recreate forward primitive
    auto conv_fwd_dst_md = conv2d_fwd.arg_dst.get_desc();

    auto conv_diff_dst_md = diff_dst.get_desc();

    std::cout << "SRC dims size: " << conv_bwd_src_md.dims().size() << "\n";
    std::cout << "Source vector md content: "
              << "\n";
    print_vector(conv_bwd_src_md.dims());
    std::cout << "Weights dims size: " << conv_weights_md.dims().size() << "\n";
    std::cout << "Weights vector md content: "
              << "\n";
    print_vector(conv_weights_md.dims());
    std::cout << "Dst dims size: " << conv_diff_dst_md.dims().size() << "\n";
    std::cout << "Dst vector md content: "
              << "\n";
    print_vector(conv_diff_dst_md.dims());
    std::cout << "Bias dims size: " << conv_bias_md.dims().size() << "\n";
    std::cout << "Bias vector md content: "
              << "\n";
    print_vector(conv_bias_md.dims());

    std::cout << "Setting dimensions\n";
    dnnl::memory::dims conv_strides = {stride_length, stride_length};
    dnnl::memory::dims conv_dilates = {dilation, dilation};
    dnnl::memory::dims conv_padding = {padding_length, padding_length};

    // Recreate forward descriptor since it is needed to create the backward primitive descriptor

    std::cout << "Recreating Convolutional layer primitive descriptor\n";
    auto conv_fwd_desc = dnnl::convolution_forward::desc(
        dnnl::prop_kind::forward, dnnl::algorithm::convolution_direct,
        conv_bwd_src_md, conv_weights_md, conv_bias_md, conv_fwd_dst_md,
        conv_strides, conv_dilates, conv_padding, conv_padding);

    std::cout << "Creating Convolutional layer primitive descriptor\n";

    auto conv_fwd_pd =
        dnnl::convolution_forward::primitive_desc(conv_fwd_desc, eng);

    std::cout << "Creating backward Convolutional layer primitive descriptor\n";
    auto conv_bwd_desc = dnnl::convolution_backward_data::desc(
        dnnl::algorithm::convolution_direct, conv_diff_src_md, conv_weights_md,
        conv_diff_dst_md, conv_strides, conv_dilates, conv_padding,
        conv_padding);

    auto conv_bwd_pd = dnnl::convolution_backward_data::primitive_desc(
        conv_bwd_desc, eng, conv_fwd_pd);

    std::cout << "Checking diff dst memory type\n";
    auto conv_diff_dst_memory =
        checkType(conv_bwd_pd.diff_dst_desc(), diff_dst, net, net_args, eng);

    arg_diff_src = conv_diff_src_memory;
    arg_diff_dst = conv_diff_dst_memory;
    arg_weights = conv_weights;

    net.push_back(dnnl::convolution_backward_data(conv_bwd_pd));
    net_args.push_back(
        {{DNNL_ARG_DIFF_SRC, conv_diff_src_memory},
         {DNNL_ARG_DIFF_DST, conv_diff_dst_memory},
         // If something does not work check this, there might be some
         // reordering needed done in a similar fashion to cnn_training_f32.cpp
         {DNNL_ARG_WEIGHTS, conv_weights}});
}

MaxPool2D_back::MaxPool2D_back(
    int kernel_size, int stride_length, MaxPool2D maxpool_fwd,
    dnnl::memory diff_dst_mem, std::vector<dnnl::primitive>& net,
    std::vector<std::unordered_map<int, dnnl::memory>>& net_args,
    dnnl::engine eng) {
    auto src_md = maxpool_fwd.arg_src.get_desc();

    long batch = src_md.dims()[0];
    long channels = src_md.dims()[1];
    long input_height = src_md.dims()[2];
    long input_width = src_md.dims()[3];
    long padding = 0;

    const dnnl::memory::dim output_height =
        (input_height - ((kernel_size - 1) * 1 + kernel_size) + padding +
         padding) /
            stride_length +
        1;
    const dnnl::memory::dim output_width =
        (input_width - ((kernel_size - 1) * 1 + kernel_size) + padding +
         padding) /
            stride_length +
        1;

    // Source (src) and destination (dst) tensors dimensions.
    dnnl::memory::dims src_dims = {batch, channels, input_height, input_width};
    dnnl::memory::dims dst_dims = {batch, channels, output_height,
                                   output_width};

    // Kernel dimensions.
    dnnl::memory::dims kernel_dims = {kernel_size, kernel_size};
    // Strides, padding dimensions.
    dnnl::memory::dims strides_dims = {stride_length, stride_length};
    dnnl::memory::dims padding_dims_l = {padding, padding};
    dnnl::memory::dims padding_dims_r = {padding, padding};
    dnnl::memory::dims dilation = {1, 1};

    auto diff_dst_md = maxpool_fwd.arg_dst.get_desc();
    auto diff_src_md = maxpool_fwd.arg_src.get_desc();
    auto diff_src_mem = dnnl::memory(diff_src_md, eng);
    std::cout << "Memory allocated\n";

    // Create descriptor.
    auto pooling_bwd_desc = dnnl::pooling_backward::desc(
        dnnl::algorithm::pooling_max, diff_src_md, diff_dst_md, strides_dims,
        kernel_dims,  padding_dims_l, padding_dims_r);
    auto pooling_fwd_desc = dnnl::pooling_forward::desc(
        dnnl::prop_kind::forward_training, dnnl::algorithm::pooling_max,
        diff_src_md, diff_dst_md, strides_dims, kernel_dims,
        padding_dims_l, padding_dims_r);
    auto pooling_fwd_pd =
        dnnl::pooling_forward::primitive_desc(pooling_fwd_desc, eng);
    std::cout << "Created descriptor\n";
    auto pooling_pd = dnnl::pooling_backward::primitive_desc(
        pooling_bwd_desc, eng, pooling_fwd_pd);
    std::cout << "Created primitive descriptor\n";

    arg_diff_src = diff_src_mem;
    arg_diff_dst = diff_dst_mem;

    net.push_back(dnnl::pooling_backward(pooling_pd));
    net_args.push_back({{DNNL_ARG_DIFF_SRC, diff_src_mem},
                        {DNNL_ARG_DIFF_DST, diff_dst_mem},
                        {DNNL_ARG_WORKSPACE, maxpool_fwd.arg_workspace}});
}

Dense_back_data::Dense_back_data(
    dnnl::memory diff_dst, Dense dense_fwd, std::vector<dnnl::primitive>& net,
    std::vector<std::unordered_map<int, dnnl::memory>>& net_args,
    dnnl::engine eng) {

    // INPUT: diff_dst, weights, bias OUTPUT: diff_src

    // Create memory area for backward pass (get types from dense_fwd)
    auto fc_diff_src_memory = dnnl::memory(dense_fwd.arg_src.get_desc(), eng);

    // Get inputs from the forward layer
    auto fc_weights = dense_fwd.arg_weights;
    auto fc_weights_md = fc_weights.get_desc();

    // This is only used to recreate fwd primitive
    auto fc_fwd_dst_md = dense_fwd.arg_dst.get_desc();
    auto fc_diff_dst_md = diff_dst.get_desc();
    auto fc_bias_md = dense_fwd.arg_bias.get_desc();
    auto fc_diff_src_md = fc_diff_src_memory.get_desc();

    // Initialize diff_src and diff_dst to zero
    std::vector<float> diff_fc_src(product(fc_diff_src_md.dims()));

    std::cout << "Initializing diff src: \n";
    for (int i = 0; i < diff_fc_src.size(); i++) {
        diff_fc_src[i] = 0;
    }
    std::cout << "\n";

    write_to_dnnl_memory(diff_fc_src.data(), fc_diff_src_memory);

    // Recreate forward descriptor (see conv2dback)

    std::cout << "Dimensions:\n";
    for (int i = 0; i < fc_diff_src_md.dims().size(); i++)
        std::cout << fc_diff_src_md.dims()[i] << " ";
    std::cout << "\n";
    for (int i = 0; i < fc_weights_md.dims().size(); i++)
        std::cout << fc_weights_md.dims()[i] << " ";
    std::cout << "\n";
    for (int i = 0; i < fc_bias_md.dims().size(); i++)
        std::cout << fc_bias_md.dims()[i] << " ";
    std::cout << "\n";
    for (int i = 0; i < fc_fwd_dst_md.dims().size(); i++)
        std::cout << fc_fwd_dst_md.dims()[i] << " ";
    std::cout << "\n";

    auto fc_fwd_desc = dnnl::inner_product_forward::desc(
        dnnl::prop_kind::forward_training, fc_diff_src_md, fc_weights_md,
        fc_bias_md, fc_fwd_dst_md);

    auto fc_fwd_pd =
        dnnl::inner_product_forward::primitive_desc(fc_fwd_desc, eng);

    std::cout << "Creating inner product data gradient primitive\n";

    auto fc_bwd_desc = dnnl::inner_product_backward_data::desc(
        fc_diff_src_md, fc_weights_md, fc_diff_dst_md);

    std::cout << "Created inner product data gradient primitive\n";

    auto fc_bwd_pd = dnnl::inner_product_backward_data::primitive_desc(
        fc_bwd_desc, eng, fc_fwd_pd);

    std::cout << "Checking memory type dst\n";
    std::cout << "The size of net_back is: " << net_args.size() << "\n";

    // Don't forget that this is the actual input
    auto fc_diff_dst_memory =
        checkType(fc_bwd_pd.diff_dst_desc(), diff_dst, net, net_args, eng);

    std::cout << "Adding backward\n";

    // Set dnnl::memory pointers inside class
    arg_diff_src = fc_diff_src_memory;
    arg_diff_dst = fc_diff_dst_memory;
    arg_weights = fc_weights;

    net.push_back(dnnl::inner_product_backward_data(fc_bwd_pd));
    net_args.push_back(
        {{DNNL_ARG_DIFF_SRC, fc_diff_src_memory},
         // fc_diff_dst_memory, not diff_dst since it might not have passed checkType
         {DNNL_ARG_DIFF_DST, fc_diff_dst_memory},
         // If something does not work check this, there might be some
         // reordering needed done in a similar fashion to cnn_training_f32.cpp
         {DNNL_ARG_WEIGHTS, fc_weights}});
}

// Only because eltwise has no weights!!!
Eltwise_back::Eltwise_back(
    dnnl::algorithm activation, float alpha, float beta, Eltwise eltwise_fwd,
    dnnl::memory diff_dst, std::vector<dnnl::primitive>& net,
    std::vector<std::unordered_map<int, dnnl::memory>>& net_args,
    dnnl::engine eng) {

    auto diff_dst_md = diff_dst.get_desc();
    //auto diff_src_md = dnnl::memory::desc(diff_dst_md.dims(), dt::f32, tag::any);

    auto diff_src_md = diff_dst_md;

    auto diff_src_mem = dnnl::memory(diff_src_md, eng);

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

Conv2D_back_weights::Conv2D_back_weights(
    dnnl::memory diff_dst, Conv2D conv2d_fwd, int stride_length,
    int padding_length, int dilation, std::vector<dnnl::primitive>& net,
    std::vector<std::unordered_map<int, dnnl::memory>>& net_args,
    dnnl::engine eng) {

    std::cout << "Allocating memory for backward convolution\n";
    // Create memory area for backward pass (get types from conv2d_fwd)
    auto conv_diff_weights_memory =
        dnnl::memory(conv2d_fwd.arg_weights.get_desc(), eng);
    auto conv_diff_bias_memory =
        dnnl::memory(conv2d_fwd.arg_bias.get_desc(), eng);

    std::cout << "Obtaining memory descriptors for backward convolution\n";
    // create memory descriptors for f32 convolution data
    auto conv_bwd_src_md = conv2d_fwd.arg_src.get_desc();
    auto conv_diff_weights_md = conv2d_fwd.arg_weights.get_desc();
    // Get dst descriptor to recreate forward primitive
    auto conv_fwd_dst_md = conv2d_fwd.arg_dst.get_desc();

    auto conv_diff_dst_md = diff_dst.get_desc();
    auto conv_diff_bias_md = conv2d_fwd.arg_bias.get_desc();

    std::cout << "SRC dims size: " << conv_bwd_src_md.dims().size() << "\n";
    std::cout << "Source vector md content: "
              << "\n";
    print_vector(conv_bwd_src_md.dims());
    std::cout << "Weights dims size: " << conv_diff_weights_md.dims().size()
              << "\n";
    std::cout << "Weights vector md content: "
              << "\n";
    print_vector(conv_diff_weights_md.dims());
    std::cout << "Dst dims size: " << conv_diff_dst_md.dims().size() << "\n";
    std::cout << "Dst vector md content: "
              << "\n";
    print_vector(conv_diff_dst_md.dims());
    std::cout << "Bias dims size: " << conv_diff_bias_md.dims().size() << "\n";
    std::cout << "Bias vector md content: "
              << "\n";
    print_vector(conv_diff_bias_md.dims());

    std::cout << "Setting dimensions\n";
    dnnl::memory::dims conv_strides = {stride_length, stride_length};
    dnnl::memory::dims conv_dilates = {dilation, dilation};
    dnnl::memory::dims conv_padding = {padding_length, padding_length};

    // Recreate forward descriptor since it is needed to create the backward primitive descriptor

    std::cout << "Recreating Convolutional layer primitive descriptor\n";
    auto conv_fwd_desc = dnnl::convolution_forward::desc(
        dnnl::prop_kind::forward, dnnl::algorithm::convolution_direct,
        conv_bwd_src_md, conv_diff_weights_md,
        conv_fwd_dst_md, conv_strides, conv_padding,
        conv_padding);
    std::cout << "Settings post-ops\n";

    auto conv_fwd_pd =
        dnnl::convolution_forward::primitive_desc(conv_fwd_desc, eng);

    auto conv_bwd_src_memory = dnnl::memory(conv_bwd_src_md, eng);

    std::cout
        << "Creating backwrard Convolutional layer primitive descriptor\n";
    auto conv_bwd_desc = dnnl::convolution_backward_weights::desc(
        dnnl::algorithm::convolution_direct, conv_bwd_src_md,
        conv_diff_weights_md, conv_diff_bias_md, conv_diff_dst_md, conv_strides,
        conv_dilates, conv_padding, conv_padding);

    auto conv_bwd_pd = dnnl::convolution_backward_weights::primitive_desc(
        conv_bwd_desc, eng, conv_fwd_pd);

    conv_bwd_src_memory = checkType(conv_bwd_pd.src_desc(), conv2d_fwd.arg_src,
                                    net, net_args, eng);
    auto conv_diff_dst_memory =
        checkType(conv_bwd_pd.diff_dst_desc(), diff_dst, net, net_args, eng);

    arg_src = conv_bwd_src_memory;
    arg_diff_dst = conv_diff_dst_memory;
    arg_diff_weights = conv_diff_weights_memory;
    arg_diff_bias = conv_diff_bias_memory;

    net.push_back(dnnl::convolution_backward_weights(conv_bwd_pd));
    net_args.push_back(
        {{DNNL_ARG_SRC, conv_bwd_src_memory},
         {DNNL_ARG_DIFF_DST, conv_diff_dst_memory},
         // If something does not work check this, there might be some
         // reordering needed done in a similar fashion to cnn_training_f32.cpp
         {DNNL_ARG_DIFF_WEIGHTS, conv_diff_weights_memory},
         {DNNL_ARG_DIFF_BIAS, conv_diff_bias_memory}});
}

Dense_back_weights::Dense_back_weights(
    dnnl::memory diff_dst, Dense dense_fwd, std::vector<dnnl::primitive>& net,
    std::vector<std::unordered_map<int, dnnl::memory>>& net_args,
    dnnl::engine eng) {
    // INPUT: diff_dst (ie. diff_src of previous layer), src OUTPUT: diff_weights, diff_bias

    // Create memory area for backward pass (get types from dense_fwd)
    auto fc_diff_weights_memory =
        dnnl::memory(dense_fwd.arg_weights.get_desc(), eng);
    auto fc_diff_bias_memory = dnnl::memory(dense_fwd.arg_bias.get_desc(), eng);

    // create memory descriptors for f32 convolution data
    auto fc_bwd_src_md = dense_fwd.arg_src.get_desc();
    auto fc_diff_weights_md = dense_fwd.arg_weights.get_desc();

    // This is only used to recreate fwd primitive
    auto fc_fwd_dst_md = dense_fwd.arg_dst.get_desc();
    auto fc_diff_dst_md = diff_dst.get_desc();
    auto fc_diff_bias_md = dense_fwd.arg_bias.get_desc();

    std::vector<float> diff_fc_weights(product(fc_diff_weights_md.dims()));
    std::vector<float> diff_fc_bias(product(fc_diff_bias_md.dims()));

    std::cout << "Initializing diff weights: \n";
    for (int i = 0; i < diff_fc_weights.size(); i++) {
        diff_fc_weights[i] = 0;
    }
    std::cout << "\n";

    std::cout << "Initializing diff bias: \n";
    for (int i = 0; i < diff_fc_bias.size(); i++) {
        diff_fc_bias[i] = 0;
    }
    std::cout << "\n";

    write_to_dnnl_memory(diff_fc_weights.data(), fc_diff_weights_memory);
    write_to_dnnl_memory(diff_fc_bias.data(), fc_diff_bias_memory);

    // Recreate forward descriptor (see conv2dback)

    auto fc_fwd_desc = dnnl::inner_product_forward::desc(
        dnnl::prop_kind::forward_training, fc_bwd_src_md, fc_diff_weights_md,
        fc_diff_bias_md, fc_fwd_dst_md);

    std::cout << "Creating inner product weights gradient primitive\n";

    auto fc_fwd_pd =
        dnnl::inner_product_forward::primitive_desc(fc_fwd_desc, eng);

    auto fc_bwd_desc = dnnl::inner_product_backward_weights::desc(
        fc_bwd_src_md, fc_diff_weights_md, fc_diff_bias_md, fc_diff_dst_md);

    std::cout << "Created inner product weights gradient primitive\n";

    auto fc_bwd_pd = dnnl::inner_product_backward_weights::primitive_desc(
        fc_bwd_desc, eng, fc_fwd_pd);

    std::cout << "Allocating source memory\n";
    auto fc_bwd_src_memory = dnnl::memory(fc_bwd_src_md, eng);
    std::cout << "Checking memory type src \n";
    fc_bwd_src_memory =
        checkType(fc_bwd_pd.src_desc(), dense_fwd.arg_src, net, net_args, eng);
    std::cout << "Checking memory type dst\n";
    std::cout << "The size of net_back is: " << net_args.size() << "\n";

    // Don't forget that this is the actual input
    auto fc_diff_dst_memory =
        checkType(fc_bwd_pd.diff_dst_desc(), diff_dst, net, net_args, eng);

    std::cout << "Adding backward\n";

    if (fc_diff_weights_memory.get_desc() != fc_bwd_pd.diff_weights_desc()) {
        std::cout << "Formats are different\n";
    }

    std::cout << "Adding to net\n";

    arg_src = fc_bwd_src_memory;
    arg_diff_dst = fc_diff_dst_memory;
    arg_diff_weights = fc_diff_weights_memory;
    arg_diff_bias = fc_diff_bias_memory;

    net.push_back(dnnl::inner_product_backward_weights(fc_bwd_pd));
    net_args.push_back(
        {{DNNL_ARG_SRC, fc_bwd_src_memory},
         {DNNL_ARG_DIFF_DST, fc_diff_dst_memory},
         // If something does not work check this, there might be some
         // reordering needed done in a similar fashion to cnn_training_f32.cpp
         {DNNL_ARG_DIFF_WEIGHTS, fc_diff_weights_memory},
         {DNNL_ARG_DIFF_BIAS, fc_diff_bias_memory}});
}

Eltwise::Eltwise(dnnl::algorithm activation, float alpha, float beta,
                 dnnl::memory input, std::vector<dnnl::primitive>& net,
                 std::vector<std::unordered_map<int, dnnl::memory>>& net_args,
                 dnnl::engine eng) {

    auto src_md = input.get_desc();

    auto dst_mem = dnnl::memory(src_md, eng);
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

#endif