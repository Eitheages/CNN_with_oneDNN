#ifndef BASIC
#define BASIC

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
        // std::cout << "Memory mismatch adding reorder primitive\n";
        auto mem_reordered = dnnl::memory(md_true_type, eng);
        net.push_back(dnnl::reorder(mem_to_check, mem_reordered));
        net_args.push_back(
            {{DNNL_ARG_FROM, mem_to_check}, {DNNL_ARG_TO, mem_reordered}});
    }
    return mem_reordered;
}

void updateWeights_SGD(
    dnnl::memory weights, dnnl::memory diff_weights, float learning_rate,
    std::vector<dnnl::primitive>& net,
    std::vector<std::unordered_map<int, dnnl::memory>>& net_args,
    dnnl::engine eng) {

    std::vector<dnnl::memory> sub_vector = {weights, diff_weights};
    std::vector<dnnl::memory::desc> sub_vector_md = {sub_vector[0].get_desc(),
                                                     sub_vector[1].get_desc()};

    // Minibatch gradient descent needs normalization
    const long minibatch_size = sub_vector_md[0].dims()[0];
    std::vector<float> scales = {1.f, (learning_rate) * (-1.f)};

    auto weights_update_pd =
        dnnl::sum::primitive_desc(sub_vector_md[0], scales, sub_vector_md, eng);

    net.push_back(dnnl::sum(weights_update_pd));

    std::unordered_map<int, dnnl::memory> sum_args;

    sum_args.insert({DNNL_ARG_DST, sub_vector[0]});
    for (int i = 0; i < sub_vector.size(); ++i) {
        sum_args.insert({DNNL_ARG_MULTIPLE_SRC + i, sub_vector[i]});
    }

    net_args.push_back(sum_args);
}

#endif