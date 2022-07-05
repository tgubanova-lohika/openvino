// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "bucketize/bucketize_kernel_ref.hpp"
#include "bucketize/bucketize_kernel_selector.hpp"
#include "bucketize_inst.hpp"
#include "impls/implementation_map.hpp"
#include "primitive_base.hpp"

namespace cldnn {
namespace ocl {

struct bucketize_impl : typed_primitive_impl_ocl<bucketize> {
    using parent = typed_primitive_impl_ocl<bucketize>;
    using parent::parent;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<bucketize_impl>(*this);
    }

    static primitive_impl* create(const bucketize_node& arg, const kernel_impl_params& impl_param) {
        auto params = get_default_params<kernel_selector::bucketize_params>(impl_param);
        auto optional_params =
            get_default_optional_params<kernel_selector::bucketize_optional_params>(arg.get_program());

        auto primitive = arg.get_primitive();
        params.with_right_bound = primitive->with_right_bound;
        params.inputs.push_back(convert_data_tensor(arg.buckets().get_output_layout()));

        const auto& kernel_selector = kernel_selector::bucketize_kernel_selector::Instance();
        const auto best_kernels = kernel_selector.GetBestKernels(params, optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        return new bucketize_impl(arg, best_kernels.front());
    }
};

namespace detail {

attach_bucketize_impl::attach_bucketize_impl() {
    implementation_map<bucketize>::add(impl_types::ocl,
                                       bucketize_impl::create,
                                       {
                                           std::make_tuple(data_types::u8, format::bfyx),
                                           std::make_tuple(data_types::u8, format::bfzyx),
                                           std::make_tuple(data_types::u8, format::bfwzyx),
                                           std::make_tuple(data_types::i8, format::bfyx),
                                           std::make_tuple(data_types::i8, format::bfzyx),
                                           std::make_tuple(data_types::i8, format::bfwzyx),
                                           std::make_tuple(data_types::f16, format::bfyx),
                                           std::make_tuple(data_types::f16, format::bfzyx),
                                           std::make_tuple(data_types::f16, format::bfwzyx),
                                           std::make_tuple(data_types::f32, format::bfyx),
                                           std::make_tuple(data_types::f32, format::bfzyx),
                                           std::make_tuple(data_types::f32, format::bfwzyx),
                                           std::make_tuple(data_types::i32, format::bfyx),
                                           std::make_tuple(data_types::i32, format::bfzyx),
                                           std::make_tuple(data_types::i32, format::bfwzyx),
                                           std::make_tuple(data_types::i64, format::bfyx),
                                           std::make_tuple(data_types::i64, format::bfzyx),
                                           std::make_tuple(data_types::i64, format::bfwzyx),
                                       });
}
}  // namespace detail

}  // namespace ocl
}  // namespace cldnn
