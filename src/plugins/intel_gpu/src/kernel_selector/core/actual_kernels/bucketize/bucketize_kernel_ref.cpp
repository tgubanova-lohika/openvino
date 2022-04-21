// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#include "bucketizes_ref.h"

#include <kernel_selector_utils.h>
#include <random>


namespace kernel_selector {

namespace {


CommonDispatchData SetDefault(const bucketize_params &params, const optional_params &) {
    CommonDispatchData dispatchData;
    dispatchData.gws = {params.output.Batch().v, 1, 1};
    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo);
    return dispatchData;
}

}  // namespace

JitConstants BucketizeKernelRef::GetJitConstants(const bucketize_params &params) const {
    return MakeBaseParamsJitConstants(params);
}


KernelsData BucketizeKernelRef::GetKernelsData(const Params &params, const optional_params &options) const {
    if (!Validate(params, options)) {
        return {};
    }

    KernelData kernel_data = KernelData::Default<bucketize_params>(params);
    const bucketize_params &new_params = dynamic_cast<const bucketize_params &>(*kernel_data.params.get());

    auto dispatch_data = SetDefault(new_params, options);
    auto entry_point = GetEntryPoint(kernelName, new_params.layerID, params, options);

    auto bucketize_jit = GetJitConstants(new_params);
    auto jit = CreateJit(kernelName, bucketize_jit, entry_point);

    FillCLKernelData(kernel_data.kernels[0], dispatch_data, params.engineInfo, kernelName, jit, entry_point, "", false,
                     false, 2);

    KernelsData kernelsData;
    kernelsData.push_back(std::move(kernel_data));
    return kernelsData;
}

KernelsPriority BucketizeKernelRef::GetKernelsPriority(const Params & /*params*/,
                                                                    const optional_params & /*options*/) const {
    return FORCE_PRIORITY_1;
}

ParamsKey BucketizeKernelRef::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputDataType(Datatype::INT32);
    k.EnableInputDataType(Datatype::INT64);

    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableDifferentTypes();
    k.EnableOutputLayout(Tensor::bfyx);
    k.EnableInputLayout(Tensor::bfyx);
    k.EnableBatching();
    return k;
}

bool BucketizeKernelRef::Validate(const Params &params, const optional_params &optionalParams) const {
    if (params.GetType() != KernelType::bucketizeS ||
        optionalParams.GetType() != KernelType::bucketizeS) {
        return false;
    }

    const bucketize_params &new_params = dynamic_cast<const bucketize_params &>(params);
    if (new_params.inputs.size() != 2) {
        return false;
    }
    return true;
}

}  // namespace kernel_selector
