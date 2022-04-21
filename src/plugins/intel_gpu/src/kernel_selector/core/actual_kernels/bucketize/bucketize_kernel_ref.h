// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "kernel_base_opencl.h"

namespace kernel_selector {

/**
 * Bucketize kernel params. All the needed inputs are in base input, output shape is static and
 * presented in output.
 */
struct bucketize_params: public base_params {
    bucketize_params() :
            base_params{KernelType::BUCKETIZE} {
    }

};

/**
 * Specific optional params is not defined for RandomUniform operation.
 */
struct bucketize_optional_params: optional_params {
    bucketize_optional_params() :
            optional_params{KernelType::bucketize} {
    }
};

/**
 * Reference GPU kernel for the RandomUniform-8 operation.
 */
class BucketizeKernelRef: public KernelBaseOpenCL {
public:
    BucketizeKernelRef() :
            KernelBaseOpenCL{"bucketize_ref"} {
    }
private:
    KernelsData GetKernelsData(const Params &params, const optional_params &options) const override;

    KernelsPriority GetKernelsPriority(const Params &params, const optional_params &options) const override;

    ParamsKey GetSupportedKey() const override;

    bool Validate(const Params &params, const optional_params &optionalParams) const override;

    JitConstants GetJitConstants(const bucketize_params &params) const;
};

} /* namespace kernel_selector */
