// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_selector.h"

namespace kernel_selector {

/**
 * GPU kernel selector for the Bucketize-6 operation
 */
class bucketize_kernel_selector : public kernel_selector_base {
public:
    static bucketize_kernel_selector &Instance();

    bucketize_kernel_selector();

    KernelsData GetBestKernels(const Params &params, const optional_params &options) const override;
};
}  // namespace kernel_selector
