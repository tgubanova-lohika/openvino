// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief a header that defines wrappers for internal GPU plugin-specific
 * shared Video Acceleration device contexts
 * and shared memory tensors which contain Video Acceleration surfaces
 *
 * @file openvino/runtime/gpu/va.hpp
 */
#pragma once

#include <memory>
#include <string>

#include "openvino/runtime/gpu/ocl.hpp"

// clang-format off
#include <va/va.h>
// clang-format on

namespace ov {
namespace runtime {
namespace gpu {

/**
 * @brief This class represents an abstraction for GPU plugin remote tensor
 * which is shared with VA output surface.
 * The plugin object derived from this class can be obtained with VAContext::create_tensor() call.
 * @note User can also obtain OpenCL 2D image handle from this class.
 */
class VASurfaceTensor : public ClImage2DTensor {
public:
    /**
     * @brief Checks that type defined runtime paramters are presented in remote object
     * @param tensor a tensor to check
     */
    static void type_check(const Tensor& tensor) {
        RemoteTensor::type_check(tensor,
                                 {{GPU_PARAM_KEY(DEV_OBJECT_HANDLE), {}},
                                  {GPU_PARAM_KEY(VA_PLANE), {}},
                                  {GPU_PARAM_KEY(SHARED_MEM_TYPE), {GPU_PARAM_VALUE(VA_SURFACE)}}});
    }
    /**
     * @brief VASurfaceID conversion operator for the VASurfaceTensor object.
     * @return `VASurfaceID` handle
     */
    operator VASurfaceID() {
        return static_cast<VASurfaceID>(get_params().at(GPU_PARAM_KEY(DEV_OBJECT_HANDLE)).as<uint32_t>());
    }

    /**
     * @brief Returns plane ID of underlying video decoder surface
     * @return Plane ID
     */
    uint32_t plane() {
        return get_params().at(GPU_PARAM_KEY(VA_PLANE)).as<uint32_t>();
    }
};

/**
 * @brief This class represents an abstraction for GPU plugin remote context
 * which is shared with VA display object.
 * The plugin object derived from this class can be obtained either with
 * ExecutableNetwork::get_context() or Core::create_context() calls.
 * @note User can also obtain OpenCL context handle from this class.
 */
class VAContext : public ClContext {
    using RemoteContext::create_tensor;
    static constexpr const char* device_name = "GPU";

public:
    /**
     * @brief Checks that type defined runtime paramters are presented in remote object
     * @param remote_context remote context to check
     */
    static void type_check(const RemoteContext& remote_context) {
        remote_type_check(
            remote_context.get_params(),
            {{GPU_PARAM_KEY(VA_DEVICE), {}}, {GPU_PARAM_KEY(CONTEXT_TYPE), {GPU_PARAM_VALUE(VA_SHARED)}}});
    }

    /**
     * @brief `VADisplay` conversion operator for the VAContext object.
     * @return Underlying `VADisplay` object handle
     */
    operator VADisplay() {
        return static_cast<VADisplay>(get_params().at(GPU_PARAM_KEY(VA_DEVICE)).as<gpu_handle_param>());
    }

    /**
     * @brief Constructs remote context object from VA display handle
     * @param core OpenVINO Runtime Core object
     * @param device A `VADisplay` to create remote context from
     */
    VAContext(Core& core, VADisplay device) {
        ParamMap context_params = {{GPU_PARAM_KEY(CONTEXT_TYPE), GPU_PARAM_VALUE(VA_SHARED)},
                                   {GPU_PARAM_KEY(VA_DEVICE), static_cast<gpu_handle_param>(device)}};
        *this = core.create_context(device_name, context_params);
    }

    /**
     * @brief This function is used to obtain a NV12 tensor from NV12 VA decoder output.
     * The resulting tensor contains two remote tensors for Y and UV planes of the surface.
     * @param height A height of Y plane
     * @param width A width of Y plane
     * @param nv12_surf NV12 `VASurfaceID` to create NV12 from
     * @return A pair of remote tensors for each plane
     */
    std::pair<VASurfaceTensor, VASurfaceTensor> create_tensor_nv12(const size_t height,
                                                                   const size_t width,
                                                                   const VASurfaceID nv12_surf) {
        ParamMap tensor_params = {{GPU_PARAM_KEY(SHARED_MEM_TYPE), GPU_PARAM_VALUE(VA_SURFACE)},
                                  {GPU_PARAM_KEY(DEV_OBJECT_HANDLE), nv12_surf},
                                  {GPU_PARAM_KEY(VA_PLANE), uint32_t(0)}};
        auto y_tensor = create_tensor(element::u8, {1, 1, height, width}, tensor_params);
        tensor_params[GPU_PARAM_KEY(VA_PLANE)] = uint32_t(1);
        auto uv_tensor = create_tensor(element::u8, {1, 2, height / 2, width / 2}, tensor_params);
        return std::make_pair(y_tensor, uv_tensor);
    }

    /**
     * @brief This function is used to create remote tensor from VA surface handle
     * @param type Tensor element type
     * @param shape Tensor shape
     * @param surface A `VASurfaceID` to create remote tensor from
     * @param plane An index of a plane inside `VASurfaceID` to create tensor from
     * @return A remote tensor wrapping `VASurfaceID`
     */
    inline VASurfaceTensor create_tensor(const element::Type type,
                                         const Shape& shape,
                                         const VASurfaceID surface,
                                         const uint32_t plane = 0) {
        ParamMap params = {{GPU_PARAM_KEY(SHARED_MEM_TYPE), GPU_PARAM_VALUE(VA_SURFACE)},
                           {GPU_PARAM_KEY(DEV_OBJECT_HANDLE), surface},
                           {GPU_PARAM_KEY(VA_PLANE), plane}};
        return create_tensor(type, shape, params);
    }
};
}  // namespace gpu
}  // namespace runtime
}  // namespace ov