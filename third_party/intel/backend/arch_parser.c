//===- arch_parser.c ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <iostream>

#include <sycl/sycl.hpp>

#if defined(_WIN32)
#define EXPORT_FUNC __declspec(dllexport)
#else
#define EXPORT_FUNC __attribute__((visibility("default")))
#endif

extern "C" EXPORT_FUNC const char *parse_device_arch(uint64_t dev_arch) {
  sycl::ext::oneapi::experimental::architecture sycl_arch =
      static_cast<sycl::ext::oneapi::experimental::architecture>(dev_arch);
  const char *arch = "";
  switch (sycl_arch) {
#if __SYCL_COMPILER_VERSION >= 20251010
  case sycl::ext::oneapi::experimental::architecture::intel_gpu_bmg_g31:
#endif
  case sycl::ext::oneapi::experimental::architecture::intel_gpu_bmg_g21:
    arch = "bmg";
    break;
  default:
    std::cerr << "Triton XPU supports only the BMG architecture; device architecture is "
              << (uint64_t)sycl_arch << std::endl;
  }

  return arch;
}
