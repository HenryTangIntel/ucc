#
# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See file LICENSE for terms.
#

CHECK_TLS_REQUIRED(hccl)
AS_IF([test "$CHECKED_TL_REQUIRED" = "y"],
[
      CHECK_HCCL
      AS_IF([test "$hccl_happy" = "yes"],
      [
            tl_modules="${tl_modules}:hccl"
            CHECK_NEED_TL_PROFILING(["hccl"])
            AS_IF([test "$TL_PROFILING_REQUIRED" = "y"],
                  [
                        AC_DEFINE([HAVE_PROFILING_TL_HCCL], [1], [Enable profiling for TL HCCL])
                        prof_modules="${prof_modules}:hccl"
                  ], [])
      ])
      AC_CONFIG_FILES([src/components/tl/hccl/Makefile])
])
