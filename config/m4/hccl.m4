#
# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See file LICENSE for terms.
#

AC_DEFUN([CHECK_HCCL],[

AS_IF([test "x$hccl_checked" != "xyes"],[

hccl_happy="no"

AC_ARG_WITH([hccl],
            [AS_HELP_STRING([--with-hccl=(DIR)], [Enable the use of HCCL (default is guess).])],
            [], [with_hccl=guess])

AS_IF([test "x$with_hccl" != "xno"],
[
    save_CPPFLAGS="$CPPFLAGS"
    save_CFLAGS="$CFLAGS"
    save_CXXFLAGS="$CXXFLAGS"
    save_LDFLAGS="$LDFLAGS"
    save_LIBS="$LIBS"

    AS_IF([test ! -z "$with_hccl" -a "x$with_hccl" != "xyes" -a "x$with_hccl" != "xguess"],
    [
        check_hccl_dir="$with_hccl"
        check_hccl_libdir="$with_hccl/lib"
        CPPFLAGS="-I$with_hccl/include $save_CPPFLAGS"
        LDFLAGS="-L$check_hccl_libdir $save_LDFLAGS"
    ])

    AS_IF([test ! -z "$with_hccl_libdir" -a "x$with_hccl_libdir" != "xyes"],
    [
        check_hccl_libdir="$with_hccl_libdir"
        LDFLAGS="-L$check_hccl_libdir $save_LDFLAGS"
    ])

    AC_CHECK_HEADERS([hccl.h], [], [hccl_happy=no])
    AS_IF([test "x$hccl_happy" != "xno"],
    [
        AC_CHECK_LIB([hcl], [hcclGetVersion], [HCCL_LIBADD="-lhcl"], [hccl_happy=no])
    ])

    AS_IF([test "x$hccl_happy" = "xyes"],
    [
        AS_IF([test ! -z "$check_hccl_dir"],
        [
            HCCL_CPPFLAGS="-I$check_hccl_dir/include"
        ])

        AS_IF([test ! -z "$check_hccl_libdir"],
        [
            HCCL_LDFLAGS="-L$check_hccl_libdir"
        ])
        hccl_happy="yes"
    ],
    [
        AS_IF([test "x$with_hccl" != "xguess"],
        [
            AC_MSG_ERROR([HCCL support is requested but HCCL packages cannot be found])
        ],
        [
            AC_MSG_WARN([HCCL not found])
        ])
    ])

    CFLAGS="$save_CFLAGS"
    CXXFLAGS="$save_CXXFLAGS"
    CPPFLAGS="$save_CPPFLAGS"
    LDFLAGS="$save_LDFLAGS"
    LIBS="$save_LIBS"
],
[
    AC_MSG_WARN([HCCL was explicitly disabled])
])

hccl_checked=yes
AC_SUBST(HCCL_CPPFLAGS)
AC_SUBST(HCCL_LDFLAGS)
AC_SUBST(HCCL_LIBADD)
AM_CONDITIONAL([HAVE_HCCL], [test "x$hccl_happy" = "xyes"])
])
])
