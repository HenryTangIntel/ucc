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
        # Check for standard layout first
        AS_IF([test -d "$with_hccl/include"],
            [CPPFLAGS="-I$with_hccl/include $save_CPPFLAGS"],
            [# Try Habana Labs directory structure
             AS_IF([test -d "$with_hccl/include/habanalabs"],
                [CPPFLAGS="-I$with_hccl/include/habanalabs $save_CPPFLAGS"
                 check_hccl_libdir="$with_hccl/lib/habanalabs"],
                [CPPFLAGS="-I$with_hccl/include $save_CPPFLAGS"])])
        LDFLAGS="-L$check_hccl_libdir $save_LDFLAGS"
    ])

    AS_IF([test ! -z "$with_hccl_libdir" -a "x$with_hccl_libdir" != "xyes"],
    [
        check_hccl_libdir="$with_hccl_libdir"
        LDFLAGS="-L$check_hccl_libdir $save_LDFLAGS"
    ])

    # HCCL requires C++11, so switch to C++ compiler with C++11 flags
    AC_LANG_PUSH([C++])
    save_CXXFLAGS_hccl="$CXXFLAGS"
    CXXFLAGS="$CXXFLAGS -std=c++11"
    
    # Check for hccl.h in standard location first
    AC_CHECK_HEADERS([hccl.h], [hccl_happy=yes], [hccl_happy=no])
    
    # If not found, try habanalabs directory structure  
    AS_IF([test "x$hccl_happy" = "xno"],
    [
        save_CPPFLAGS2="$CPPFLAGS"
        CPPFLAGS="-I/usr/include/habanalabs $CPPFLAGS"
        # Force recheck by clearing cache variable
        unset ac_cv_header_hccl_h
        AC_CHECK_HEADERS([hccl.h], [hccl_happy=yes], [hccl_happy=no])
        AS_IF([test "x$hccl_happy" = "xno"], [CPPFLAGS="$save_CPPFLAGS2"])
    ])
    
    CXXFLAGS="$save_CXXFLAGS_hccl"
    AC_LANG_POP([C++])
    AS_IF([test "x$hccl_happy" != "xno"],
    [
        # HCCL library test - just check if library can be linked
        AC_LANG_PUSH([C++])
        CXXFLAGS="$CXXFLAGS -std=c++11"
        
        # Try standard library path first
        save_LDFLAGS2="$LDFLAGS"
        AC_MSG_CHECKING([for HCCL library])
        LIBS_save="$LIBS"
        LIBS="-lhcl $LIBS"
        AC_LINK_IFELSE([AC_LANG_PROGRAM([[]], [[return 0;]])],
                       [hccl_lib_found=yes], [hccl_lib_found=no])
        
        AS_IF([test "x$hccl_lib_found" = "xno"],
        [
            # Try habanalabs library directory
            LDFLAGS="-L/usr/lib/habanalabs $save_LDFLAGS"
            AC_LINK_IFELSE([AC_LANG_PROGRAM([[]], [[return 0;]])],
                           [hccl_lib_found=yes
                            HCCL_LDFLAGS="-L/usr/lib/habanalabs"], 
                           [hccl_lib_found=no])
        ])
        
        AS_IF([test "x$hccl_lib_found" = "xyes"],
              [HCCL_LIBADD="-lhcl"
               AC_MSG_RESULT([yes])],
              [hccl_happy=no
               AC_MSG_RESULT([no])
               LDFLAGS="$save_LDFLAGS2"])
        
        LIBS="$LIBS_save"
        AC_LANG_POP([C++])
    ])

    AS_IF([test "x$hccl_happy" = "xyes"],
    [
        AS_IF([test ! -z "$check_hccl_dir"],
        [
            HCCL_CPPFLAGS="-I$check_hccl_dir/include"
        ],
        [
            # If using system paths, check if we need habanalabs path
            AS_IF([test -f "/usr/include/habanalabs/hccl.h"],
                [HCCL_CPPFLAGS="-I/usr/include/habanalabs"],
                [HCCL_CPPFLAGS=""])
        ])

        AS_IF([test ! -z "$check_hccl_libdir"],
        [
            HCCL_LDFLAGS="-L$check_hccl_libdir"
        ],
        [
            # If using system paths and library was found in habanalabs, set that path
            AS_IF([test "x$HCCL_LDFLAGS" = "x" -a -f "/usr/lib/habanalabs/libhcl.so"],
                [HCCL_LDFLAGS="-L/usr/lib/habanalabs"])
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
