#
# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See file LICENSE for terms.
#

sources = \
	tl_hccl_lib.cpp \
	tl_hccl_context.cpp \
	tl_hccl_team.cpp \
	tl_hccl_coll.cpp \
	tl_hccl.cpp

module_LTLIBRARIES     = libucc_tl_hccl.la
libucc_tl_hccl_la_SOURCES  = $(sources)
libucc_tl_hccl_la_CPPFLAGS = $(BASE_CPPFLAGS) $(AM_CPPFLAGS) $(HCCL_CPPFLAGS)
libucc_tl_hccl_la_CXXFLAGS = $(BASE_CXXFLAGS) -std=c++11
libucc_tl_hccl_la_LDFLAGS  = -version-info $(SOVERSION) --as-needed $(HCCL_LDFLAGS)
libucc_tl_hccl_la_LIBADD   = $(HCCL_LIBADD) $(UCC_TOP_BUILDDIR)/src/libucc.la

include $(top_srcdir)/config/module.am
