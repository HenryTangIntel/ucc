#
# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See file LICENSE for terms.
#

sources = \
	tl_hccl_lib.c \
	tl_hccl_context.c \
	tl_hccl_team.c \
	tl_hccl_coll.c \
	tl_hccl.c

module_LTLIBRARIES     = libucc_tl_hccl.la
libucc_tl_hccl_la_SOURCES  = $(sources)
libucc_tl_hccl_la_CPPFLAGS = $(AM_CPPFLAGS) $(HCCL_CPPFLAGS)
libucc_tl_hccl_la_CFLAGS   = $(BASE_CFLAGS)
libucc_tl_hccl_la_LDFLAGS  = -version-info $(SOVERSION) --as-needed $(HCCL_LDFLAGS)
libucc_tl_hccl_la_LIBADD   = $(HCCL_LIBADD) $(UCC_TOP_BUILDDIR)/src/libucc.la

include $(top_srcdir)/config/module.am
