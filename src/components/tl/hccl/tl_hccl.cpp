/**
 * Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "config.h"
#include "tl_hccl.h"
#include "tl_hccl_coll.h"
#include "core/ucc_progress_queue.h"
#include "components/mc/ucc_mc.h"
#include "utils/arch/cpu.h"
#include "utils/ucc_math.h"

/* C++ compatibility - use __typeof__ instead of typeof */
#ifdef __cplusplus
#define typeof __typeof__
#endif

const char* ucc_tl_hccl_completion_sync_names[] = {
    [UCC_TL_HCCL_COMPLETION_SYNC_TYPE_EVENT]  = "event",
    [UCC_TL_HCCL_COMPLETION_SYNC_TYPE_MEMOPS] = "memops",
    [UCC_TL_HCCL_COMPLETION_SYNC_TYPE_AUTO]   = "auto",
    [UCC_TL_HCCL_COMPLETION_SYNC_TYPE_LAST]   = NULL
};

static ucc_config_field_t ucc_tl_hccl_lib_config_table[] __attribute__((unused)) = {
    {"", "", NULL, ucc_offsetof(ucc_tl_hccl_lib_config_t, super),
     UCC_CONFIG_TYPE_TABLE(ucc_tl_lib_config_table)},

    {NULL}
};

static ucc_config_field_t ucc_tl_hccl_context_config_table[] __attribute__((unused)) = {
    {"", "", NULL, ucc_offsetof(ucc_tl_hccl_context_config_t, super),
     UCC_CONFIG_TYPE_TABLE(ucc_tl_context_config_table)},

    {"SYNC", "auto",
     "Completion synchronization mechanism. auto - try to use the most "
     "optimal available mechanism; event - use events; memops - use "
     "memory operations API (available since synapse 1.6)",
     ucc_offsetof(ucc_tl_hccl_context_config_t, sync_type),
     UCS_CONFIG_TYPE_ENUM(ucc_tl_hccl_completion_sync_names)},

    {"HCCL_BLOCKING", "0",
     "Use blocking HCCL operations",
     ucc_offsetof(ucc_tl_hccl_context_config_t, hccl_cfg_blocking),
     UCC_CONFIG_TYPE_INT},

    {"HCCL_LAZY_INIT", "1",
     "Initialize HCCL communicator on first collective call",
     ucc_offsetof(ucc_tl_hccl_context_config_t, hccl_lazy_init),
     UCC_CONFIG_TYPE_INT},

    {NULL}
};

UCC_CLASS_DEFINE_NEW_FUNC(ucc_tl_hccl_lib_t, ucc_base_lib_t,
                          const ucc_base_lib_params_t *,
                          const ucc_base_config_t *);

static ucc_status_t ucc_tl_hccl_lib_get_attr(ucc_base_lib_t *lib,
                                             ucc_base_lib_attr_t *base_attr)
{
    ucc_tl_lib_attr_t *attr = ucc_derived_of(base_attr, ucc_tl_lib_attr_t);

    attr->super.attr.thread_mode = UCC_THREAD_SINGLE;
    attr->super.attr.coll_types  = UCC_TL_HCCL_SUPPORTED_COLLS;
    attr->super.flags           = 0;
    if (base_attr->mask & UCC_BASE_LIB_ATTR_FIELD_MIN_TEAM_SIZE) {
        attr->super.min_team_size = 1;
    }

    if (base_attr->mask & UCC_BASE_LIB_ATTR_FIELD_MAX_TEAM_SIZE) {
        attr->super.max_team_size = UCC_RANK_MAX;
    }

    /* Library name is set by base class */
    return UCC_OK;
}

UCC_CLASS_DEFINE_DELETE_FUNC(ucc_tl_hccl_lib_t, ucc_base_lib_t);

UCC_CLASS_DEFINE_NEW_FUNC(ucc_tl_hccl_context_t, ucc_base_context_t,
                          const ucc_base_context_params_t *,
                          const ucc_base_config_t *);

static ucc_status_t ucc_tl_hccl_context_get_attr(ucc_base_context_t *context,
                                                 ucc_base_ctx_attr_t *base_attr)
{
    /* Context attributes are set by base class */
    (void)context;
    (void)base_attr;
    return UCC_OK;
}

UCC_CLASS_DEFINE_DELETE_FUNC(ucc_tl_hccl_context_t, ucc_base_context_t);

/* Interface will be declared by UCC_TL_IFACE_DECLARE */

/* Use extern "C" for C++ compatibility with UCC interface */
extern "C" {

/* Rename the functions to match expected interface naming */
extern ucc_status_t ucc_tl_hccl_get_lib_attr(const ucc_base_lib_t *lib,
                                             ucc_base_lib_attr_t *base_attr) {
    return ucc_tl_hccl_lib_get_attr(const_cast<ucc_base_lib_t*>(lib), base_attr);
}

extern ucc_status_t ucc_tl_hccl_get_lib_properties(ucc_base_lib_properties_t *prop) {
    prop->default_team_size = 1;
    prop->min_team_size = 1;
    prop->max_team_size = UCC_RANK_MAX;
    return UCC_OK;
}

extern ucc_status_t ucc_tl_hccl_get_context_attr(const ucc_base_context_t *context,
                                                 ucc_base_ctx_attr_t *base_attr) {
    return ucc_tl_hccl_context_get_attr(const_cast<ucc_base_context_t*>(context), base_attr);
}

} /* extern "C" */

/* Create a simple interface definition that works with C++ */
ucc_tl_hccl_iface_t ucc_tl_hccl = {
    /* Interface will be set up by the build system */
};
