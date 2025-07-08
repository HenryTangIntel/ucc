/**
 * Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "config.h"
#include "tl_hccl.h"
#include "utils/ucc_malloc.h"

UCC_CLASS_INIT_FUNC(ucc_tl_hccl_context_t,
                    const ucc_base_context_params_t *params,
                    const ucc_base_config_t *config)
{
    ucc_tl_hccl_context_config_t *tl_hccl_config =
        ucc_derived_of(config, ucc_tl_hccl_context_config_t);
    ucc_status_t status;

    UCC_CLASS_CALL_SUPER_INIT(ucc_tl_context_t, &self->super,
                              &tl_hccl_config->super, params->context);

    memcpy(&self->cfg, tl_hccl_config, sizeof(*tl_hccl_config));

    status = ucc_mpool_init(&self->req_mp, 0, sizeof(ucc_tl_hccl_task_t), 0,
                           UCC_CACHE_LINE_SIZE, 8, UINT_MAX,
                           &ucc_coll_task_mpool_ops, params->thread_mode,
                           "tl_hccl_req_mp");
    if (status != UCC_OK) {
        tl_error(self->super.super.lib, "failed to create task mpool");
        return status;
    }

    self->scratch_buf = NULL;
    tl_debug(self->super.super.lib, "initialized tl context: %p", self);
    return UCC_OK;
}

UCC_CLASS_CLEANUP_FUNC(ucc_tl_hccl_context_t)
{
    tl_debug(self->super.super.lib, "finalizing tl context: %p", self);
    if (self->scratch_buf) {
        ucc_free(self->scratch_buf);
    }
    ucc_mpool_cleanup(&self->req_mp, 1);
    UCC_CLASS_CALL_SUPER_FINALIZE(ucc_tl_context_t, &self->super);
}
