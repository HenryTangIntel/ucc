/**
 * Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "config.h"
#include "tl_hccl.h"
#include "utils/ucc_malloc.h"
#include "core/ucc_team.h"

ucc_status_t ucc_tl_hccl_comm_init(ucc_tl_hccl_team_t *team)
{
    ucc_tl_hccl_context_t *ctx  = UCC_TL_HCCL_TEAM_CTX(team);
    ucc_base_team_t       *tl_team = &team->super.super;
    hcclResult_t           hccl_st;
    ucc_status_t           status;

    hccl_st = hcclCommInitRank(&team->hccl_comm, tl_team->size, 
                              *(team->unique_id), tl_team->rank);
    if (hccl_st != hcclSuccess) {
        tl_error(ctx->super.super.lib, "failed to initialize HCCL communicator");
        return UCC_ERR_NO_MESSAGE;
    }

    team->comm_state = TL_HCCL_COMM_STATE_READY;
    tl_debug(ctx->super.super.lib, "initialized HCCL communicator");
    return UCC_OK;
}

UCC_CLASS_INIT_FUNC(ucc_tl_hccl_team_t, ucc_base_context_t *tl_context,
                    const ucc_base_team_params_t *params)
{
    ucc_tl_hccl_context_t *ctx = ucc_derived_of(tl_context, ucc_tl_hccl_context_t);
    ucc_status_t           status;

    UCC_CLASS_CALL_SUPER_INIT(ucc_tl_team_t, &self->super, tl_context, params);

    self->comm_state = TL_HCCL_COMM_STATE_OOB;
    self->unique_id  = NULL;
    self->oob_req    = NULL;
    self->stream     = NULL;

    if (!ctx->cfg.hccl_lazy_init) {
        /* Initialize HCCL communicator immediately */
        status = ucc_tl_hccl_comm_init(self);
        if (status != UCC_OK) {
            return status;
        }
    }

    tl_debug(tl_context->lib, "created team: %p", self);
    return UCC_OK;
}

UCC_CLASS_CLEANUP_FUNC(ucc_tl_hccl_team_t)
{
    tl_debug(self->super.super.context->lib, "finalizing team: %p", self);

    if (self->hccl_comm) {
        hcclCommDestroy(self->hccl_comm);
    }
    if (self->unique_id) {
        ucc_free(self->unique_id);
    }
    UCC_CLASS_CALL_SUPER_FINALIZE(ucc_tl_team_t, &self->super);
}

ucc_status_t ucc_tl_hccl_team_create_post(ucc_base_context_t *tl_context,
                                          ucc_base_team_t *tl_team)
{
    ucc_tl_hccl_context_t *ctx = ucc_derived_of(tl_context, ucc_tl_hccl_context_t);
    ucc_tl_hccl_team_t    *team;
    ucc_status_t           status;

    status = UCC_TL_HCCL_TEAM_INIT(team, tl_context, tl_team);
    if (status != UCC_OK) {
        return status;
    }
    *tl_team = &team->super.super;
    return UCC_OK;
}

ucc_status_t ucc_tl_hccl_team_destroy(ucc_base_team_t *tl_team)
{
    ucc_tl_hccl_team_t *team = ucc_derived_of(tl_team, ucc_tl_hccl_team_t);

    UCC_CLASS_DELETE_FUNC_NAME(ucc_tl_hccl_team_t)(team);
    return UCC_OK;
}

ucc_status_t ucc_tl_hccl_team_get_scores(ucc_base_team_t   *tl_team,
                                         ucc_coll_score_t **score_p)
{
    ucc_tl_hccl_team_t *team = ucc_derived_of(tl_team, ucc_tl_hccl_team_t);
    ucc_base_context_t *ctx  = UCC_TL_TEAM_CTX(team);
    ucc_memory_type_t   mem_types[] = {UCC_MEMORY_TYPE_HOST};
    ucc_coll_score_t   *score;
    ucc_status_t        status;
    int                 i;
    ucc_coll_score_team_info_t team_info;

    team_info.alg_id             = 0;
    team_info.team               = &team->super;
    team_info.team_size          = UCC_TL_TEAM_SIZE(team);
    team_info.topo               = NULL;

    status = ucc_coll_score_alloc(&score);
    if (UCC_OK != status) {
        tl_error(ctx->lib, "failed to alloc score");
        return status;
    }

    for (i = 0; i < UCC_TL_HCCL_SUPPORTED_COLLS; i++) {
        /* Add default score for each supported collective */
        status = ucc_coll_score_add_range(
            score, UCC_COLL_TYPE_ALLREDUCE, UCC_MEMORY_TYPE_HOST, 0,
            UCC_MSG_MAX, UCC_TL_HCCL_DEFAULT_SCORE,
            ucc_tl_hccl_coll_init, &team->super);
        if (UCC_OK != status) {
            tl_error(ctx->lib, "failed to add score range");
            goto err;
        }
    }

    *score_p = score;
    return UCC_OK;

err:
    ucc_coll_score_free(score);
    return status;
}
