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
#include "utils/ucc_math.h"

static inline ucc_status_t
ucc_tl_hccl_coll_base_init(ucc_base_coll_args_t *coll_args,
                           ucc_base_team_t *team, ucc_tl_hccl_task_t **task_h)
{
    ucc_tl_hccl_team_t    *hccl_team = ucc_derived_of(team, ucc_tl_hccl_team_t);
    ucc_tl_hccl_context_t *ctx       = UCC_TL_HCCL_TEAM_CTX(hccl_team);
    ucc_tl_hccl_task_t    *task;

    task = ucc_mpool_get(&ctx->req_mp);
    if (ucc_unlikely(!task)) {
        return UCC_ERR_NO_MEMORY;
    }

    UCC_TL_HCCL_PROFILE_REQUEST_NEW(task, "hccl_coll", 0);
    ucc_coll_task_init(&task->super, coll_args, team);

    task->hccl_progress_st = UCC_INPROGRESS;
    task->host_status      = UCC_INPROGRESS;
    task->dev_status       = NULL;
    task->completed        = NULL;

    *task_h = task;
    return UCC_OK;
}

static inline ucc_status_t ucc_tl_hccl_coll_finalize_task(ucc_coll_task_t *coll_task)
{
    ucc_tl_hccl_task_t *task = ucc_derived_of(coll_task, ucc_tl_hccl_task_t);
    
    UCC_TL_HCCL_PROFILE_REQUEST_FREE(task);
    ucc_mpool_put(task);
    return UCC_OK;
}

/* Convert UCC data types to HCCL data types */
static hcclDataType_t ucc_to_hccl_dtype(ucc_datatype_t dt)
{
    switch (dt) {
    case UCC_DT_INT8:
        return hcclInt8;
    case UCC_DT_UINT8:
        return hcclUint8;
    case UCC_DT_INT16:
        return hcclInt16;
    case UCC_DT_UINT16:
        return hcclUint16;
    case UCC_DT_INT32:
        return hcclInt32;
    case UCC_DT_UINT32:
        return hcclUint32;
    case UCC_DT_INT64:
        return hcclInt64;
    case UCC_DT_UINT64:
        return hcclUint64;
    case UCC_DT_FLOAT16:
        return hcclFloat16;
    case UCC_DT_FLOAT32:
        return hcclFloat32;
    case UCC_DT_FLOAT64:
        return hcclFloat64;
    case UCC_DT_BFLOAT16:
        return hcclBfloat16;
    default:
        return hcclFloat32; /* fallback */
    }
}

/* Convert UCC reduction operations to HCCL reduction operations */
static hcclRedOp_t ucc_to_hccl_reduce_op(ucc_reduction_op_t op)
{
    switch (op) {
    case UCC_OP_SUM:
        return hcclSum;
    case UCC_OP_PROD:
        return hcclProd;
    case UCC_OP_MAX:
        return hcclMax;
    case UCC_OP_MIN:
        return hcclMin;
    default:
        return hcclSum; /* fallback */
    }
}

static ucc_status_t ucc_tl_hccl_coll_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_hccl_task_t    *task = ucc_derived_of(coll_task, ucc_tl_hccl_task_t);
    ucc_tl_hccl_team_t    *team = TASK_TEAM(task);
    
    if (task->host_status == UCC_INPROGRESS) {
        /* For now, assume synchronous completion */
        task->host_status = UCC_OK;
        task->super.status = UCC_OK;
    }
    
    return task->super.status;
}

static ucc_status_t ucc_tl_hccl_coll_post(ucc_coll_task_t *coll_task)
{
    /* HCCL operations are typically enqueued and asynchronous */
    ucc_tl_hccl_task_t *task = ucc_derived_of(coll_task, ucc_tl_hccl_task_t);
    
    task->super.status = UCC_INPROGRESS;
    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(task)->pq, &task->super);
}

ucc_status_t ucc_tl_hccl_allreduce_init(ucc_base_coll_args_t *coll_args,
                                        ucc_base_team_t *team,
                                        ucc_coll_task_t **task_h)
{
    ucc_tl_hccl_task_t    *task;
    ucc_tl_hccl_team_t    *hccl_team = ucc_derived_of(team, ucc_tl_hccl_team_t);
    ucc_coll_args_t       *args      = &coll_args->args;
    ucc_status_t           status;
    hcclResult_t           hccl_st;

    status = ucc_tl_hccl_coll_base_init(coll_args, team, &task);
    if (status != UCC_OK) {
        return status;
    }

    task->super.post     = ucc_tl_hccl_coll_post;
    task->super.progress = ucc_tl_hccl_coll_progress;
    task->super.finalize = ucc_tl_hccl_coll_finalize_task;

    /* Enqueue HCCL allreduce operation */
    hccl_st = hcclAllReduce(args->src.info.buffer,
                           args->dst.info.buffer,
                           args->dst.info.count,
                           ucc_to_hccl_dtype(args->dst.info.datatype),
                           ucc_to_hccl_reduce_op(args->op),
                           hccl_team->hccl_comm,
                           hccl_team->stream);

    if (hccl_st != hcclSuccess) {
        tl_error(UCC_TASK_LIB(task), "failed to start HCCL allreduce");
        ucc_tl_hccl_coll_finalize_task(&task->super);
        return UCC_ERR_NO_MESSAGE;
    }

    *task_h = &task->super;
    return UCC_OK;
}

ucc_status_t ucc_tl_hccl_allgather_init(ucc_base_coll_args_t *coll_args,
                                        ucc_base_team_t *team,
                                        ucc_coll_task_t **task_h)
{
    ucc_tl_hccl_task_t    *task;
    ucc_tl_hccl_team_t    *hccl_team = ucc_derived_of(team, ucc_tl_hccl_team_t);
    ucc_coll_args_t       *args      = &coll_args->args;
    ucc_status_t           status;
    hcclResult_t           hccl_st;

    status = ucc_tl_hccl_coll_base_init(coll_args, team, &task);
    if (status != UCC_OK) {
        return status;
    }

    task->super.post     = ucc_tl_hccl_coll_post;
    task->super.progress = ucc_tl_hccl_coll_progress;
    task->super.finalize = ucc_tl_hccl_coll_finalize_task;

    hccl_st = hcclAllGather(args->src.info.buffer,
                           args->dst.info.buffer,
                           args->src.info.count,
                           ucc_to_hccl_dtype(args->src.info.datatype),
                           hccl_team->hccl_comm,
                           hccl_team->stream);

    if (hccl_st != hcclSuccess) {
        tl_error(UCC_TASK_LIB(task), "failed to start HCCL allgather");
        ucc_tl_hccl_coll_finalize_task(&task->super);
        return UCC_ERR_NO_MESSAGE;
    }

    *task_h = &task->super;
    return UCC_OK;
}

ucc_status_t ucc_tl_hccl_broadcast_init(ucc_base_coll_args_t *coll_args,
                                        ucc_base_team_t *team,
                                        ucc_coll_task_t **task_h)
{
    ucc_tl_hccl_task_t    *task;
    ucc_tl_hccl_team_t    *hccl_team = ucc_derived_of(team, ucc_tl_hccl_team_t);
    ucc_coll_args_t       *args      = &coll_args->args;
    ucc_status_t           status;
    hcclResult_t           hccl_st;

    status = ucc_tl_hccl_coll_base_init(coll_args, team, &task);
    if (status != UCC_OK) {
        return status;
    }

    task->super.post     = ucc_tl_hccl_coll_post;
    task->super.progress = ucc_tl_hccl_coll_progress;
    task->super.finalize = ucc_tl_hccl_coll_finalize_task;

    hccl_st = hcclBroadcast(args->src.info.buffer,
                           args->dst.info.buffer,
                           args->dst.info.count,
                           ucc_to_hccl_dtype(args->dst.info.datatype),
                           args->root,
                           hccl_team->hccl_comm,
                           hccl_team->stream);

    if (hccl_st != hcclSuccess) {
        tl_error(UCC_TASK_LIB(task), "failed to start HCCL broadcast");
        ucc_tl_hccl_coll_finalize_task(&task->super);
        return UCC_ERR_NO_MESSAGE;
    }

    *task_h = &task->super;
    return UCC_OK;
}

ucc_status_t ucc_tl_hccl_barrier_init(ucc_base_coll_args_t *coll_args,
                                      ucc_base_team_t *team,
                                      ucc_coll_task_t **task_h)
{
    ucc_tl_hccl_task_t    *task;
    ucc_tl_hccl_team_t    *hccl_team = ucc_derived_of(team, ucc_tl_hccl_team_t);
    ucc_status_t           status;
    hcclResult_t           hccl_st;

    status = ucc_tl_hccl_coll_base_init(coll_args, team, &task);
    if (status != UCC_OK) {
        return status;
    }

    task->super.post     = ucc_tl_hccl_coll_post;
    task->super.progress = ucc_tl_hccl_coll_progress;
    task->super.finalize = ucc_tl_hccl_coll_finalize_task;

    hccl_st = hcclBarrier(hccl_team->hccl_comm, hccl_team->stream);

    if (hccl_st != hcclSuccess) {
        tl_error(UCC_TASK_LIB(task), "failed to start HCCL barrier");
        ucc_tl_hccl_coll_finalize_task(&task->super);
        return UCC_ERR_NO_MESSAGE;
    }

    *task_h = &task->super;
    return UCC_OK;
}

ucc_status_t ucc_tl_hccl_coll_init(ucc_base_coll_args_t *coll_args,
                                   ucc_base_team_t *team,
                                   ucc_coll_task_t **task_h)
{
    ucc_coll_args_t *args = &coll_args->args;

    switch (args->coll_type) {
    case UCC_COLL_TYPE_ALLREDUCE:
        return ucc_tl_hccl_allreduce_init(coll_args, team, task_h);
    case UCC_COLL_TYPE_ALLGATHER:
        return ucc_tl_hccl_allgather_init(coll_args, team, task_h);
    case UCC_COLL_TYPE_BCAST:
        return ucc_tl_hccl_broadcast_init(coll_args, team, task_h);
    case UCC_COLL_TYPE_BARRIER:
        return ucc_tl_hccl_barrier_init(coll_args, team, task_h);
    default:
        tl_error(team->context->lib, "collective %s is not supported by TL HCCL",
                 ucc_coll_type_str(args->coll_type));
        return UCC_ERR_NOT_SUPPORTED;
    }
}

ucc_status_t ucc_tl_hccl_coll_finalize(ucc_coll_task_t *coll_task)
{
    return ucc_tl_hccl_coll_finalize_task(coll_task);
}
