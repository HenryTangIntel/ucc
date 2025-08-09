/**
 * Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_TL_HCCL_H_
#define UCC_TL_HCCL_H_

#include "components/tl/ucc_tl.h"
#include "components/tl/ucc_tl_log.h"
#include "utils/ucc_mpool.h"
#include "coll_score/ucc_coll_score.h"

#include <hccl.h>

#ifndef UCC_TL_HCCL_DEFAULT_SCORE
#define UCC_TL_HCCL_DEFAULT_SCORE 20
#endif

#ifdef HAVE_PROFILING_TL_HCCL
#include "utils/profile/ucc_profile.h"
#else
#include "utils/profile/ucc_profile_off.h"
#endif

#define UCC_TL_HCCL_PROFILE_FUNC UCC_PROFILE_FUNC
#define UCC_TL_HCCL_PROFILE_FUNC_VOID UCC_PROFILE_FUNC_VOID
#define UCC_TL_HCCL_PROFILE_REQUEST_NEW UCC_PROFILE_REQUEST_NEW
#define UCC_TL_HCCL_PROFILE_REQUEST_EVENT UCC_PROFILE_REQUEST_EVENT
#define UCC_TL_HCCL_PROFILE_REQUEST_FREE UCC_PROFILE_REQUEST_FREE

enum {
    TL_HCCL_COMM_STATE_ERROR,
    TL_HCCL_COMM_STATE_OOB,
    TL_HCCL_COMM_STATE_INIT_TEAM,
    TL_HCCL_COMM_STATE_INIT_COMM,
    TL_HCCL_COMM_STATE_DESTROY_COMM,
    TL_HCCL_COMM_STATE_READY,
};

typedef struct ucc_tl_hccl_iface {
    ucc_tl_iface_t super;
} ucc_tl_hccl_iface_t;

extern ucc_tl_hccl_iface_t ucc_tl_hccl;

typedef struct ucc_tl_hccl_lib_config {
    ucc_tl_lib_config_t super;
} ucc_tl_hccl_lib_config_t;

typedef enum ucc_tl_hccl_completion_sync_type {
    UCC_TL_HCCL_COMPLETION_SYNC_TYPE_EVENT,
    UCC_TL_HCCL_COMPLETION_SYNC_TYPE_MEMOPS,
    UCC_TL_HCCL_COMPLETION_SYNC_TYPE_AUTO,
    UCC_TL_HCCL_COMPLETION_SYNC_TYPE_LAST
} ucc_tl_hccl_completion_sync_type_t;

typedef struct ucc_tl_hccl_context_config {
    ucc_tl_context_config_t            super;
    ucc_tl_hccl_completion_sync_type_t sync_type;
    int                                hccl_cfg_blocking;
    int                                hccl_lazy_init;
} ucc_tl_hccl_context_config_t;

typedef struct ucc_tl_hccl_lib {
    ucc_tl_lib_t super;
} ucc_tl_hccl_lib_t;
UCC_CLASS_DECLARE(ucc_tl_hccl_lib_t, const ucc_base_lib_params_t *,
                  const ucc_base_config_t *);

typedef struct ucc_tl_hccl_context {
    ucc_tl_context_t             super;
    ucc_tl_hccl_context_config_t cfg;
    ucc_mpool_t                  req_mp;
    void                        *scratch_buf;
} ucc_tl_hccl_context_t;
UCC_CLASS_DECLARE(ucc_tl_hccl_context_t, const ucc_base_context_params_t *,
                  const ucc_base_config_t *);

typedef struct ucc_tl_hccl_team {
    ucc_tl_team_t       super;
    int                 comm_state;
    hcclUniqueId       *unique_id;
    void               *oob_req;
    hcclComm_t          hccl_comm;
    void               *stream;  /* Habana stream handle */
} ucc_tl_hccl_team_t;

typedef struct ucc_tl_hccl_task {
    ucc_coll_task_t         super;
    ucc_status_t            host_status;
    ucc_status_t            hccl_progress_st;
    ucc_status_t           *dev_status;
    void                   *completed;
    union {
        struct {
            ucc_mc_buffer_header_t *scratch;
            size_t                  max_count;
        } allgatherv_bcopy;
    };
} ucc_tl_hccl_task_t;

#define TASK_TEAM(_task)                                                       \
    (ucc_derived_of((_task)->super.team, ucc_tl_hccl_team_t))
#define TASK_CTX(_task)                                                        \
    (ucc_derived_of((_task)->super.team->context, ucc_tl_hccl_context_t))
#define TASK_LIB(_task)                                                        \
    (ucc_derived_of((_task)->super.team->context->lib, ucc_tl_hccl_lib_t))
#define TASK_ARGS(_task) (_task)->super.bargs.args

#define UCC_TL_HCCL_SUPPORTED_COLLS                                           \
    (UCC_COLL_TYPE_ALLTOALL       | UCC_COLL_TYPE_ALLTOALLV  |                \
     UCC_COLL_TYPE_ALLGATHER      | UCC_COLL_TYPE_ALLGATHERV |                \
     UCC_COLL_TYPE_ALLREDUCE      | UCC_COLL_TYPE_BCAST      |                \
     UCC_COLL_TYPE_REDUCE_SCATTER | UCC_COLL_TYPE_REDUCE     |                \
     UCC_COLL_TYPE_BARRIER        | UCC_COLL_TYPE_GATHER     |                \
     UCC_COLL_TYPE_GATHERV        | UCC_COLL_TYPE_SCATTER    |                \
     UCC_COLL_TYPE_SCATTERV)

UCC_CLASS_DECLARE(ucc_tl_hccl_team_t, ucc_base_context_t *,
                  const ucc_base_team_params_t *);

static inline ucc_status_t ucc_tl_hccl_check_async_error(hcclResult_t *hccl_status,
                                                         ucc_status_t *task_st,
                                                         hcclComm_t hccl_comm)
{
    /* Check for async errors if HCCL supports it */
    hcclResult_t async_error;
    hcclResult_t st = hcclCommGetAsyncError(hccl_comm, &async_error);
    if (st != hcclSuccess) {
        return UCC_ERR_NO_MESSAGE;
    }
    if (async_error != hcclSuccess) {
        *hccl_status = async_error;
        *task_st = UCC_ERR_NO_MESSAGE;
        return UCC_ERR_NO_MESSAGE;
    }
    return UCC_OK;
}

ucc_status_t ucc_tl_hccl_comm_init(ucc_tl_hccl_team_t *team);

/* Function prototypes for the interface */
ucc_status_t ucc_tl_hccl_team_create_post(ucc_base_context_t *tl_context,
                                          ucc_base_team_t *tl_team);
ucc_status_t ucc_tl_hccl_team_create_test(ucc_base_team_t *tl_team);
ucc_status_t ucc_tl_hccl_team_destroy(ucc_base_team_t *tl_team);
ucc_status_t ucc_tl_hccl_team_get_scores(ucc_base_team_t *tl_team,
                                         ucc_coll_score_t **score_p);

/* Use the generated constructor/destructor names */
#define ucc_tl_hccl_lib_constructor    UCC_CLASS_NEW_FUNC_NAME(ucc_tl_hccl_lib_t)
#define ucc_tl_hccl_lib_destructor     UCC_CLASS_DELETE_FUNC_NAME(ucc_tl_hccl_lib_t)
#define ucc_tl_hccl_context_constructor UCC_CLASS_NEW_FUNC_NAME(ucc_tl_hccl_context_t)
#define ucc_tl_hccl_context_destructor  UCC_CLASS_DELETE_FUNC_NAME(ucc_tl_hccl_context_t)

#define HCCLCHECK_GOTO(_cmd, _label, _st, _lib, _task_st, _comm)               \
    do {                                                                       \
        hcclResult_t e = _cmd;                                                 \
        _st = ucc_tl_hccl_check_async_error(&e, _task_st, _comm);              \
        if (_st != UCC_OK && hcclSuccess != e) {                              \
            tl_error(_lib, "HCCL error %d %s", e, hcclGetErrorString(e));     \
            _st = UCC_ERR_NO_MESSAGE;                                          \
            goto _label;                                                       \
        }                                                                      \
    } while (0)

#define UCC_TL_HCCL_TEAM_LIB(_team)                                            \
    (ucc_derived_of((_team)->super.super.context->lib, ucc_tl_hccl_lib_t))

#define UCC_TL_HCCL_TEAM_CTX(_team)                                            \
    (ucc_derived_of((_team)->super.super.context, ucc_tl_hccl_context_t))

#endif
