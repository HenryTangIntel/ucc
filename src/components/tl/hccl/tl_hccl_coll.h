/**
 * Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_TL_HCCL_COLL_H_
#define UCC_TL_HCCL_COLL_H_

#include "tl_hccl.h"

ucc_status_t ucc_tl_hccl_coll_init(ucc_base_coll_args_t *coll_args,
                                   ucc_base_team_t *team,
                                   ucc_coll_task_t **task_h);

ucc_status_t ucc_tl_hccl_coll_finalize(ucc_coll_task_t *coll_task);

ucc_status_t ucc_tl_hccl_allreduce_init(ucc_base_coll_args_t *coll_args,
                                        ucc_base_team_t *team,
                                        ucc_coll_task_t **task_h);

ucc_status_t ucc_tl_hccl_allgather_init(ucc_base_coll_args_t *coll_args,
                                        ucc_base_team_t *team,
                                        ucc_coll_task_t **task_h);

ucc_status_t ucc_tl_hccl_broadcast_init(ucc_base_coll_args_t *coll_args,
                                        ucc_base_team_t *team,
                                        ucc_coll_task_t **task_h);

ucc_status_t ucc_tl_hccl_reduce_init(ucc_base_coll_args_t *coll_args,
                                     ucc_base_team_t *team,
                                     ucc_coll_task_t **task_h);

ucc_status_t ucc_tl_hccl_barrier_init(ucc_base_coll_args_t *coll_args,
                                      ucc_base_team_t *team,
                                      ucc_coll_task_t **task_h);

#endif
