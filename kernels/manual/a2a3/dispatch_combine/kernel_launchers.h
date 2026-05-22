#pragma once

#include <stdint.h>

void launchDispatchCombine(float *out, float *x, int32_t *expert_idx, float *probs, int32_t *active, int32_t *count,
                           int32_t *src_expert_offset, int32_t *expanded_row_idx, float *src_packed_x,
                           float *dispatch_x, float *return_y, int ranks, int experts_per_rank, int tokens, int hidden,
                           int topk, void *stream);
