@pto.func
def call_both(
    ffts_addr: pto.ffts_type,
    gm_slot_buffer: "ptr_fp32",
    gm_slot_buffer_fp16: "ptr_fp16",
    gm_q: "ptr_fp16",
    gm_k: "ptr_fp16",
    gm_v: "ptr_fp16",
    gm_o: "ptr_fp32",
    s0_i64: "i64",
    s1_i64: "i64",
) -> None:
    pto.set_ffts(ffts_addr)
    pto.call(cube_kernel, gm_slot_buffer, gm_slot_buffer_fp16, gm_q, gm_k, gm_v, s0_i64, s1_i64)
    pto.call(vector_kernel, gm_slot_buffer, gm_slot_buffer_fp16, gm_o, s0_i64, s1_i64)
