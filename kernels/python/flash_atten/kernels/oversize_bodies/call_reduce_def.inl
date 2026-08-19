@pto.func
def call_reduce(
    ffts_addr: pto.ffts_type,
    gm_slot_buffer: "ptr_fp32",
    gm_slot_buffer_fp16: "ptr_fp16",
    gm_o: "ptr_fp32",
    s0_i64: "i64",
    s1_i64: "i64",
) -> None:
    pto.set_ffts(ffts_addr)
    pto.call(reduce_kernel, gm_slot_buffer, gm_o, s0_i64, s1_i64)
