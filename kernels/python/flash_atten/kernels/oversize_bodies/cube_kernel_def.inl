@pto.func(kernel="cube")
def cube_kernel(
    gm_slot_buffer: "ptr_fp32",
    gm_slot_buffer_fp16: "ptr_fp16",
    gm_q: "ptr_fp16",
    gm_k: "ptr_fp16",
    gm_v: "ptr_fp16",
    s0_i64: "i64",
    s1_i64: "i64",
) -> None:
    _exec_body("cube_kernel_body.inl", dict(module_env, **locals()))
