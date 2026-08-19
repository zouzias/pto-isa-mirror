@pto.func(kernel="vector")
def vector_kernel(
    gm_slot_buffer: "ptr_fp32",
    gm_slot_buffer_fp16: "ptr_fp16",
    gm_o: "ptr_fp32",
    s0_i64: "i64",
    s1_i64: "i64",
) -> None:
    _exec_body("vector_kernel_body.inl", dict(module_env, **locals()))
