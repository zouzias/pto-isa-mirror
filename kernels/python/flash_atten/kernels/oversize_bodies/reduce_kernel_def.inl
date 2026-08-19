@pto.func(kernel="vector")
def reduce_kernel(
    gm_slot_buffer: "ptr_fp32",
    gm_o: "ptr_fp32",
    s0_i64: "i64",
    s1_i64: "i64",
) -> None:
    _exec_body("reduce_kernel_body.inl", dict(reduce_env, **locals()))
