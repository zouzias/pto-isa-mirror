    size_t ARow = 32, ACol = 64, BRow = 64, BCol = 512, CRow = 32, CCol = 512;
    size_t ASize = ARow * ACol * sizeof(T);
    size_t BSize = BRow * BCol * sizeof(T);
    size_t CSize = CRow * CCol * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost, *srcAHost, *srcBHost, *srcCHost;
    T *dstDevice, *srcADevice, *srcBDevice, *srcCDevice;

    aclrtMallocHost((void **)(&dstHost), CSize);
    aclrtMallocHost((void **)(&srcAHost), ASize);
    aclrtMallocHost((void **)(&srcBHost), BSize);
    aclrtMallocHost((void **)(&srcCHost), CSize);

    aclrtMalloc((void **)&dstDevice, CSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&srcADevice, ASize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&srcBDevice, BSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&srcCDevice, CSize, ACL_MEM_MALLOC_HUGE_FIRST);

    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/a.bin", ASize, srcAHost, ASize));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/b.bin", BSize, srcBHost, BSize));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/c.bin", CSize, srcCHost, CSize));

    aclrtMemcpy(srcADevice, ASize, srcAHost, ASize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(srcBDevice, BSize, srcBHost, BSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(srcCDevice, CSize, srcCHost, CSize, ACL_MEMCPY_HOST_TO_DEVICE);

    if constexpr (key == 1) {
        LaunchTPut_BI_LEFTRIGHT(dstDevice, srcADevice, srcBDevice, srcCDevice);
    } else if constexpr (key == 2) {
        LaunchTPut_BI_TOPDOWN(dstDevice, srcADevice, srcBDevice, srcCDevice);
    } else if constexpr (key == 3) {
        LaunchTPut_BI_NOSPLIT(dstDevice, srcADevice, srcBDevice, srcCDevice);
    } else if constexpr (key == 4) {
        LaunchTPut_C2V_NOSPLIT(dstDevice, srcADevice, srcBDevice, srcCDevice);
    }

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, CSize, dstDevice, CSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", srcCDevice, CSize);

    aclrtFree(dstDevice);
    aclrtFree(srcADevice);
    aclrtFree(srcBDevice);
    aclrtFree(srcCDevice);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcAHost);
    aclrtFreeHost(srcBHost);
    aclrtFreeHost(srcCHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    size_t elem_count = CSize / sizeof(T);

    std::vector<T> golden(elem_count);
    std::vector<T> devFinal(elem_count);
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/golden.bin", CSize, golden.data(), CSize));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/output.bin", CSize, devFinal.data(), CSize));

    bool ret = ResultCmp<T>(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
