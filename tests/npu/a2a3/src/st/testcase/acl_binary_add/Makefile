# --------------------------------------------------------------------------------
# Minimal ACL binary-load add ST
#   device : device-only vector kernel (no host code in the TU), compiled with
#            ccec --npu-arch=dav-2201 --cce-aicore-only so the object carries
#            .Kernel_Type / .ascend.meta metadata, then linked with ld.lld
#   host   : loads the device .o with aclrtBinaryLoadFromFile / aclrtBinaryGetFunction
#            and launches it with aclrtLaunchKernelWithConfig
# --------------------------------------------------------------------------------

ASCEND_HOME_PATH ?= /usr/local/Ascend/cann_92b1/cann-9.2.0-beta.1
SOC_VERSION      ?= Ascend910B1
DRIVER_LIB       ?= $(ASCEND_HOME_PATH)/devlib
# Full NPU arch (not dav-c220-vec): makes ccec emit the kernel-type metadata
# (.Kernel_Type_add_custom / .ascend.meta) that the runtime needs to dispatch.
NPU_ARCH         ?= dav-2201

export ASCEND_HOME_PATH

CCEC_BIN  := $(ASCEND_HOME_PATH)/tools/ccec_compiler/bin
CCEC      := $(CCEC_BIN)/ccec
LD_LLD    := $(CCEC_BIN)/ld.lld
LLVM_NM   := llvm-nm
OBJDUMP   := $(CCEC_BIN)/llvm-objdump
SIM_LIB   := $(ASCEND_HOME_PATH)/tools/simulator/$(SOC_VERSION)/lib
TIKCFW    := $(ASCEND_HOME_PATH)/compiler/tikcpp/tikcfw

PTO_ROOT  := $(abspath $(CURDIR)/../../../../../../..)

DEVICE_SRC := add_custom.cpp
DEVICE_PART:= add_custom.part.o
DEVICE_OBJ := add_custom_device.o

HOST_SRC     := main.cpp
HOST_BIN_NPU := acl_binary_add
HOST_BIN_SIM := acl_binary_add_sim

DEVICE_INC := -I$(PTO_ROOT)/include -I$(TIKCFW) -I$(TIKCFW)/interface -I$(TIKCFW)/impl -I$(ASCEND_HOME_PATH)/include

DEVICE_FLAGS := -xcce -O2 -std=c++17 \
  --npu-arch=$(NPU_ARCH) --cce-aicore-only --cce-auto-sync \
  -DMEMORY_BASE -DTILING_KEY_VAR=0

HOST_FLAGS := -O2 -std=c++17 -Wall -I$(ASCEND_HOME_PATH)/include
HOST_LIBS_NPU := -L$(ASCEND_HOME_PATH)/lib64 -lascendcl
# sim: libruntime_camodel must stay in DT_NEEDED and be loaded before libascendcl,
# because it provides the acl*Impl entry points the ascendcl shims call.
HOST_LIBS_SIM := -Wl,--no-as-needed -L$(SIM_LIB) -lruntime_camodel \
  -L$(ASCEND_HOME_PATH)/lib64 -lascendcl -lplatform -lc_sec -lnnopbase \
  -Wl,-rpath,$(SIM_LIB) -Wl,-rpath,$(ASCEND_HOME_PATH)/lib64

.PHONY: all nm run-sim run-npu clean

all: $(DEVICE_OBJ) $(HOST_BIN_SIM) $(HOST_BIN_NPU)

$(DEVICE_PART): $(DEVICE_SRC)
	$(CCEC) $(DEVICE_FLAGS) $(DEVICE_INC) -c $< -o $@

$(DEVICE_OBJ): $(DEVICE_PART)
	$(LD_LLD) -m aicorelinux -Ttext 0 --no-mmap-output-file $< -o $@

$(HOST_BIN_NPU): $(HOST_SRC)
	g++ $(HOST_FLAGS) $< -o $@ $(HOST_LIBS_NPU)

$(HOST_BIN_SIM): $(HOST_SRC)
	g++ $(HOST_FLAGS) $< -o $@ $(HOST_LIBS_SIM)

nm: $(DEVICE_OBJ)
	@echo "--- $(LLVM_NM) --defined-only $(DEVICE_OBJ)"
	@$(LLVM_NM) --defined-only $(DEVICE_OBJ)
	@echo "--- sections"
	@$(OBJDUMP) -h $(DEVICE_OBJ)

run-sim: $(DEVICE_OBJ) $(HOST_BIN_SIM)
	@mkdir -p camodel_log log/ub_log
	CAMODEL_LOG_PATH=camodel_log \
	LD_LIBRARY_PATH=$(SIM_LIB):$(ASCEND_HOME_PATH)/lib64:$(ASCEND_HOME_PATH)/runtime/lib64/stub:$(DRIVER_LIB):$(DRIVER_LIB)/common:$(DRIVER_LIB)/driver \
	./$(HOST_BIN_SIM) ./$(DEVICE_OBJ)

run-npu: $(DEVICE_OBJ) $(HOST_BIN_NPU)
	LD_LIBRARY_PATH=$(ASCEND_HOME_PATH)/lib64:$(ASCEND_HOME_PATH)/runtime/lib64:$$LD_LIBRARY_PATH \
	./$(HOST_BIN_NPU) ./$(DEVICE_OBJ)

clean:
	rm -f $(DEVICE_PART) $(DEVICE_OBJ) $(HOST_BIN_NPU) $(HOST_BIN_SIM)
	rm -rf camodel_log log
