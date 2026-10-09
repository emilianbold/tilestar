CC ?= cc
UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Darwin)
NATIVE_CPU_FLAG ?= -mcpu=native
else
NATIVE_CPU_FLAG ?= -march=native
endif
SAMPLING_TEST := tests/test_sampling

DEBUG_FLAGS ?= -g
CFLAGS ?= -O3 -ffast-math $(DEBUG_FLAGS) $(NATIVE_CPU_FLAG) -Wall -Wextra -std=c99
OBJCFLAGS ?= -O3 -ffast-math $(DEBUG_FLAGS) $(NATIVE_CPU_FLAG) -Wall -Wextra -fobjc-arc
QUALITY_CFLAGS ?= -O3 $(DEBUG_FLAGS) $(NATIVE_CPU_FLAG) -Wall -Wextra -std=c11

LDLIBS ?= -lm -pthread

DS4_TEST_MODEL ?= ds4flash.gguf
DS4_TEST_MTP ?= gguf/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf
DS4_DSPARK_MODEL ?= $(DS4_TEST_MODEL)
DS4_DSPARK_SUPPORT ?= gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf

ifeq ($(UNAME_S),Darwin)
CFLAGS += -ffunction-sections -fdata-sections -Oz -fno-asynchronous-unwind-tables -flto
OBJCFLAGS += -ffunction-sections -fdata-sections -Oz -fno-asynchronous-unwind-tables -flto
METAL_LDLIBS := $(LDLIBS) -framework Foundation -framework Metal -Wl,-dead_strip
CORE_OBJS = ds4.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_metal.o ds4_layer_pack.o ds4_engram.o puff.o ds4_kernels_gz.o
CPU_CORE_OBJS = ds4_cpu.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_layer_pack.o
else
# This demo branch is Metal-only; the non-Darwin path builds the CPU reference only.
CPU_CORE_OBJS = ds4_cpu.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_layer_pack.o
METAL_LDLIBS := $(LDLIBS)
endif


.PHONY: all help clean test dspark-acceptance dspark-verify-depth mtp-verify-depth cpu

ifeq ($(UNAME_S),Darwin)
.PHONY: gen-kernels gen-check metal-source metal-blob test-tl-cpu

all: ds4 ds4-server ds4-bench ds4-eval ds4-agent

help:
	@echo "DS4 build targets:"
	@echo "  make              Build Metal ./ds4, ./ds4-server, ./ds4-bench, ./ds4-eval, and ./ds4-agent"
	@echo "  make cpu          Build CPU-only ./ds4, ./ds4-server, ./ds4-bench, ./ds4-eval, and ./ds4-agent"
	@echo "  make test         Build and run tests"
	
	@echo "  make dspark-verify-depth  Run DSpark speculative verification smoke if support GGUF is present"
	@echo "  make mtp-verify-depth  Run legacy MTP speculative verification smoke if MTP GGUF is present"
	@echo "  make clean        Remove build outputs"

ds4: ds4_cli.o ds4_help.o ds4_prompt_prefix.o linenoise.o ds4_gpu_args.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_cli.o ds4_help.o ds4_prompt_prefix.o linenoise.o ds4_gpu_args.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-server: ds4_server.o ds4_kvstore.o rax.o ds4_gpu_args.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_server.o ds4_kvstore.o rax.o ds4_gpu_args.o $(CORE_OBJS) $(METAL_LDLIBS)
	strip -x $@

ds4-bench: ds4_bench.o ds4_help.o ds4_gpu_args.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_bench.o ds4_help.o ds4_gpu_args.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-eval: ds4_eval.o ds4_eval_cases.o ds4_help.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_eval.o ds4_eval_cases.o ds4_help.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-agent: ds4_agent.o ds4_help.o ds4_prompt_prefix.o ds4_web.o ds4_kvstore.o linenoise.o ds4_gpu_args.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_agent.o ds4_help.o ds4_prompt_prefix.o ds4_web.o ds4_kvstore.o linenoise.o ds4_gpu_args.o $(CORE_OBJS) $(METAL_LDLIBS)


tests/test_metal_command_memory: tests/test_metal_command_memory.c ds4_gpu.h $(CORE_OBJS)
	$(CC) $(CFLAGS) -I. -o $@ $< $(CORE_OBJS) $(METAL_LDLIBS)

.PHONY: test-metal-command-memory
test-metal-command-memory: tests/test_metal_command_memory
	MTL_DEBUG_LAYER=1 ./tests/test_metal_command_memory
	MTL_DEBUG_LAYER=1 ./tests/test_metal_command_memory row
	MTL_DEBUG_LAYER=1 ./tests/test_metal_command_memory session
	MTL_DEBUG_LAYER=1 ./tests/test_metal_command_memory batch
	MTL_DEBUG_LAYER=1 ./tests/test_metal_command_memory big

cpu: ds4_cli_cpu.o ds4_server_cpu.o ds4_bench_cpu.o ds4_eval_cpu.o ds4_eval_cases.o ds4_agent_cpu.o ds4_help.o ds4_prompt_prefix.o ds4_web.o ds4_kvstore.o linenoise.o rax.o ds4_gpu_args_cpu.o $(CPU_CORE_OBJS)
	$(CC) $(CFLAGS) -o ds4 ds4_cli_cpu.o ds4_help.o ds4_prompt_prefix.o linenoise.o ds4_gpu_args_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-server ds4_server_cpu.o ds4_help.o ds4_kvstore.o rax.o ds4_gpu_args_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-bench ds4_bench_cpu.o ds4_help.o ds4_gpu_args_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-eval ds4_eval_cpu.o ds4_eval_cases.o ds4_help.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-agent ds4_agent_cpu.o ds4_help.o ds4_prompt_prefix.o ds4_web.o ds4_kvstore.o linenoise.o ds4_gpu_args_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
endif


# TileLang-generated kernels (see tilelang-kernels/README.md). gen/ is checked in;
# regenerating needs a built TileLang checkout (TILELANG_ROOT, ~/tilelang).
TILELANG_ROOT ?= $(HOME)/tilelang
TL_PYTHON ?= $(if $(wildcard $(TILELANG_ROOT)/.venv/bin/python3),$(TILELANG_ROOT)/.venv/bin/python3,python3)
TL_CPU_CXX ?= g++
TL_CPU_CXXFLAGS ?= -O3 -ffast-math $(DEBUG_FLAGS) $(NATIVE_CPU_FLAG) -std=c++17 -w -Ithird_party/tilelang
TL_CPU_SRCS := $(wildcard gen/cpu/*.cpp)
TL_CPU_OBJS := $(TL_CPU_SRCS:.cpp=.o)

gen-kernels:
	TILELANG_ROOT=$(TILELANG_ROOT) $(TL_PYTHON) tilelang-kernels/gen.py

gen-check:
	TILELANG_ROOT=$(TILELANG_ROOT) $(TL_PYTHON) tilelang-kernels/gen.py --check

# The embedded Metal source is generated, never committed:
# prelude.metal + gen/metal/tl_*.metal + no-op stand-ins for the
# hand-written kernels this build does not carry
# (tools/assemble-metal-source.py), raw-deflated into ds4_kernels_gz.c
# (tools/gen-metal-blob.py) and linked into every binary via CORE_OBJS.
# ds4_metal.m is a prerequisite because the assembler reads the host's
# kernel-name literals from it.
TUNED ?= tilelang-kernels/tuned/apple-m5-max.json

build:
	mkdir -p $@

build/ds4_source.metal: prelude.metal tools/assemble-metal-source.py ds4_metal.m $(wildcard gen/metal/tl_*.metal) | build
	python3 tools/assemble-metal-source.py prelude.metal gen/metal $@

ds4_kernels_gz.c: build/ds4_source.metal $(TUNED) tools/gen-metal-blob.py
	python3 tools/gen-metal-blob.py build/ds4_source.metal $(TUNED) > $@.tmp || { rm -f $@.tmp; exit 1; }
	mv $@.tmp $@

# Aliases for forcing/inspecting the generated files by hand.
metal-source: build/ds4_source.metal
metal-blob: ds4_kernels_gz.c

gen/cpu/%.o: gen/cpu/%.cpp gen/cpu/tl_launch_meta.h
	$(TL_CPU_CXX) $(TL_CPU_CXXFLAGS) -c -o $@ $<

tests/test_tl_cpu.o: tests/test_tl_cpu.c ds4.c ds4.h gen/cpu/tl_launch_meta.h
	$(CC) $(CFLAGS) -Wno-unused-function -DDS4_NO_GPU -I. -c -o $@ $<

tests/test_tl_cpu: tests/test_tl_cpu.o $(TL_CPU_OBJS) $(filter-out ds4_cpu.o,$(CPU_CORE_OBJS))
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS) -lstdc++

test-tl-cpu: tests/test_tl_cpu
	./tests/test_tl_cpu

ifeq ($(UNAME_S),Darwin)
.PHONY: test-tl-metal
tests/test_tl_metal.o: tests/test_tl_metal.c ds4_gpu.h ds4_iq2_tables.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<

tests/test_tl_metal: tests/test_tl_metal.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)

test-tl-metal: tests/test_tl_metal
	./tests/test_tl_metal
endif


ds4.o: ds4.c ds4.h ds4_ssd.h ds4_distributed.h ds4_gpu.h ds4_gpu_tp.h ds4_deepseek41_gpu.h ds4_linux_memory.h ds4_engram.h
	$(CC) $(CFLAGS) -c -o $@ ds4.c

ds4_ssd.o: ds4_ssd.c ds4_ssd.h
	$(CC) $(CFLAGS) -c -o $@ ds4_ssd.c

ds4_engram.o: ds4_engram.c ds4_engram.h
	$(CC) $(filter-out -ffast-math,$(CFLAGS)) -c -o $@ ds4_engram.c

ds4_cli.o: ds4_cli.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h ds4_prompt_prefix.h linenoise.h
	$(CC) $(CFLAGS) -c -o $@ ds4_cli.c

ds4_distributed.o: ds4_distributed.c ds4_distributed.h ds4.h ds4_ssd.h
	$(CC) $(CFLAGS) -c -o $@ ds4_distributed.c

ds4_tp.o: ds4_tp.c ds4_tp.h ds4.h ds4_ssd.h ds4_gpu.h ds4_gpu_tp.h
	$(CC) $(CFLAGS) -c -o $@ ds4_tp.c

ds4_help.o: ds4_help.c ds4_help.h
	$(CC) $(CFLAGS) -c -o $@ ds4_help.c

ds4_prompt_prefix.o: ds4_prompt_prefix.c ds4_prompt_prefix.h ds4.h
	$(CC) $(CFLAGS) -c -o $@ ds4_prompt_prefix.c

ds4_gpu_args.o: ds4_gpu_args.c ds4_gpu_args.h ds4_gpu_mgpu.h
	$(CC) $(CFLAGS) -c -o $@ ds4_gpu_args.c

ds4_server.o: ds4_server.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h ds4_kvstore.h rax.h
	$(CC) $(CFLAGS) -c -o $@ ds4_server.c

ds4_bench.o: ds4_bench.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h
	$(CC) $(CFLAGS) -c -o $@ ds4_bench.c

ds4_eval.o: ds4_eval.c ds4_eval_cases.h ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h
	$(CC) $(CFLAGS) -c -o $@ ds4_eval.c

ds4_eval_cases.o: ds4_eval_cases.c ds4_eval_cases.h
	$(CC) $(CFLAGS) -c -o $@ ds4_eval_cases.c

ds4_agent.o: ds4_agent.c ds4.h ds4_ssd.h ds4_distributed.h ds4_tp.h ds4_help.h ds4_prompt_prefix.h ds4_kvstore.h ds4_web.h linenoise.h
	$(CC) $(CFLAGS) -c -o $@ ds4_agent.c

ds4_web.o: ds4_web.c ds4_web.h
	$(CC) $(CFLAGS) -c -o $@ ds4_web.c

ds4_kvstore.o: ds4_kvstore.c ds4_kvstore.h ds4.h ds4_ssd.h
	$(CC) $(CFLAGS) -c -o $@ ds4_kvstore.c

ds4_test.o: tests/ds4_test.c ds4_server.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h ds4_kvstore.h rax.h
	$(CC) $(CFLAGS) -Wno-unused-function -c -o $@ tests/ds4_test.c

ds4_agent_test.o: tests/ds4_agent_test.c ds4_agent.c ds4.h ds4_ssd.h ds4_distributed.h ds4_tp.h ds4_help.h ds4_prompt_prefix.h ds4_kvstore.h ds4_web.h linenoise.h
	$(CC) $(CFLAGS) -Wno-unused-function -c -o $@ tests/ds4_agent_test.c


rax.o: rax.c rax.h rax_malloc.h
	$(CC) $(CFLAGS) -c -o $@ rax.c

linenoise.o: linenoise.c linenoise.h
	$(CC) $(CFLAGS) -c -o $@ linenoise.c

ds4_cpu.o: ds4.c ds4.h ds4_ssd.h ds4_distributed.h ds4_gpu.h ds4_gpu_tp.h
	$(CC) $(CFLAGS) -Wno-unused-function -DDS4_NO_GPU -c -o $@ ds4.c

ds4_cli_cpu.o: ds4_cli.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h ds4_prompt_prefix.h linenoise.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_cli.c

ds4_gpu_args_cpu.o: ds4_gpu_args.c ds4_gpu_args.h ds4_gpu_mgpu.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_gpu_args.c

ds4_server_cpu.o: ds4_server.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h ds4_kvstore.h rax.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_server.c

ds4_bench_cpu.o: ds4_bench.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_bench.c

ds4_eval_cpu.o: ds4_eval.c ds4_eval_cases.h ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_eval.c

ds4_agent_cpu.o: ds4_agent.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h ds4_prompt_prefix.h ds4_kvstore.h ds4_web.h linenoise.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_agent.c

ds4_metal.o: ds4_metal.m ds4.h ds4_gpu.h ds4_gpu_tp.h ds4_deepseek41_gpu.h ds4_image.h ds4_tl_metal.h ds4_iq2_tables.h gen/metal/tl_launch_meta.h
	$(CC) $(OBJCFLAGS) -c -o $@ ds4_metal.m


# DS4_QWEN4_SNAPSHOT=<HF checkpoint dir> DS4_QWEN4_MMPROJ=<mmproj.gguf> DS4_QWEN4_IMAGE=<image>


ds4.o ds4_cpu.o ds4_cpu_test_hooks.o ds4_metal.o: ds4_qwen4_vision.h


ds4_cpu_test_hooks.o: ds4.c ds4.h ds4_image.h ds4_gpu.h ds4_gpu_mgpu.h ds4_layer_pack.h
	$(CC) $(CFLAGS) -Wno-unused-function -DDS4_NO_GPU -DDS4_TEST_HOOKS -c -o $@ ds4.c


tests/test_sampling.o: tests/test_sampling.c ds4.h
	$(CC) $(CFLAGS) -fno-finite-math-only -DDS4_TEST_HOOKS -I. -c -o $@ $<

tests/test_sampling: tests/test_sampling.o ds4_cpu_test_hooks.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_layer_pack.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

tests/test_session_state.o: tests/test_session_state.c ds4.c ds4.h ds4_gpu.h ds4_image.h ds4_tp.h
	$(CC) $(CFLAGS) -Wno-unused-function -DDS4_NO_GPU -I. -c -o $@ $<

tests/test_session_state: tests/test_session_state.o $(filter-out ds4_cpu.o,$(CPU_CORE_OBJS))
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

tests/test_session_state_gpu.o: tests/test_session_state.c ds4.c ds4.h ds4_gpu.h ds4_image.h ds4_tp.h
	$(CC) $(CFLAGS) -Wno-unused-function -I. -c -o $@ $<

tests/test_session_state_gpu: tests/test_session_state_gpu.o $(filter-out ds4.o,$(CORE_OBJS))


.PHONY: test-session-state
test-session-state: tests/test_session_state
	./tests/test_session_state


ds4_test: ds4_test.o ds4_help.o ds4_kvstore.o rax.o $(CORE_OBJS)
ifeq ($(UNAME_S),Darwin)
	$(CC) $(CFLAGS) -o $@ ds4_test.o ds4_help.o ds4_kvstore.o rax.o $(CORE_OBJS) $(METAL_LDLIBS)
else
	$(CC) $(CFLAGS) -o $@ ds4_test.o ds4_help.o ds4_kvstore.o rax.o $(CORE_OBJS) $(LDLIBS)
endif

ds4_agent_test: ds4_agent_test.o ds4_help.o ds4_prompt_prefix.o ds4_web.o ds4_kvstore.o linenoise.o $(CORE_OBJS)
ifeq ($(UNAME_S),Darwin)
	$(CC) $(CFLAGS) -o $@ ds4_agent_test.o ds4_help.o ds4_prompt_prefix.o ds4_web.o ds4_kvstore.o linenoise.o $(CORE_OBJS) $(METAL_LDLIBS)
else
	$(CC) $(CFLAGS) -o $@ ds4_agent_test.o ds4_help.o ds4_prompt_prefix.o ds4_web.o ds4_kvstore.o linenoise.o $(CORE_OBJS) $(LDLIBS)
endif

tests/test_prompt_prefix.o: tests/test_prompt_prefix.c ds4_prompt_prefix.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<

tests/test_prompt_prefix: tests/test_prompt_prefix.o ds4_prompt_prefix.o
	$(CC) $(CFLAGS) -o $@ $^

.PHONY: test-frontends
test-frontends: ds4_test ds4_agent_test test-web-recovery
	./ds4_test --server
	./ds4_agent_test

.PHONY: test-web-recovery
test-web-recovery: tests/test_web_recovery
	python3 tests/test_web_recovery.py ./tests/test_web_recovery

tests/test_web_recovery: tests/test_web_recovery.c ds4_web.c ds4_web.h
	$(CC) $(CFLAGS) -Wno-unused-function -o $@ tests/test_web_recovery.c

test: ds4_test ds4_agent_test ds4-eval q4k-dot-test mxfp4-dot-test test-session-state test-web-recovery \
	tests/test_prompt_prefix $(SAMPLING_TEST) ds4 ds4-server ds4-bench ds4-agent
	./ds4-eval --validate-cases
	./ds4-eval --self-test-extractors
	./ds4_agent_test
	./ds4_test
	./tests/test_prompt_prefix
	./tests/test_sampling

dspark-acceptance: ds4
	DS4_DSPARK_MODEL="$(DS4_DSPARK_MODEL)" \
	DS4_DSPARK_SUPPORT="$(DS4_DSPARK_SUPPORT)" \
	sh tests/dspark_acceptance_fixture.sh

dspark-verify-depth: ds4_test
	@if [ ! -f "$(DS4_TEST_MODEL)" ]; then \
		echo "dspark-verify-depth: skipped, missing model $(DS4_TEST_MODEL)"; \
	elif [ ! -f "$(DS4_DSPARK_SUPPORT)" ]; then \
		echo "dspark-verify-depth: skipped, missing DSpark support $(DS4_DSPARK_SUPPORT)"; \
		echo "dspark-verify-depth: run ./download_model.sh ds4f-dspark or set DS4_DSPARK_SUPPORT=FILE"; \
	else \
		DS4_TEST_MODEL="$(DS4_TEST_MODEL)" DS4_TEST_DSPARK="$(DS4_DSPARK_SUPPORT)" ./ds4_test --dspark-verify-depth; \
	fi

mtp-verify-depth: ds4_test
	@if [ ! -f "$(DS4_TEST_MODEL)" ]; then \
		echo "mtp-verify-depth: skipped, missing model $(DS4_TEST_MODEL)"; \
	elif [ ! -f "$(DS4_TEST_MTP)" ]; then \
		echo "mtp-verify-depth: skipped, missing MTP support $(DS4_TEST_MTP)"; \
		echo "mtp-verify-depth: set DS4_TEST_MTP=FILE to a legacy support GGUF"; \
	else \
		DS4_TEST_MODEL="$(DS4_TEST_MODEL)" DS4_TEST_MTP="$(DS4_TEST_MTP)" ./ds4_test --mtp-verify-depth; \
	fi

q4k-dot-test: tests/test_q4k_dot.c
	$(CC) -O2 -Wall -Wextra -std=c99 -o tests/test_q4k_dot tests/test_q4k_dot.c -lm -pthread
	./tests/test_q4k_dot

mxfp4-dot-test: tests/test_mxfp4_dot.c
	$(CC) -O2 -Wall -Wextra -std=c99 -o tests/test_mxfp4_dot tests/test_mxfp4_dot.c -lm
	./tests/test_mxfp4_dot

ds4.o ds4_cpu.o ds4_agent.o ds4_agent_cpu.o ds4_server.o ds4_server_cpu.o \
ds4_test.o ds4_agent_test.o \
ds4_cpu_test_hooks.o tests/test_session_state.o \
tests/test_session_state_gpu.o: ds4_tool_text.h

clean:
	rm -f ds4_kernels_gz.c ds4_kernels_gz.c.tmp build/ds4_source.metal
	rm -f tests/test_web_recovery
	rm -f tests/test_metal_command_memory
	rm -f tests/test_session_state tests/test_session_state_gpu
	rm -f ds4 ds4-server ds4-bench ds4-eval ds4-agent ds4_cpu ds4_native ds4_server_test ds4_test ds4_agent_test tests/test_q4k_dot tests/test_mxfp4_dot tests/test_prompt_prefix tests/test_sampling tests/*.o *.o

# The active tokenizer includes generated Unicode classes.
ds4.o ds4_cpu.o ds4_cpu_test_hooks.o: ds4_qwen4_unicode.inc
