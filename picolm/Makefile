CC      = gcc
CFLAGS  = -O2 -std=c11 -D_GNU_SOURCE -Wall -Wextra -Wpedantic
LDFLAGS = -lm -lpthread
SRCS    = picolm.c model.c tensor.c quant.c tokenizer.c sampler.c grammar.c
TARGET  = picolm

PREFIX  ?= /usr/local
MODEL_DIR ?= /opt/picolm/models

TOOLCHAIN_DIR = riscv64-lp64d--glibc--stable-2025.08-1
TOOLCHAIN_PATH = $(shell pwd)/$(TOOLCHAIN_DIR)

# --- Native build (auto-detect CPU, enables SSE2/AVX on x86) ---
native: CFLAGS += -march=native
native: $(TARGET)

$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# --- Static build for single-binary deployment ---
static: CFLAGS += -march=native
static: LDFLAGS += -static
static: $(TARGET)

# --- Pi 3/4/5 (64-bit ARM with NEON SIMD) ---
pi: CFLAGS += -march=armv8-a+simd
pi: $(TARGET)

# --- Pi Zero / Pi 1 (32-bit ARM) ---
pi-arm32: CC = arm-linux-gnueabihf-gcc
pi-arm32: CFLAGS += -march=armv6 -mfpu=vfp
pi-arm32: $(TARGET)

# --- Cross-compile for Pi64 from platforms (e.g. x86) ---
cross-pi: CC = aarch64-linux-gnu-gcc
cross-pi: CFLAGS += -march=armv8-a+simd
cross-pi: LDFLAGS += -static
cross-pi: $(TARGET)

# --- RISC-V (Sipeed LicheeRV, etc.) ---
riscv: CC = riscv64-linux-gnu-gcc
riscv: CFLAGS += -march=rv64gc -mabi=lp64d
riscv: $(TARGET)

#---- modified cross compiler
riscv-mod: CC = $(TOOLCHAIN_PATH)/bin/riscv64-linux-gcc
# --sysroot tells the compiler to use the headers inside the Bootlin folder
riscv-mod: CFLAGS += -march=rv64gc -mabi=lp64d --sysroot=$(TOOLCHAIN_PATH)/riscv64-buildroot-linux-gnu/sysroot -static
riscv-mod: LDFLAGS += -static
riscv-mod: $(TARGET)

# --- Debug build ---
debug: CFLAGS = -std=c11 -Wall -Wextra -Wpedantic -g -O0
debug: $(TARGET)

# --- Install ---
install: $(TARGET)
	install -d $(DESTDIR)$(PREFIX)/bin
	install -m 755 $(TARGET) $(DESTDIR)$(PREFIX)/bin/
	@echo "Installed $(TARGET) to $(DESTDIR)$(PREFIX)/bin/"

# --- Download default model ---
model:
	@mkdir -p $(MODEL_DIR)
	@if [ ! -f "$(MODEL_DIR)/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf" ]; then \
		echo "Downloading TinyLlama 1.1B Q4_K_M (638 MB)..."; \
		curl -L -o "$(MODEL_DIR)/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf" \
			"https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"; \
		echo "Done."; \
	else \
		echo "Model already exists at $(MODEL_DIR)/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"; \
	fi

model_update:
	@mkdir -p $(MODEL_DIR)
	@if [ ! -f "$(MODEL_DIR)/Qwen3.5-0.8B.Q4_K_M.gguf" ]; then \
                echo "Downloading Qwen 3.5 - 0.8B..."; \
                curl -L -o "$(MODEL_DIR)/Qwen3.5-0.8B.Q4_K_M.gguf" \
		"https://huggingface.co/Jackrong/Qwen3.5-0.8B-Claude-4.6-Opus-Reasoning-Distilled-GGUF/resolve/main/Qwen3.5-0.8B.Q4_K_M.gguf" && echo "Done."; \
        else \
                echo "Model already exists at $(MODEL_DIR)/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"; \
        fi
clean:
	rm -f $(TARGET) $(TARGET).exe *.obj *.o

.PHONY: native static pi pi-arm32 cross-pi riscv cross-riscv debug install model clean
