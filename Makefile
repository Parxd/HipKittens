# really just a convenience makefile

ROCM_PATH   ?= /opt/rocm-7.1.0
PYTHON      ?= python3
VENV_DIR    := /root/HipKittens/.venv
PY_VERSION  := 3.12

HIPCC       := $(ROCM_PATH)/bin/hipcc
PY_INCLUDE  := /usr/include/python$(PY_VERSION)
PY_LIB      := /usr/lib/python$(PY_VERSION)/config-$(PY_VERSION)-x86_64-linux-gnu
PY_LIB2     := /usr/lib/x86_64-linux-gnu
PYBIND_INC  := $(VENV_DIR)/lib/python$(PY_VERSION)/site-packages/pybind11/include
KITTENS_INC := /root/HipKittens/include
PROTO_INC   := /root/HipKittens/prototype

TARGET      := tk_kernel.cpython-$(subst .,,$(PY_VERSION))-x86_64-linux-gnu.so
SRC         := 256_256_64_16.cpp

CXXFLAGS    := -DKITTENS_CDNA3 \
               --offload-arch=gfx942 \
               -std=c++20 -w \
               -shared -fPIC \
               -Rpass-analysis=kernel-resource-usage

INCLUDES    := -I$(KITTENS_INC) \
               -I$(PROTO_INC) \
               -I$(PY_INCLUDE) \
               -I$(PYBIND_INC) \
               -I$(ROCM_PATH)/include \
               -I$(ROCM_PATH)/include/hip

LDFLAGS     := -L$(PY_LIB) -L$(PY_LIB2) -ldl -lm

# ── Targets ──────────────────────────────────────────────────────────────────

.PHONY: all setup deps venv build clean

all: build

setup: deps venv

deps:
	apt-get install -y bear python3.12-venv

venv:
	$(PYTHON) -m venv $(VENV_DIR)
	$(VENV_DIR)/bin/pip install pybind11

compile_commands: $(SRC)
	bear -- $(MAKE) build