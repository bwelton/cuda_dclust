
NVCC=/opt/nvidia/cudatoolkit7.0/7.0.28-1.0502.10280.4.1/bin/nvcc
CUDA_INCLUDES = -I/opt/nvidia/cudatoolkit7.0/7.0.28-1.0502.10280.4.1/include -I/opt/nvidia/cudatoolkit7.0/7.0.28-1.0502.10280.4.1/extras/CUPTI/include -I/opt/nvidia/cudatoolkit7.0/7.0.28-1.0502.10280.4.1/extras/Debugger/include
OPTS = -std=c++11 -O4 -arch=sm_30
PREFIX = ${DESTDIR}/usr/local
LOCAL_INCLUDES = -I/sw/xk6/boost/1.57.0/cle5.2up04_gnu4.9.0/include 
LIBS=-L/sw/xk6/boost/1.57.0/cle5.2up04_gnu4.9.0/lib -lboost_date_time -ldl

SOURCE_FILES = dbscan.cu

HEADER_FILES = dbscan.h

all: dbscan
	echo "building all"

dbscan:
	$(NVCC) $(OPTS) $(CUDA_INCLUDES) $(SOURCE_FILES) $(LOCAL_INCLUDES) $(LIBS) -o ../../bin/cuda_dbscan 

install: dbscan 
	echo "Installed to $(PREFIX)/bin"
