# src/ directory Makefile
# build the image_processor program

PROJECT_ROOT = ..
BINDIR       = $(PROJECT_ROOT)/bin
INCLUDES     = -I$(PROJECT_ROOT)/include -Icpu -Icuda -Iframework

EXECUTABLE = image_processor
SRC        = $(EXECUTABLE:=.cpp)
CUDA_SRC   = $(wildcard cuda/*.cu)
CUDA_OBJ   = $(CUDA_SRC:.cu=.o)
OBJ        = $(CUDA_OBJ) $(EXECUTABLE:=.o)

LIBS          = -lX11 -lpng
WARNING_FLAGS = -Wall  -Wextra  -Wno-unused-parameter  -pedantic -Werror
FLAGS         = $(INCLUDES)
CXXFLAGS      = $(FLAGS)  -std=c++17 -ggdb -pthread `libpng-config --cflags` -Dcimg_use_png  $(WARNING_FLAGS)
CXX           = g++ $(CXXFLAGS)
NVCC_FLAGS    = $(FLAGS) -Xcompiler -ggdb
NVCC          = nvcc $(NVCC_FLAGS)
LDFLAGS       = $(LIBS) `libpng-config --ldflags`

.SUFFIXES:
.SUFFIXES: .cpp .cu .o

.cu.o:
	$(NVCC) -c $< -o $@

.cpp.o:
	$(CXX) -c $< -o $@
	
all: $(EXECUTABLE)

install: $(BINDIR)/$(EXECUTABLE)

$(BINDIR)/$(EXECUTABLE): $(EXECUTABLE)
	install -c -s -m 755 $(EXECUTABLE) $(BINDIR)

$(EXECUTABLE): $(OBJ)
	$(NVCC) $(OBJ) $(LIBS) $(LDFLAGS) -o $(EXECUTABLE)
	chmod 755 $(EXECUTABLE)

clean:
	rm -f $(OBJ) $(EXECUTABLE)
