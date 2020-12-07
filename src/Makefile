# Makefile - image_processor/src/ directory Makefile

# set project file paths
PROJECT_ROOT = ..
BINDIR       = $(PROJECT_ROOT)/bin
INCLUDEDIR   = $(PROJECT_ROOT)/include

# set source/object/executable files
EXECUTABLE = image_processor
SRC        = $(EXECUTABLE:=.cpp)
CUDA_SRC   = $(wildcard cuda/*.cu)
CUDA_OBJ   = $(CUDA_SRC:.cu=.o)
OBJ        = $(CUDA_OBJ) $(EXECUTABLE:=.o)

# set compiler and linker flags
INCLUDES      = -I$(INCLUDEDIR) -Icpu -Icuda -Iframework
LIBS          = -lX11 -lpng `libpng-config --ldflags`
WARNING_FLAGS = -Wall  -Wextra  -Wno-unused-parameter  -pedantic -Werror
FLAGS         = $(INCLUDES)
CXXFLAGS      = $(FLAGS)  -std=c++17 -ggdb -pthread `libpng-config --cflags` -Dcimg_use_png  $(WARNING_FLAGS)
CXX           = g++ $(CXXFLAGS)
NVCC_FLAGS    = $(FLAGS) -Xcompiler -ggdb
NVCC          = nvcc $(NVCC_FLAGS)

# add .cu to SUFFIXES
.SUFFIXES:
.SUFFIXES: .cpp .cu .o

#
# generic make targets
#

.cu.o:
	$(NVCC) -c $< -o $@

.cpp.o:
	$(CXX) -c $< -o $@
	
all: $(EXECUTABLE)

install: $(BINDIR)/$(EXECUTABLE)

$(BINDIR)/$(EXECUTABLE): $(EXECUTABLE)
	install -c -s -m 755 $(EXECUTABLE) $(BINDIR)

$(EXECUTABLE): $(OBJ)
	$(NVCC) $(OBJ) $(LIBS) -o $(EXECUTABLE)
	chmod 755 $(EXECUTABLE)

clean:
	rm -f $(OBJ) $(EXECUTABLE)
