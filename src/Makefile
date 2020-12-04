# src/ directory Makefile
# build the image_processor program

PROJECT_ROOT = ..
BINDIR       = $(PROJECT_ROOT)/bin
INCLUDEDIR   = $(PROJECT_ROOT)/include

EXECUTABLE = image_processor
SRC        = $(EXECUTABLE:=.cpp)
OBJ        = $(EXECUTABLE:=.o)

LIBS          = -lX11 -lpng
WARNING_FLAGS = -Wall -Wextra -Wno-unused-parameter -Werror -pedantic
CFLAGS        = -I$(INCLUDEDIR) -std=c++17 -pthread `libpng-config --cflags` -Dcimg_use_png $(WARNING_FLAGS) -ggdb
CXX           = g++ $(CFLAGS)
LDFLAGS       = $(LIBS) `libpng-config --ldflags`

.cpp.o:
	$(CXX) -c $<

all: $(EXECUTABLE)

install: $(BINDIR)/$(EXECUTABLE)

$(BINDIR)/$(EXECUTABLE): $(EXECUTABLE)
	install -c -s -m 755 $(EXECUTABLE) $(BINDIR)

$(EXECUTABLE): $(OBJ)
	$(CXX) $(OBJ) $(LIBS) $(LDFLAGS) -o $(EXECUTABLE)
	chmod 755 $(EXECUTABLE)

clean:
	rm -f $(OBJ) $(EXECUTABLE)
