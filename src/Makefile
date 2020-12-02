CFLAGS=-pthread -Dcimg_use_png -std=c++17
LIBS=-lX11 -lpng
all:
	g++ -c main.cpp $(CFLAGS) `libpng-config --cflags` -o main.o
	g++ -o main main.o $(LIBS) $(CFLAGS) `libpng-config --ldflags`