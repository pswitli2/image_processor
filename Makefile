# Makefile - image_processor project Makefile

SUBDIRS = src

all:
	-X=`pwd`; \
	for i in $(SUBDIRS); \
	do echo '<<<' $$i '>>>'; cd $$X/$$i; make $@; done

install:
	-X=`pwd`; \
	mkdir -p bin; \
	for i in $(SUBDIRS); \
	do echo '<<<' $$i '>>>'; cd $$X/$$i; make $@; done

clean:
	-X=`pwd`; \
	for i in $(SUBDIRS); \
	do echo '<<<' $$i '>>>'; cd $$X/$$i; make $@; done
	cd ../..
	rm -rf bin
