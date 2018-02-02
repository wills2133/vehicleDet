#
# (C) Copyleft 2011
# Late Lee from http://www.latelee.org
# 
# A simple Makefile for *ONE* project(c or/and cpp file) in *ONE*  directory
#
# note: 
# you can put head file(s) in 'include' directory, so it looks 
# a little neat.
#
# usage: $ make
#        $ make debug=y
###############################################################################
#t mean to set arm compile
# testmode = t
#cross compile...
ifeq ($(testmode), t)
CROSS_COMPILE = arm-linux-
else
CROSS_COMPILE =
endif

CC = $(CROSS_COMPILE)gcc
CXX = $(CROSS_COMPILE)g++
AR = $(CROSS_COMPILE)ar

ARFLAGS = cr
RM = -rm -rf
MAKE = make

ifeq ($(testmode), t)
CFLAGS = -shared -Wall -fpic
else
CFLAGS = -Wall
endif

#debug = y

ifeq ($(debug), y)
CFLAGS += -g
else
CFLAGS += -O3 -s
endif

DEFS = 

CFLAGS += $(DEFS)

LDFLAGS = $(LIBS) -pthread
#CFLAGS+=-ljpeg

OPENCV_INCLUDE_PATH=/home/wills/opencv/opencv-2.4.11/modules
OPENCV_ROOT_PATH=/home/wills/opencv/opencv-2.4.11

ifeq ($(testmode), t)
PLIB=/usr/local/opencv_arm_2.4.11/
else
#PLIB=/usr/local/lib/
PLIB=/usr/local/opencv-2.4.11/
endif


INCLUDE_OPENCV=\
	-I$(INCLUDE_DIR)\
	-I$(OPENCV_ROOT_PATH)/include/opencv\
	-I$(OPENCV_ROOT_PATH)/include/opencv2\
	-I$(OPENCV_ROOT_PATH)/include\
	-I$(OPENCV_INCLUDE_PATH)/contrib/include/\
	-I$(OPENCV_INCLUDE_PATH)/legacy/include\
	-I$(OPENCV_INCLUDE_PATH)/calib3d/include\
	-I$(OPENCV_INCLUDE_PATH)/features2d/include\
	-I$(OPENCV_INCLUDE_PATH)/nonfree/include\
	-I$(OPENCV_INCLUDE_PATH)/photo/include\
	-I$(OPENCV_INCLUDE_PATH)/objdetect/include\
	-I$(OPENCV_INCLUDE_PATH)/video/include\
	-I$(OPENCV_INCLUDE_PATH)/ml/include\
	-I$(OPENCV_INCLUDE_PATH)/highgui/include\
	-I$(OPENCV_INCLUDE_PATH)/imgproc/include\
	-I$(OPENCV_INCLUDE_PATH)/flann/include\
	-I$(OPENCV_INCLUDE_PATH)/core/include\


INCDIRS = ./

CFLAGS += -I$(INCDIRS)
CFLAGS += $(INCLUDE_OPENCV)

LOCAL_SHARED_LIBRARIES =\
			$(PLIB)libopencv_calib3d.so\
			$(PLIB)libopencv_core.so\
			$(PLIB)libopencv_contrib.so \
			$(PLIB)libopencv_features2d.so \
			$(PLIB)libopencv_flann.so \
			$(PLIB)libopencv_highgui.so\
			$(PLIB)libopencv_imgproc.so \
			$(PLIB)libopencv_legacy.so\
			$(PLIB)libopencv_ml.so \
			$(PLIB)libopencv_nonfree.so\
			$(PLIB)libopencv_objdetect.so \
			$(PLIB)libopencv_photo.so \
			$(PLIB)libopencv_stitching.so \
			$(PLIB)libopencv_video.so \
			$(PLIB)libopencv_videostab.so

LDFLAGS += -L$(LOCAL_SHARED_LIBRARIES)

# source file(s), including c file(s) cpp file(s)
# you can also use $(wildcard *.c), etc.
SRC_C   := $(wildcard *.c)
SRC_CPP := $(wildcard *.cpp)

# object file(s)
OBJ_C   := $(patsubst %.c,%.o,$(SRC_C))
OBJ_CPP := $(patsubst %.cpp,%.o,$(SRC_CPP))

GTK_FLAGS = `pkg-config --cflags --libs`


# executable file
ifeq ($(testmode), t)
target = libvehDet.so
else
target = detect.exe
endif

###############################################################################

all: $(target)

$(target): $(OBJ_C) $(OBJ_CPP)
	@echo "Generating executable file..." $(notdir $(target))
	@$(CXX) $(CFLAGS) $^ -o $(target) $(LDFLAGS) $(GTK_FLAGS)

# make all .c or .cpp
%.o: %.c
	@echo "Compling: " $(addsuffix .c, $(basename $(notdir $@)))
	@$(CC) $(CFLAGS) -c $< -o $@ $(GTK_FLAGS)

%.o: %.cpp
	@echo "Compling: " $(addsuffix .cpp, $(basename $(notdir $@)))
	@$(CXX) $(CFLAGS) -c $< -o $@ $(GTK_FLAGS)

clean:
	@echo "cleaning..."
	@$(RM) $(target)
	@$(RM) *.o *.back *.exe *.so *.a *~

.PHONY: all clean
