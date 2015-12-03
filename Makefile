CXX = g++
CXXFLAGS = -O2 -g -Wall -fmessage-length=0 
CPPFLAGS = -I/usr/local/include/eigen3

OBJS =      LKof.o

LDFLAGS = -L/usr/local/include/eigen3
LDLIBS =  -lopencv_core -lopencv_imgproc -lopencv_calib3d -lopencv_video \
          -lopencv_features2d -lopencv_ml -lopencv_highgui -lopencv_objdetect \
          -lopencv_contrib -lopencv_legacy -lopencv_gpu

TARGET =  LKof

.PHONY: all
all: $(TARGET)
$(TARGET):  $(OBJS)
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@

.PHONY: clean
clean:
	rm -f $(OBJS) $(TARGET)
