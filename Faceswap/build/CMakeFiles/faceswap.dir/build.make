# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/hkhpysddd/faceswap

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/hkhpysddd/faceswap/build

# Include any dependencies generated for this target.
include CMakeFiles/faceswap.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/faceswap.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/faceswap.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/faceswap.dir/flags.make

CMakeFiles/faceswap.dir/faceswap.cpp.o: CMakeFiles/faceswap.dir/flags.make
CMakeFiles/faceswap.dir/faceswap.cpp.o: ../faceswap.cpp
CMakeFiles/faceswap.dir/faceswap.cpp.o: CMakeFiles/faceswap.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/hkhpysddd/faceswap/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/faceswap.dir/faceswap.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/faceswap.dir/faceswap.cpp.o -MF CMakeFiles/faceswap.dir/faceswap.cpp.o.d -o CMakeFiles/faceswap.dir/faceswap.cpp.o -c /home/hkhpysddd/faceswap/faceswap.cpp

CMakeFiles/faceswap.dir/faceswap.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/faceswap.dir/faceswap.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/hkhpysddd/faceswap/faceswap.cpp > CMakeFiles/faceswap.dir/faceswap.cpp.i

CMakeFiles/faceswap.dir/faceswap.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/faceswap.dir/faceswap.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/hkhpysddd/faceswap/faceswap.cpp -o CMakeFiles/faceswap.dir/faceswap.cpp.s

# Object files for target faceswap
faceswap_OBJECTS = \
"CMakeFiles/faceswap.dir/faceswap.cpp.o"

# External object files for target faceswap
faceswap_EXTERNAL_OBJECTS =

faceswap: CMakeFiles/faceswap.dir/faceswap.cpp.o
faceswap: CMakeFiles/faceswap.dir/build.make
faceswap: /usr/local/lib/libopencv_stitching.so.3.4.13
faceswap: /usr/local/lib/libopencv_superres.so.3.4.13
faceswap: /usr/local/lib/libopencv_videostab.so.3.4.13
faceswap: /usr/local/lib/libopencv_aruco.so.3.4.13
faceswap: /usr/local/lib/libopencv_bgsegm.so.3.4.13
faceswap: /usr/local/lib/libopencv_bioinspired.so.3.4.13
faceswap: /usr/local/lib/libopencv_ccalib.so.3.4.13
faceswap: /usr/local/lib/libopencv_dnn_objdetect.so.3.4.13
faceswap: /usr/local/lib/libopencv_dpm.so.3.4.13
faceswap: /usr/local/lib/libopencv_face.so.3.4.13
faceswap: /usr/local/lib/libopencv_freetype.so.3.4.13
faceswap: /usr/local/lib/libopencv_fuzzy.so.3.4.13
faceswap: /usr/local/lib/libopencv_hfs.so.3.4.13
faceswap: /usr/local/lib/libopencv_img_hash.so.3.4.13
faceswap: /usr/local/lib/libopencv_line_descriptor.so.3.4.13
faceswap: /usr/local/lib/libopencv_optflow.so.3.4.13
faceswap: /usr/local/lib/libopencv_reg.so.3.4.13
faceswap: /usr/local/lib/libopencv_rgbd.so.3.4.13
faceswap: /usr/local/lib/libopencv_saliency.so.3.4.13
faceswap: /usr/local/lib/libopencv_stereo.so.3.4.13
faceswap: /usr/local/lib/libopencv_structured_light.so.3.4.13
faceswap: /usr/local/lib/libopencv_surface_matching.so.3.4.13
faceswap: /usr/local/lib/libopencv_tracking.so.3.4.13
faceswap: /usr/local/lib/libopencv_xfeatures2d.so.3.4.13
faceswap: /usr/local/lib/libopencv_ximgproc.so.3.4.13
faceswap: /usr/local/lib/libopencv_xobjdetect.so.3.4.13
faceswap: /usr/local/lib/libopencv_xphoto.so.3.4.13
faceswap: /usr/local/lib/libopencv_shape.so.3.4.13
faceswap: /usr/local/lib/libopencv_highgui.so.3.4.13
faceswap: /usr/local/lib/libopencv_videoio.so.3.4.13
faceswap: /usr/local/lib/libopencv_phase_unwrapping.so.3.4.13
faceswap: /usr/local/lib/libopencv_video.so.3.4.13
faceswap: /usr/local/lib/libopencv_datasets.so.3.4.13
faceswap: /usr/local/lib/libopencv_plot.so.3.4.13
faceswap: /usr/local/lib/libopencv_text.so.3.4.13
faceswap: /usr/local/lib/libopencv_dnn.so.3.4.13
faceswap: /usr/local/lib/libopencv_ml.so.3.4.13
faceswap: /usr/local/lib/libopencv_imgcodecs.so.3.4.13
faceswap: /usr/local/lib/libopencv_objdetect.so.3.4.13
faceswap: /usr/local/lib/libopencv_calib3d.so.3.4.13
faceswap: /usr/local/lib/libopencv_features2d.so.3.4.13
faceswap: /usr/local/lib/libopencv_flann.so.3.4.13
faceswap: /usr/local/lib/libopencv_photo.so.3.4.13
faceswap: /usr/local/lib/libopencv_imgproc.so.3.4.13
faceswap: /usr/local/lib/libopencv_core.so.3.4.13
faceswap: CMakeFiles/faceswap.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/hkhpysddd/faceswap/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable faceswap"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/faceswap.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/faceswap.dir/build: faceswap
.PHONY : CMakeFiles/faceswap.dir/build

CMakeFiles/faceswap.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/faceswap.dir/cmake_clean.cmake
.PHONY : CMakeFiles/faceswap.dir/clean

CMakeFiles/faceswap.dir/depend:
	cd /home/hkhpysddd/faceswap/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/hkhpysddd/faceswap /home/hkhpysddd/faceswap /home/hkhpysddd/faceswap/build /home/hkhpysddd/faceswap/build /home/hkhpysddd/faceswap/build/CMakeFiles/faceswap.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/faceswap.dir/depend

