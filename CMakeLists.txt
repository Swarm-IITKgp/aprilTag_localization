cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

# Find the Ignition_Transport library
find_package(ignition-transport3 QUIET REQUIRED)
find_package(OpenCV REQUIRED)
pkg_check_modules(AprilTag apriltag)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${IGNITION-TRANSPORT_CXX_FLAGS}")
include_directories(${IGNITION-TRANSPORT_INCLUDE_DIRS} ${AprilTag_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${AprilTag_INCLUDE_DIRS}/apriltag)
link_directories(${IGNITION-TRANSPORT_LIBRARY_DIRS} ${AprilTag_LIBRARY_DIRS})

# Generate examples
if (EXISTS "${CMAKE_SOURCE_DIR}/posePub.cc")
  add_executable(posePub posePub.cc)
  target_link_libraries(posePub ${IGNITION-TRANSPORT_LIBRARIES} ${OpenCV_LIBS} ${AprilTag_LIBRARIES})
endif()
