cmake_minimum_required (VERSION 3.0)
project (g2o_curve_fitting)


set (CMAKE_BUILD_TYPE "Release") # Debug, Release, RelWithDebInfo and MinSizeRel
set (CMAKE_CXX_FLAGS "-std=c++11 -O3")
# the compiler flags for compiling C++ sources. Note you can also specify switches with ADD_DEFINITIONS()

# # 添加cmake模块以使用g2o库,  FindG2O.cmake可以在下载的G2O库里找到
list (APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)
# CMAKE_MODULE_PAHT: specifying a search path for CMake modules to be loaded by the the include() or 
#					 find_package() commands before checking the default modules that come with CMake. 

find_package (G2O REQUIRED)
include_directories (
		${G2O_INCLUDE_DIRS}
		"/usr/include/eigen3")

find_package (OpenCV REQUIRED)
include_directories (${OpenCV_INCLUDE_DIRS})

add_executable (curve_fitting_g2o main.cpp)

target_link_libraries(	curve_fitting_g2o 
						${OpenCV_LIBS} 
						g2o_core 
						g2o_stuff	)