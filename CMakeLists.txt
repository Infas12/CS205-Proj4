cmake_minimum_required(VERSION 3.14)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_definitions(-DWITH_AVX2)
set(CMAKE_C_FLAGS "-mavx -fopenmp")
set(CMAKE_CXX_FLAGS "-mavx -fopenmp")

# set the project name
project(MatMul)

# source
aux_source_directory (src DIR_SRCS)

# include
include_directories(inc)




add_custom_target(debug
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Debug ${CMAKE_SOURCE_DIR}
  COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target all
  COMMENT "Switch CMAKE_BUILD_TYPE to Debug"
  )

add_custom_target(release
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Release ${CMAKE_SOURCE_DIR}
  COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target all
  COMMENT "Switch CMAKE_BUILD_TYPE to Release"
  )


find_package(OpenMP)

find_package(OpenBLAS)



# add the executable
add_executable(MatMul main.c ${DIR_SRCS})

target_link_libraries(MatMul PUBLIC openblas)