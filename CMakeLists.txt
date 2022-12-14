cmake_minimum_required(VERSION 3.12)

enable_language(CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -Wall -pedantic -Wextra -Wnon-virtual-dtor -Wshadow")

include(FetchContent)

FetchContent_Declare(fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt.git
  GIT_TAG master
)

FetchContent_MakeAvailable(fmt)

project(voronoi)

add_executable(${PROJECT_NAME} src/main.cpp)

find_package(Threads)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)
endif()


target_link_libraries(${PROJECT_NAME} 
  PRIVATE fmt::fmt
  PRIVATE ${CMAKE_THREAD_LIBS_INIT}
)
