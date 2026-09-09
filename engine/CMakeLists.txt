cmake_minimum_required(VERSION 3.16)
project(chlorine LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(chlorine
  src/main.cpp
  src/hgn.cpp
  src/serve.cpp)
target_compile_options(chlorine PRIVATE -Wall -Wextra -O2)

# HIP trunk backend (engine/src/generator.hip). Built with hipcc when available;
# without it the engine falls back to the stub generator (wire protocol only).
find_program(HIPCC_EXECUTABLE hipcc)
if(HIPCC_EXECUTABLE)
  find_library(HIP_RUNTIME amdhip64 HINTS /opt/rocm/lib /opt/rocm/lib64)
  execute_process(COMMAND ${HIPCC_EXECUTABLE} --rocm-path
                  OUTPUT_VARIABLE ROCM_PATH OUTPUT_STRIP_TRAILING_WHITESPACE
                  ERROR_QUIET)
  find_library(OMP_RUNTIME omp
               HINTS ${ROCM_PATH}/lib/llvm/lib ${ROCM_PATH}/lib /opt/rocm/lib/llvm/lib)
  if(NOT OMP_RUNTIME)
    find_library(OMP_RUNTIME gomp)
  endif()
  set(GEN_OBJ ${CMAKE_CURRENT_BINARY_DIR}/generator.hip.o)
  add_custom_command(OUTPUT ${GEN_OBJ}
    COMMAND ${HIPCC_EXECUTABLE} --offload-arch=gfx1151 -O3 -fopenmp -fPIC -c
            ${CMAKE_CURRENT_SOURCE_DIR}/src/generator.hip -o ${GEN_OBJ}
    DEPENDS src/generator.hip
    COMMENT "hipcc generator.hip -> gfx1151")
  target_sources(chlorine PRIVATE ${GEN_OBJ})
  if(HIP_RUNTIME)
    target_link_libraries(chlorine PRIVATE ${HIP_RUNTIME})
  endif()
  if(OMP_RUNTIME)
    target_link_libraries(chlorine PRIVATE ${OMP_RUNTIME})
  endif()
  target_compile_definitions(chlorine PRIVATE CHLORINE_HAS_TRUNK=1)
endif()
