cmake_minimum_required(VERSION 3.18)
project(pim_vla_sim2 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(RAMULATOR_PYTHON_BINDINGS OFF CACHE BOOL "" FORCE)
add_subdirectory(ramulator2)
add_executable(pim_vla_sim
  ramulator2/src/ramulator/controller/impl/pim_controller.cpp
  ramulator2/src/ramulator/controller/impl/pim_lookahead.cpp
  ramulator2/src/ramulator/plugin/bus_checker.cpp
  ramulator2/src/ramulator/frontend/impl/json_input.cpp
  ramulator2/src/ramulator/frontend/impl/fused_kernel_frontend.cpp
  ramulator2/src/ramulator/frontend/impl/pim_vla_frontend.cpp)
target_include_directories(pim_vla_sim PRIVATE ramulator2/src)
target_link_libraries(pim_vla_sim PRIVATE ramulator)
target_compile_options(pim_vla_sim PRIVATE -Wall -Wextra -O3)
add_executable(native_spec_tests tests/native_spec_tests.cpp)
target_link_libraries(native_spec_tests PRIVATE ramulator)
add_executable(native_microbench tests/native_microbench.cpp
  ramulator2/src/ramulator/controller/impl/pim_controller.cpp
  ramulator2/src/ramulator/controller/impl/pim_lookahead.cpp
  ramulator2/src/ramulator/plugin/bus_checker.cpp)
target_link_libraries(native_microbench PRIVATE ramulator)
add_executable(test_pim_resources tests/test_pim_resources.cpp)
target_include_directories(test_pim_resources PRIVATE ramulator2/src)
target_compile_options(test_pim_resources PRIVATE -UNDEBUG -Wall -Wextra -Werror)
add_executable(native_lookahead_tests tests/native_lookahead_tests.cpp
  ramulator2/src/ramulator/controller/impl/pim_lookahead.cpp)
target_link_libraries(native_lookahead_tests PRIVATE ramulator)
add_executable(benchmark_program_input tools/benchmark_program_input.cpp
  ramulator2/src/ramulator/frontend/impl/json_input.cpp)
target_link_libraries(benchmark_program_input PRIVATE ramulator)
add_executable(native_sequencer_tests tests/native_sequencer_tests.cpp
  ramulator2/src/ramulator/controller/impl/pim_controller.cpp
  ramulator2/src/ramulator/controller/impl/pim_lookahead.cpp
  ramulator2/src/ramulator/plugin/bus_checker.cpp)
target_link_libraries(native_sequencer_tests PRIVATE ramulator)
target_compile_options(native_sequencer_tests PRIVATE -Wall -Wextra -O3)
add_executable(native_refresh_tests tests/native_refresh_tests.cpp
  ramulator2/src/ramulator/controller/impl/pim_controller.cpp
  ramulator2/src/ramulator/controller/impl/pim_lookahead.cpp
  ramulator2/src/ramulator/plugin/bus_checker.cpp)
target_link_libraries(native_refresh_tests PRIVATE ramulator)
target_compile_options(native_refresh_tests PRIVATE -Wall -Wextra -O3)
