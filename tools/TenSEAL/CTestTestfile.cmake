# CMake generated Testfile for 
# Source directory: /home/mlonfils/Documents/Doctorat/fork/TenSEAL
# Build directory: /home/mlonfils/Documents/Doctorat/fork/TenSEAL
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(tenseal_tests "tenseal_tests")
set_tests_properties(tenseal_tests PROPERTIES  _BACKTRACE_TRIPLES "/home/mlonfils/Documents/Doctorat/fork/TenSEAL/cmake/tests.cmake;25;add_test;/home/mlonfils/Documents/Doctorat/fork/TenSEAL/cmake/tests.cmake;0;;/home/mlonfils/Documents/Doctorat/fork/TenSEAL/CMakeLists.txt;22;include;/home/mlonfils/Documents/Doctorat/fork/TenSEAL/CMakeLists.txt;0;")
subdirs("_deps/com_microsoft_seal-build")
subdirs("_deps/com_pybind_pybind11-build")
subdirs("_deps/com_nlohmann_json-build")
subdirs("_deps/com_xtensorstack_xtl-build")
subdirs("_deps/com_xtensorstack_xsimd-build")
subdirs("_deps/com_xtensorstack_xtensor-build")
subdirs("_deps/googletest-build")
