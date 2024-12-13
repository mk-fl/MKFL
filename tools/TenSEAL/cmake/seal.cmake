include(FetchContent)

set(SEAL_BUILD_DEPS ON)
set(SEAL_USE_MSGSL ON)
set(SEAL_USE_INTEL_HEXL OFF)

FetchContent_Declare(
  com_microsoft_seal
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../SEAL
)
FetchContent_MakeAvailable(com_microsoft_seal)

include_directories(${com_microsoft_seal_SOURCE_DIR}/native/src)
include_directories(${com_microsoft_seal_SOURCE_DIR}/thirdparty/msgsl-src/include/)
include_directories(${com_microsoft_seal_SOURCE_DIR}/thirdparty/hexl-src/hexl/include/)
include_directories(${com_microsoft_seal_BINARY_DIR}/native/src)
