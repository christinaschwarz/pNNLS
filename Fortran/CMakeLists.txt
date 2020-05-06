enable_language(Fortran)

get_filename_component(Fortran_COMPILER_NAME ${CMAKE_Fortran_COMPILER} NAME) 

message("Fortran Compiler: ${CMAKE_Fortran_COMPILER}")
message("Fortran Compiler ID: ${CMAKE_Fortran_COMPILER_ID}\n\n")

if(CMAKE_Fortran_COMPILER_ID MATCHES Intel)
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -Wall")
    set(CMAKE_Fortran_FLAGS_DEBUG "-g -traceback")
    set(CMAKE_Fortran_FLAGS_RELEASE "-O3 -ip -xHOST")
elseif(CMAKE_Fortran_COMPILER_ID MATCHES GNU)
    set(CMAKE_Fortran_FLAGS         "${CMAKE_Fortran_FLAGS} -Wall")
    set(CMAKE_Fortran_FLAGS_DEBUG   "-fno-f2c -O0 -g")
    set(CMAKE_Fortran_FLAGS_RELEASE "-funroll-all-loops -fno-f2c -O3")
endif(CMAKE_Fortran_COMPILER_ID MATCHES Intel)

## Declare all source files the target ExtFortranRoutines consists of:
SET(TARGET_FortranRoutines_src
    ${CMAKE_CURRENT_SOURCE_DIR}/source/lawson_hanson_nnls.f
    ${CMAKE_CURRENT_SOURCE_DIR}/source/nnnpls.f
    ${CMAKE_CURRENT_SOURCE_DIR}/source/FortranRoutines.cpp)
    
ADD_LIBRARY(${TARGETLIB_FortranRoutines} SHARED ${TARGET_FortranRoutines_src})

target_include_directories(
    ${TARGETLIB_FortranRoutines}
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

DEAL_II_SETUP_TARGET(${TARGETLIB_FortranRoutines})


