add_executable(NNLS NNLS.cpp)

target_link_libraries(NNLS ${TARGETLIB_FortranRoutines} ${TARGETLIB_ReadWrite})

DEAL_II_SETUP_TARGET(NNLS)