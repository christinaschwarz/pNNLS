add_executable(pNNLS pNNLS.cpp)

target_link_libraries(pNNLS ${TARGETLIB_ScaLAPACKMatrix} ${TARGETLIB_ReadWrite} ${Boost_LIBRARIES})

DEAL_II_SETUP_TARGET(pNNLS)