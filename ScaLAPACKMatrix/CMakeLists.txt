SET(ScaLAPACKMatrix_src
    ${CMAKE_CURRENT_SOURCE_DIR}/source/ProcessGrid.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/ScaLAPACKMat.cpp)

ADD_LIBRARY(${TARGETLIB_ScaLAPACKMatrix} SHARED ${ScaLAPACKMatrix_src})

target_include_directories(
    ${TARGETLIB_ScaLAPACKMatrix}
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

DEAL_II_SETUP_TARGET( ${TARGETLIB_ScaLAPACKMatrix} )

# TARGET_LINK_LIBRARIES( ${TARGETLIB_ScaLAPACKMatrix} ${Boost_LIBRARIES} )