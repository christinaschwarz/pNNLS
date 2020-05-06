SET(ReadWrite_src
    ${CMAKE_CURRENT_SOURCE_DIR}/source/ReadWrite.cpp)

add_library( ${TARGETLIB_ReadWrite} SHARED ${ReadWrite_src} )

target_include_directories(
    ${TARGETLIB_ReadWrite}
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

DEAL_II_SETUP_TARGET( ${TARGETLIB_ReadWrite} )

# TARGET_LINK_LIBRARIES( ${TARGETLIB_ReadWrite} ${Boost_LIBRARIES} )