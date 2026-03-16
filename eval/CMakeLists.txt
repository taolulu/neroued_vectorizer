add_library(neroued_vectorizer_eval STATIC
    src/vectorize_eval.cpp
    src/pixel_metrics.cpp
    src/edge_metrics.cpp
    src/path_metrics.cpp
    src/svg_rasterizer.cpp
    src/svg_geometry.cpp
    src/benchmark.cpp)

add_library(neroued::vectorizer_eval ALIAS neroued_vectorizer_eval)

target_include_directories(neroued_vectorizer_eval
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/../src)

target_link_libraries(neroued_vectorizer_eval
    PUBLIC neroued::vectorizer
    PRIVATE nanosvg)

install(TARGETS neroued_vectorizer_eval
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(DIRECTORY include/neroued
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.h")
