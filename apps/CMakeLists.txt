add_executable(raster_to_svg raster_to_svg.cpp)
target_link_libraries(raster_to_svg PRIVATE neroued::vectorizer)
install(TARGETS raster_to_svg RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

if(NV_BUILD_EVAL)
    add_executable(evaluate_svg evaluate_svg.cpp)
    target_link_libraries(evaluate_svg PRIVATE neroued::vectorizer_eval)
    install(TARGETS evaluate_svg RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
