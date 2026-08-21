set(TEST_NAMES
    test_event_sim
    # test_event_queue
    # test_event_queue_sort
    # test_event_queue_sort_thrust
    # test_event_queue_sort_thrust_cuda
    # test_event_queue_sort_thrust_hip
)

foreach(test ${TEST_NAMES})
    add_executable(${test} test_main.cpp ${test}.cpp)
    target_compile_features(${test} PRIVATE cxx_std_17)
    target_link_libraries(${test} PRIVATE event-sim-core Catch2::Catch2)

    catch_discover_tests(${test}
        PROPERTIES
        SKIP_RETURN_CODE 4
        TEST_PREFIX "${test}::"
        EXTRA_ARGS "--reporter;console;--out=/dev/stdout")
endforeach()

set(
TEST_FILES
pwr_pincell.h5m
)

foreach(file ${TEST_FILES})
    ADD_CUSTOM_TARGET(${file} ALL
                      COMMAND ${CMAKE_COMMAND} -E create_symlink
                      "${CMAKE_CURRENT_SOURCE_DIR}/test_files/${file}"
                      "${CMAKE_CURRENT_BINARY_DIR}/${file}")
endforeach()
