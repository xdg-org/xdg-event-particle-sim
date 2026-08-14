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
