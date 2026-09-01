set(TEST_NAMES
    test_event_sim
    test_particle_sorting
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

add_executable(test_history_sim test_main.cpp test_history_sim.cpp)
target_compile_features(test_history_sim PRIVATE cxx_std_17)
target_link_libraries(test_history_sim PRIVATE history-sim-core Catch2::Catch2)

catch_discover_tests(test_history_sim
    PROPERTIES
    TEST_PREFIX "test_history_sim::"
    EXTRA_ARGS "--reporter;console;--out=/dev/stdout")

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
