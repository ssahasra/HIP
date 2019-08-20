hostcall_add_test(testname accumulator ${CMAKE_CURRENT_LIST_DIR}/one-workitem.cpp)
set_tests_properties(${testname} PROPERTIES PASS_REGULAR_EXPRESSION
    "Lorem ipsum dolor sit amet, consectetur nullam\\. In mollis imperdiet nibh nec ullamcorper\\.
Carpe diem\\.
Integer: 42\\.
Begin\\. Pointer: 0xffffffffffffffd5\\. Middle\\. Floating point: 4\\.200000\\. End\\.
PASSED\\!
")

hostcall_build_test(testname accumulator ${CMAKE_CURRENT_LIST_DIR}/one-wave.cpp)
add_test(NAME ${testname}
    COMMAND ${CMAKE_BINARY_DIR}/tests/hostcall/sort-uniq.sh ${CMAKE_CURRENT_BINARY_DIR}/${testname})
set_tests_properties(${testname} PROPERTIES PASS_REGULAR_EXPRESSION
    "    130 Carpe diem\\.
    170 Curabitur nec metus sit amet augue vehicula ultrices ut id leo\\. Lorem ipsum dolor sit amet, consectetur adipiscing elit amet\\.
    170 Lorem ipsum dolor sit amet, consectetur nullam\\. In mollis imperdiet nibh nec ullamcorper\\.
      1 PASSED\\!
")

hostcall_build_test(testname accumulator ${CMAKE_CURRENT_LIST_DIR}/many-waves.cpp)
add_test(NAME ${testname}
    COMMAND ${CMAKE_BINARY_DIR}/tests/hostcall/sort-uniq.sh ${CMAKE_CURRENT_BINARY_DIR}/${testname})
set_tests_properties(${testname} PROPERTIES PASS_REGULAR_EXPRESSION
    "  11667 Carpe diem\\.
   8166 Curabitur nec metus sit amet augue vehicula ultrices ut id leo\\. Lorem ipsum dolor sit amet, consectetur adipiscing elit amet\\.
   8167 Lorem ipsum dolor sit amet, consectetur nullam\\. In mollis imperdiet nibh nec ullamcorper\\.
      1 PASSED\\!
")
