file(GLOB files *.cpp)

# These two tests are invoked differently
list(REMOVE_ITEM files ${CMAKE_CURRENT_LIST_DIR}/one-wave.cpp)
list(REMOVE_ITEM files ${CMAKE_CURRENT_LIST_DIR}/many-waves.cpp)

hostcall_add_tests("${files}")

hostcall_build_test(testname ${CMAKE_CURRENT_LIST_DIR}/one-wave.cpp)
add_test(NAME ${testname}
    COMMAND ${CMAKE_BINARY_DIR}/tests/hostcall/sort-uniq.sh ${CMAKE_CURRENT_BINARY_DIR}/${testname})
set_tests_properties(${testname} PROPERTIES PASS_REGULAR_EXPRESSION
    "    130 Carpe diem\\.
    170 Curabitur nec metus sit amet augue vehicula ultrices ut id leo\\. Lorem ipsum dolor sit amet, consectetur adipiscing elit amet\\.
    170 Lorem ipsum dolor sit amet, consectetur nullam\\. In mollis imperdiet nibh nec ullamcorper\\.
      1 PASSED\\!
")

hostcall_build_test(testname ${CMAKE_CURRENT_LIST_DIR}/many-waves.cpp)
add_test(NAME ${testname}
    COMMAND ${CMAKE_BINARY_DIR}/tests/hostcall/sort-uniq.sh ${CMAKE_CURRENT_BINARY_DIR}/${testname})
set_tests_properties(${testname} PROPERTIES PASS_REGULAR_EXPRESSION
    "  11667 Carpe diem\\.
   8166 Curabitur nec metus sit amet augue vehicula ultrices ut id leo\\. Lorem ipsum dolor sit amet, consectetur adipiscing elit amet\\.
   8167 Lorem ipsum dolor sit amet, consectetur nullam\\. In mollis imperdiet nibh nec ullamcorper\\.
      1 PASSED\\!
")

set_tests_properties(hostcall-printf-alt-forms PROPERTIES
    PASS_REGULAR_EXPRESSION
    "^042
0x42
0X42
0x000042
-123\\.456000
123\\.456000
1\\.234560e\\+02
-1\\.234560E\\+02
-123\\.456
123\\.456
0x1\\.edd2f1a9fbe77p\\+6
-0X1\\.EDD2F1A9FBE77P\\+6
0x00000042
      0x00000042
0x00000042      
PASSED\\!
")

set_tests_properties(hostcall-printf-flags PROPERTIES
    PASS_REGULAR_EXPRESSION
    "^00000042
-0000042
00000042
0123\\.456
\\+0000042
-42
\\+0000042
xyzzy   
-42
00000042        
        00000042
PASSED\\!
")

set_tests_properties(hostcall-printf-specifiers PROPERTIES
    PASS_REGULAR_EXPRESSION
    "^xyzzy
%
hello % world
%s
%s0xf01dab1eca55e77e
%cxyzzy
sep
-42
42
123\\.456000
-123\\.456000
-1\\.234560e\\+02
1\\.234560E\\+02
123\\.456
-123\\.456
x

\\(nil\\)
PASSED\\!
")

set_tests_properties(hostcall-printf-star PROPERTIES
    PASS_REGULAR_EXPRESSION
    "^              42
00000042
00000042        
    123\\.45600000 hello \\* world
PASSED\\!
")

set_tests_properties(hostcall-printf-width-precision PROPERTIES
    PASS_REGULAR_EXPRESSION
    "^              42
00000042
          -00042
00000042
00000042
  1\\.23456789e\\+04
 -12345\\.67891000
       12345\\.679
-1\\.2346e\\+04
12345\\.6789
1\\.235e\\+04
12345\\.68
12345\\.7
hello
PASSED\\!
")
