# CIF (Context Information Field) Unit Tests
# Organized by functional area

# Field Proxy Interface Tests
add_executable(proxy_test proxy_test.cpp)
target_link_libraries(proxy_test PRIVATE vrtio GTest::gtest_main)
add_test(NAME cif_proxy_test COMMAND proxy_test)

# Interpreted Value Tests (Q52.12 ↔ Hz conversion)
# Tests bandwidth and sample_rate fields
add_executable(value_test value_test.cpp)
target_link_libraries(value_test PRIVATE vrtio GTest::gtest_main)
add_test(NAME cif_value_test COMMAND value_test)

# CIF1 Field Tests
add_executable(cif1_test cif1_test.cpp)
target_link_libraries(cif1_test PRIVATE vrtio GTest::gtest_main)
add_test(NAME cif1_test COMMAND cif1_test)

# CIF2 Field Tests
add_executable(cif2_test cif2_test.cpp)
target_link_libraries(cif2_test PRIVATE vrtio GTest::gtest_main)
add_test(NAME cif2_test COMMAND cif2_test)

# CIF3 Field Tests
add_executable(cif3_test cif3_test.cpp)
target_link_libraries(cif3_test PRIVATE vrtio GTest::gtest_main)
add_test(NAME cif3_test COMMAND cif3_test)
