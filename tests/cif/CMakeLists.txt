# CIF (Context Information Field) Unit Tests
# Organized by functional area

# Field Proxy Interface Tests
vrtigo_add_gtest(proxy_test proxy_test.cpp NAME cif_proxy_test)

# Interpreted Value Tests (Q52.12 ↔ Hz conversion)
# Tests bandwidth and sample_rate fields
vrtigo_add_gtest(value_test value_test.cpp NAME cif_value_test)

# CIF1 Field Tests
vrtigo_add_gtest(cif1_test cif1_test.cpp)

# CIF2 Field Tests
vrtigo_add_gtest(cif2_test cif2_test.cpp)

# CIF3 Field Tests
vrtigo_add_gtest(cif3_test cif3_test.cpp)
