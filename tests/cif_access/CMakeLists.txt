# CIF Access Pattern Tests
# Self-documenting examples that generate markdown documentation
# Each test file creates a corresponding page in docs/cif_access/
#
# Naming convention: cif{N}_{bit}_{fieldname}_test.cpp
# Example: cif0_15_payload_format_test.cpp → docs/cif_access/cif0_15_payload_format.md

# CIF0 bit 15: Data Packet Payload Format
vrtigo_add_gtest(cif0_15_payload_format_test cif0_15_payload_format_test.cpp)

# CIF0 bit 21: Sample Rate
vrtigo_add_gtest(cif0_21_sample_rate_test cif0_21_sample_rate_test.cpp)

# CIF0 bit 29: Bandwidth
vrtigo_add_gtest(cif0_29_bandwidth_test cif0_29_bandwidth_test.cpp)
