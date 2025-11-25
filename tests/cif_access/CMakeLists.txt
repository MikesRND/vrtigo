# CIF Access Pattern Tests
# Self-documenting examples that generate markdown documentation
# Each test file creates a corresponding page in docs/cif_access/
#
# Naming convention: cif{N}_{bit}_{fieldname}_test.cpp
# Example: cif0_15_payload_format_test.cpp → docs/cif_access/cif0_15_payload_format.md

# CIF0 bit 15: Data Packet Payload Format
vrtigo_add_gtest(cif0_15_payload_format_test cif0_15_payload_format_test.cpp)

# CIF0 bit 17: Device Identifier
vrtigo_add_gtest(cif0_17_device_id_test cif0_17_device_id_test.cpp)

# CIF0 bit 21: Sample Rate
vrtigo_add_gtest(cif0_21_sample_rate_test cif0_21_sample_rate_test.cpp)

# CIF0 bit 22: Over-Range Count
vrtigo_add_gtest(cif0_22_over_range_count_test cif0_22_over_range_count_test.cpp)

# CIF0 bit 23: Gain
vrtigo_add_gtest(cif0_23_gain_test cif0_23_gain_test.cpp)

# CIF0 bit 24: Reference Level
vrtigo_add_gtest(cif0_24_reference_level_test cif0_24_reference_level_test.cpp)

# CIF0 bit 25: IF Band Offset
vrtigo_add_gtest(cif0_25_if_band_offset_test cif0_25_if_band_offset_test.cpp)

# CIF0 bit 26: RF Frequency Offset
vrtigo_add_gtest(cif0_26_rf_frequency_offset_test cif0_26_rf_frequency_offset_test.cpp)

# CIF0 bit 27: RF Reference Frequency
vrtigo_add_gtest(cif0_27_rf_reference_frequency_test cif0_27_rf_reference_frequency_test.cpp)

# CIF0 bit 28: IF Reference Frequency
vrtigo_add_gtest(cif0_28_if_reference_frequency_test cif0_28_if_reference_frequency_test.cpp)

# CIF0 bit 29: Bandwidth
vrtigo_add_gtest(cif0_29_bandwidth_test cif0_29_bandwidth_test.cpp)

# CIF0 bit 30: Reference Point Identifier
vrtigo_add_gtest(cif0_30_reference_point_id_test cif0_30_reference_point_id_test.cpp)

# CIF0 bit 31: Context Field Change Indicator
vrtigo_add_gtest(cif0_31_change_indicator_test cif0_31_change_indicator_test.cpp)
