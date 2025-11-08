#pragma once

#include <vrtio/core/types.hpp>
#include <type_traits>

namespace vrtio {

// Forward declaration
template<tsi_type TSI, tsf_type TSF>
class TimeStamp;

/**
 * @brief Traits for extracting timestamp type information
 *
 * Primary template defaults to invalid/not a timestamp type.
 * Specializations provide TSI/TSF extraction for valid timestamp types.
 */
template<typename T>
struct TimestampTraits {
    static constexpr bool is_valid = false;
    static constexpr bool has_timestamp = false;
};

/**
 * @brief Specialization for TimeStamp types
 *
 * Extracts TSI and TSF from TimeStamp template parameters
 */
template<tsi_type TSI, tsf_type TSF>
struct TimestampTraits<TimeStamp<TSI, TSF>> {
    static constexpr bool is_valid = true;
    static constexpr bool has_timestamp = true;
    static constexpr tsi_type tsi = TSI;
    static constexpr tsf_type tsf = TSF;
    using type = TimeStamp<TSI, TSF>;
};

/**
 * @brief Marker type for packets without timestamps
 *
 * Use this as the TimeStamp template parameter when no timestamps are needed.
 */
struct NoTimeStamp {};

/**
 * @brief Specialization for NoTimeStamp marker
 */
template<>
struct TimestampTraits<NoTimeStamp> {
    static constexpr bool is_valid = true;
    static constexpr bool has_timestamp = false;
    static constexpr tsi_type tsi = tsi_type::none;
    static constexpr tsf_type tsf = tsf_type::none;
};

/**
 * @brief Concept to check if a type is a valid timestamp type
 */
template<typename T>
concept ValidTimestampType = TimestampTraits<T>::is_valid;

/**
 * @brief Concept to check if a type has actual timestamp data
 */
template<typename T>
concept HasTimestamp = TimestampTraits<T>::has_timestamp;

}  // namespace vrtio