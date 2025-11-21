#pragma once

#include <type_traits>

#include <vrtigo/types.hpp>

namespace vrtigo {

// Forward declaration
template <TsiType TSI, TsfType TSF>
class TimeStamp;

/**
 * @brief Traits for extracting timestamp type information
 *
 * Primary template defaults to invalid/not a timestamp type.
 * Specializations provide TSI/TSF extraction for valid timestamp types.
 */
template <typename T>
struct TimestampTraits {
    static constexpr bool is_valid = false;
    static constexpr bool has_timestamp = false;
};

/**
 * @brief Specialization for TimeStamp types
 *
 * Extracts TSI and TSF from TimeStamp template parameters
 */
template <TsiType TSI, TsfType TSF>
struct TimestampTraits<TimeStamp<TSI, TSF>> {
    static constexpr bool is_valid = true;
    static constexpr bool has_timestamp = true;
    static constexpr TsiType tsi = TSI;
    static constexpr TsfType tsf = TSF;
    using type = TimeStamp<TSI, TSF>;
};

/**
 * @brief Specialization for NoTimeStamp marker
 */
template <>
struct TimestampTraits<NoTimeStamp> {
    static constexpr bool is_valid = true;
    static constexpr bool has_timestamp = false;
    static constexpr TsiType tsi = TsiType::none;
    static constexpr TsfType tsf = TsfType::none;
};

/**
 * @brief Concept to check if a type is a valid timestamp type
 */
template <typename T>
concept ValidTimestampType = TimestampTraits<T>::is_valid;

/**
 * @brief Concept to check if a type has actual timestamp data
 */
template <typename T>
concept HasTimestamp = TimestampTraits<T>::has_timestamp;

} // namespace vrtigo