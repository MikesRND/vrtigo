#pragma once

namespace vrtigo::detail {

/**
 * Tag type for identifying bitfields within a layout.
 *
 * This is distinct from field_tag which identifies CIF fields.
 * bitfield_tag is used for accessing individual bitfields within
 * complex structured CIF fields.
 *
 * @tparam FieldType The BitField type this tag represents
 */
template <typename FieldType>
struct bitfield_tag {
    using type = FieldType;
    constexpr bitfield_tag() noexcept = default;
};

} // namespace vrtigo::detail
