#pragma once

#include <array>

#include <cstring>
#include <gtest/gtest.h>
#include <vrtigo.hpp>

using namespace vrtigo;

// Shared test fixture for all context packet tests
class ContextPacketTest : public ::testing::Test {
protected:
    // Buffer for testing
    alignas(4) std::array<uint8_t, 4096> buffer{};

    void SetUp() override { std::memset(buffer.data(), 0, buffer.size()); }
};
