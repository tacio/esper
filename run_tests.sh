#!/usr/bin/env bash
set -e

echo "Starting Esper Test Suite..."

echo "Running Arena Allocator Tests..."
mojo run tests/test_arena.mojo

echo "All tests passed successfully."
