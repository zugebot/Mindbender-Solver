#pragma once
// src/solver/inram/right_frontier_table.hpp

#include "src/solver/inram/right_frontier_index.hpp"

// The production solver has one right-frontier implementation: a full-state,
// exact index.  Experimental fingerprint/hash-only tables were removed because
// they are not valid correctness gates for all-solution enumeration.
using RightFrontierTableB1B2 = RightFrontierIndexB1B2;
