#pragma once
#include "ramulator/dram/commands/ACT2.h"
namespace Ramulator::Cmd {
// Internal-only primitive; no CA bus occupancy or separately issued request.
template<class T> struct ACT2_PULSE : ACT2<T> {};
}
