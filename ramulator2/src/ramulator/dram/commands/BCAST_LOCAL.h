#pragma once
#include "ramulator/dram/commands/BCAST.h"
namespace Ramulator::Cmd {
template<class T> struct BCAST_LOCAL : BCAST<T> {};
}
