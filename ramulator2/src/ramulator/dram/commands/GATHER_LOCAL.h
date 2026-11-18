#pragma once
#include "ramulator/dram/commands/GATHER.h"
namespace Ramulator::Cmd {
template<class T> struct GATHER_LOCAL : GATHER<T> {};
}
