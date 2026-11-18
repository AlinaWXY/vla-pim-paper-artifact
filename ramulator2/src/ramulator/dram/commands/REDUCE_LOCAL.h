#pragma once
#include "ramulator/dram/commands/REDUCE.h"
namespace Ramulator::Cmd {
template<class T> struct REDUCE_LOCAL : REDUCE<T> {};
}
