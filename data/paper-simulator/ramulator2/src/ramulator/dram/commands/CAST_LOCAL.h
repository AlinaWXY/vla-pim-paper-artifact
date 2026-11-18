#pragma once
#include "ramulator/dram/commands/CAST.h"
namespace Ramulator::Cmd {
template<class T> struct CAST_LOCAL : CAST<T> {};
}
