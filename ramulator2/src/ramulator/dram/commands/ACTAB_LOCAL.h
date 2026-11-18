#pragma once
#include "ramulator/dram/commands/ACTAB.h"
namespace Ramulator::Cmd {
template<class T> struct ACTAB_LOCAL : ACTAB<T> {};
}
