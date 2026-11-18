#pragma once
#include "ramulator/dram/commands/PIMCommandsCommon.h"
namespace Ramulator::Cmd {
template<class T> struct MACAB_M16_LOCAL : PIMArrayReadAll<T, true> {};
}
