#pragma once
#include "ramulator/dram/commands/PIMCommandsCommon.h"
namespace Ramulator::Cmd {
template<class T> struct MACAB_M8_LOCAL : PIMArrayReadAll<T, true> {};
}
