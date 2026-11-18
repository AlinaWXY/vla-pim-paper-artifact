#pragma once
#include "ramulator/dram/commands/PIMCommandsCommon.h"
namespace Ramulator::Cmd {
template<class T> struct MACQK_M1_LOCAL : PIMArrayReadAll<T, true> {};
}
