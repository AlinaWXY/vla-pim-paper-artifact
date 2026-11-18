#pragma once
#include "ramulator/dram/commands/PIMCommandsCommon.h"
namespace Ramulator::Cmd {
template<class T> struct MACQK_LOCAL : PIMArrayReadAll<T, true> {};
}
