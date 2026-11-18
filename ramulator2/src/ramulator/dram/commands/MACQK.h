#pragma once
#include "ramulator/dram/commands/PIMCommandsCommon.h"
namespace Ramulator::Cmd {
template<class T> struct MACQK : PIMArrayReadAll<T> {};
}
