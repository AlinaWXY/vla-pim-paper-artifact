#pragma once
#include "ramulator/dram/commands/ROPE_EXEC.h"
namespace Ramulator::Cmd {
template<class T> struct ROPE_EXEC_LOCAL : ROPE_EXEC<T> {};
}
