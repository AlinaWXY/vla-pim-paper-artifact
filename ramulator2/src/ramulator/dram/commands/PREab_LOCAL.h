#pragma once
#include "ramulator/dram/commands/PIMCommandsCommon.h"
namespace Ramulator::Cmd {
template<class T> struct PREab_LOCAL : PREab<T> {
  static int preq(DRAMNode*, int cmd, const AddrVec_t&, Clk_t) {
    return cmd;
  }
};
}
