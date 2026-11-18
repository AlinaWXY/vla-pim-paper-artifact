#pragma once
#include "ramulator/dram/commands/PIMCommandsCommon.h"
namespace Ramulator::Cmd {
template<class T> struct ACT1_LOCAL : ACT1<T> {
  static int preq(DRAMNode* bank, int cmd, const AddrVec_t& addr, Clk_t clk) {
    int p = ACT1<T>::preq(bank, cmd, addr, clk);
    if (p == T::Command::ACT1) return T::Command::ACT1_LOCAL;
    if (p == T::Command::PREpb) return T::Command::PREpb_LOCAL;
    return p;
  }
};
}
