#pragma once
#include "ramulator/dram/commands/PIMCommandsCommon.h"
namespace Ramulator::Cmd {
template<class T> struct PREpb_LOCAL : PREpb<T> {
  static int preq(DRAMNode* bank, int cmd, const AddrVec_t& addr, Clk_t clk) {
    int p = PREpb<T>::preq(bank, cmd, addr, clk);
    return p == T::Command::PREpb ? T::Command::PREpb_LOCAL : p;
  }
};
}
