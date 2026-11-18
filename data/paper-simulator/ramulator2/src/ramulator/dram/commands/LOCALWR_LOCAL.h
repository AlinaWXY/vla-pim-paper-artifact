#pragma once
#include "ramulator/dram/commands/LOCALWR.h"
namespace Ramulator::Cmd {
template<class T> struct LOCALWR_LOCAL : LOCALWR<T> {
  static int preq(DRAMNode* bank, int cmd, const AddrVec_t& addr, Clk_t clk) {
    if (bank->m_spec->bank_targets[cmd] == BankTarget::All)
      return PIMArrayReadAll<T>::preq(bank, cmd, addr, clk);
    return LOCALWR<T>::preq(bank, cmd, addr, clk);
  }
};
}
