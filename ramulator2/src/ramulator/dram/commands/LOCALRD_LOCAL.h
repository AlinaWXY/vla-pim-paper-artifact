#pragma once
#include "ramulator/dram/commands/LOCALRD.h"
namespace Ramulator::Cmd {
template<class T> struct LOCALRD_LOCAL : LOCALRD<T> {
  static int preq(DRAMNode* bank, int cmd, const AddrVec_t& addr, Clk_t clk) {
    if (bank->m_spec->bank_targets[cmd] == BankTarget::All)
      return PIMArrayReadAll<T>::preq(bank, cmd, addr, clk);
    return LOCALRD<T>::preq(bank, cmd, addr, clk);
  }
};
}
