#pragma once
#include "ramulator/dram/commands/ACT2.h"
#include "ramulator/dram/node.h"
namespace Ramulator::Cmd {
template<class T> struct ACTAB {
  static constexpr DRAMCommandMeta meta = {.is_opening = true};
  static constexpr BankTarget bank_target = BankTarget::All;
  static void action(DRAMNode* bank, int cmd, const AddrVec_t& addr, Clk_t clk) {
    if (bank->m_spec->command_expansions[cmd].empty()) ACT2<T>::action(bank, cmd, addr, clk);
  }
  static int preq(DRAMNode* bank, int cmd, const AddrVec_t&, Clk_t) {
    if (bank->m_state == T::State::Closed) return cmd;
    return cmd == T::Command::ACTAB_LOCAL ? T::Command::PREab_LOCAL : T::Command::PREab;
  }
};
}
