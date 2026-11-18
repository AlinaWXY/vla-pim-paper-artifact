#ifndef RAMULATOR_PIM_COMMANDS_COMMON_H
#define RAMULATOR_PIM_COMMANDS_COMMON_H
#include "ramulator/dram/commands/RD.h"
#include "ramulator/dram/commands/WR.h"
#include "ramulator/dram/commands/ACT1.h"
#include "ramulator/dram/commands/ACT2.h"
#include "ramulator/dram/commands/PREpb.h"
#include "ramulator/dram/commands/PREab.h"
namespace Ramulator::Cmd {
// SRF broadcast/pool IO does not require or alter any open row.
template<class T> struct PIMBufferIO {
  static constexpr DRAMCommandMeta meta = {};
  static constexpr BankTarget bank_target = BankTarget::All;
};
// Full-bank MAC opens a uniform rank row through the native ACT4x4 command.
// Mixed rows require PREab before ACTAB; single-bank local loads retain RD's
// ordinary split-ACT prerequisites.
template<class T, bool Local = false> struct PIMArrayReadAll : RD<T> {
  static constexpr BankTarget bank_target = BankTarget::All;
  static int preq(DRAMNode* bank, int cmd, const AddrVec_t& addr, Clk_t clk) {
    if (RD<T>::rowhit(bank, cmd, addr, clk)) return cmd;
    auto* rank = bank;
    while (rank->m_level != T::Level::Rank) rank = rank->m_parent_node;
    bool all_closed = true;
    rank->for_each_at_level(T::Level::Bank, [&](DRAMNode* b) {
      all_closed = all_closed && b->m_state == T::State::Closed;
    });
    if (!all_closed) return Local ? T::Command::PREab_LOCAL : T::Command::PREab;
    return Local ? T::Command::ACTAB_LOCAL : T::Command::ACTAB;
  }
};
}
#endif
