#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
#include "ramulator/base/type.h"

namespace Ramulator { class ConfigNode; class DRAMDevice; }
namespace PimVla {
struct Operation;
struct ForecastEntry {
  int cmd=-1;
  Ramulator::AddrVec_t addr;
  int64_t at=0;
};

// A bounded conservative scheduling prediction, never a legality oracle.
// All commands are issued only into a private native DRAMDevice snapshot.
class NativeLookahead {
 public:
  NativeLookahead(int channel,const Ramulator::ConfigNode& spec_cfg);
  ~NativeLookahead();
  NativeLookahead(NativeLookahead&&) noexcept;
  NativeLookahead& operator=(NativeLookahead&&) noexcept;
  NativeLookahead(const NativeLookahead&)=delete;
  NativeLookahead& operator=(const NativeLookahead&)=delete;
  std::vector<ForecastEntry> predict(const Ramulator::DRAMDevice& live,
      const std::vector<Operation>& ops,size_t next,const std::vector<int64_t>& done,
      int64_t now,int64_t horizon,bool local,int row_offset=0);
 private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};
}
