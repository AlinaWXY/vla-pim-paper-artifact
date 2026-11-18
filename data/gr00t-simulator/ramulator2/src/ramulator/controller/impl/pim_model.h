#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <deque>
#include <fstream>
#include <map>
#include <memory>
#include <queue>
#include <set>
#include <string>
#include <vector>
#include "ramulator/base/config.h"
#include "ramulator/dram/device.h"
#include "ramulator/controller/impl/pim_resources.h"
#include "ramulator/controller/impl/pim_power_stats.h"
#include "ramulator/controller/impl/pim_lookahead.h"

namespace PimVla {
using namespace Ramulator;
using Clock = int64_t;
constexpr Clock NEVER = INT64_MAX / 4;
inline void require(bool ok, const std::string& why) { if (!ok) throw std::runtime_error(why); }
inline void require(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
struct CommandInfo {
  int latency=0, burst=0, completion=1, external_ca=1, private_ca=0;
  char direction=0;
  std::string wire_command;
};
struct Operation {
  std::string name, epoch;
  AddrVec_t addr, spill_addr;
  int id=0,slot=0,result_slot=0,rows=1,pairs=1;
  int dest_slot=-1,source_bank=0,tensor_fragment=-1,output_columns=1,output_column=0;
  int fragment=0,output_tile=0,kv_token=0,lanes=16,elements=16,bytes=32,token=0;
  int reduction_word=16, weight_lane_offset=0;
  bool release_result=false;
  std::string dest,source;
  std::vector<ResourceRegion> source_regions;
  std::vector<int> deps;
  std::vector<int> issue_deps;
  std::vector<Operation> children;
};
struct ProgramStage { std::string name; std::array<std::vector<Operation>,16> channels; };
struct Kernel {
  int ordinal=0; std::string name,kind;
  double us=0; uint64_t reads=0,writes=0,base=0; bool proxy=true;
  uint64_t read_address_extent_sectors=0;
  Clock start=0,nominal=0,end=0;
  Clock paused=0;
  int mlp=0; // Zero selects the historical uniform release path.
  Clock memory_done=0;
};
struct BusUtilization {
  uint64_t duration=0,private_ca=0;
  std::array<uint64_t,2> ca{},dq{}; // GPU rank0 and PIM rank1.
};
struct BusStat {
  uint64_t ca=0,dq=0,read=0,write=0,commands=0,gpu_read=0,gpu_write=0,pim_read=0,pim_write=0;
  std::array<BusUtilization,2> phases{}; // Non-I/O and fused I/O.
};
struct Burst { Clock start,end; char direction; int rank,command,stage; };
class BusChecker {
 public:
  const DRAMSpec* spec=nullptr;
  const std::vector<CommandInfo>* info=nullptr;
  std::deque<Burst> bursts;
  Clock external_start=0,external_end=0,private_end=0,account_clock=0;
  Clock private_start=0;
  int account_stage=-1,external_rank=0;
  bool split_traffic=false,io_epoch=false;
  std::ostream* dq_trace=nullptr;
  std::ostream* ca_trace=nullptr;
  std::ostream* epoch_trace=nullptr;
  const std::string* trace_stage_name=nullptr;
  int trace_channel=0;
  Clock trace_epoch_start=0;
  uint64_t checked=0;
  std::map<int,BusStat> stats;
  void observe(int cmd,const AddrVec_t& addr,Clock now,int stage,int opcode=-1,uint16_t bank_mask=0);
  void advance(Clock now);
  void trace_epoch_change(Clock now,bool active);
  void set_stage(Clock now,int stage) {
    advance(now);
    if(epoch_trace)trace_epoch_change(now,false);
    account_stage=stage;io_epoch=false;
  }
  void retire(Clock now);
};
struct GpuRequest { AddrVec_t addr; int cmd; uint64_t sequence; Clock arrival; bool array_preq=false; };
struct Completion { Clock at; bool gpu; bool operator>(const Completion& b) const {return at>b.at;} };
struct PendingAct { int cmd=-1; AddrVec_t addr; Clock deadline=NEVER; };
struct Cursor {
  const std::vector<Operation>* ops=nullptr;
  size_t next=0;
  size_t decoded_until=0;
  std::vector<Clock> done;
  std::vector<uint8_t> dispatched;
  Clock tail=0;
  uint16_t issued_banks=0;
  Clock batch_tail=0;
  void reset(const std::vector<Operation>* p,Clock now) {ops=p;next=0;decoded_until=p?std::min<size_t>(32,p->size()):0;done.assign(p?p->size():0,NEVER);dispatched.assign(p?p->size():0,0);tail=now;issued_banks=0;batch_tail=0;}
  void mark(size_t index,Clock complete) {
    require(index==next&&index<decoded_until&&!dispatched[index],"instruction FIFO requires head-only dispatch");
    dispatched[index]=1;done[index]=complete;tail=std::max(tail,complete);
    // Retire the queue entry on issue, not completion: dependencies and native
    // resource/timing readiness govern overlap with the following instruction.
    ++next;
    decoded_until=std::min(ops->size(),next+32);
  }
  bool issued() const {return !ops || next==ops->size();}
};
struct Candidate {
  int cmd=-1; AddrVec_t addr; Clock at=NEVER;
  Clock arrival=0; bool dependency_ready=false;
  enum Source {GPU,PIM,REFRESH,ACT2} source=GPU;
  size_t index=0; bool final=false,local=false,rowhit=false;
  int opcode=-1;
  size_t group=1;
  uint16_t bank_mask=0, pending_banks=0;
};
class PIMController {
 public:
  DRAMDevice device;
  std::vector<CommandInfo> info;
  BusChecker checker;
  std::unique_ptr<DramPowerStats> power_stats;
  PimResources resources;
  std::unique_ptr<NativeLookahead> lookahead;
  std::vector<ForecastEntry> forecast;
  bool forecast_dirty=true;
  Clock forecast_time=-1;
  int channel=0,stage=-1,window=64,write_high=48,write_low=16;
  bool fused=false,write_drain=false,epoch_owned=false,gap_write_phase=false;
  bool exclusive_q_exchange=false;
  bool legacy_gap_batch=false; // Keep uniform replays byte-identical.
  bool mc_control=false, pim_mode=false, srf_write_selected=false;
  bool batch_array_commands=false;
  int srf_select_cmd=-1;
  int all_bank_row=-1;
  AddrVec_t broadcast_addr;
  std::set<std::pair<int,int>> input_rows;
  Clock export_store_done=-1;
  int row_offset=0;
  std::string policy="frfcfs";
  Clock now=0,private_free=0,external_free=0,gpu_dq_end=0,pim_dq_end=0,last_progress=0,pim_arrival=0;
  std::array<Clock,2> refresh_due{};
  std::array<int,2> refresh_bank{};
  std::array<PendingAct,2> pending_act;
  // Fixed hardware policy: GPU rank0 uses paired REFpb; PIM rank1 uses REFab.
  static constexpr std::array<bool,2> refresh_all_bank{false,true};
  std::array<std::array<uint64_t,2>,2> refresh_counts{}; // [rank][REFpb,REFab]
  // Heads of per-bank/direction queues are sufficient for the monotonic
  // per-kernel tensor regions; request arrival order is retained within each.
  std::array<std::deque<GpuRequest>,32> gpu_queues;
  std::priority_queue<Completion,std::vector<Completion>,std::greater<Completion>> completions;
  int gpu_outstanding=0,queued_writes=0,queued_gpu=0;
  const Kernel* kernel=nullptr;
  Clock kernel_memory_done=0;
  uint64_t kernel_write_base=0;
  uint64_t kernel_total=0,kernel_next=0,kernel_reads=0,kernel_writes=0,issued_reads=0,issued_writes=0;
  Cursor top,inner;
  // -1 retains request-level scheduling; 0 uploads/launches, 1 computes
  // privately, and 2 exports after the current calibrated GPU kernel joins.
  int coarse_phase=-1,coarse_descriptor=-1;
  int descriptor_index=-1;
  Clock descriptor_ready=0;
  uint64_t gpu_read_bytes=0,gpu_write_bytes=0,pim_read_bytes=0,pim_write_bytes=0;
  std::vector<uint64_t> gpu_command_counts;
  uint64_t gpu_row_hits=0,gpu_row_misses=0;
  uint64_t out_of_order_issues=0,prefetched_prerequisites=0;
  explicit PIMController(int ch,const ConfigNode& spec_cfg,const ConfigNode& metadata);
  void set_stage(const std::vector<Operation>& ops,int stage_id,Clock clk,const std::string& name="microbench");
  void begin_kernel(const Kernel* k,Clock clk);
  void end_kernel();
  void reset_serial_epoch(Clock origin);
  bool kernel_finished() const;
  bool stage_finished(Clock clk) const;
  Clock coarse_upload_ready() const;
  Clock coarse_result_ready() const;
  void settle_coarse(Clock clk);
  Clock step(Clock clk,bool permit_gpu,bool release_gpu=true);
  int command(const Operation& op,bool local) const;
  Clock ready_at(int cmd,const AddrVec_t& addr,uint16_t bank_mask=0) const;
  Clock spacing(int pre,const AddrVec_t& a,int following,const AddrVec_t& b) const;
  Clock refresh_interval(int rank) const;
  void issue(int cmd,const AddrVec_t& addr,Clock clk,int stat_stage,int opcode=-1,uint16_t bank_mask=0);
 private:
  void retire_and_fill(Clock clk,bool permit_gpu);
  Candidate next_pim(Clock clk);
  Candidate next_gpu(Clock clk);
  Candidate next_refresh(Clock clk);
  Candidate make_candidate(int final,const AddrVec_t& addr,Candidate::Source src,Clock earliest,
                           bool local=false,uint16_t bank_mask=0);
  Candidate prepare_mc_request(const Operation& op, AddrVec_t addr, Clock earliest, bool broadcast_read,
                               int command_id, uint16_t bank_mask);
  void update_mc_mode(int cmd, const AddrVec_t& addr);
  void finish_pim(const Candidate& c,Clock clk);
  bool blocked_by_owner(const Candidate& c,Clock clk) const;
  Clock dependencies(const Operation& op,const Cursor& cursor) const;
  bool protect_pim(const Candidate& gpu,const Candidate& pim,Clock clk);
};
std::vector<Operation> parse_operations(const ConfigNode& node);
std::string json_string(const std::string& value);
}
