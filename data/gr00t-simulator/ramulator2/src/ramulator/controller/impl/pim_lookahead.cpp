#include "ramulator/controller/impl/pim_lookahead.h"
#include "ramulator/controller/impl/pim_model.h"
#include <unordered_map>

namespace PimVla {
namespace {
void copy_dynamic(DRAMNode& dst,const DRAMNode& src) {
  require(dst.m_child_nodes.size()==src.m_child_nodes.size(),"lookahead topology mismatch");
  dst.m_state=src.m_state;
  dst.m_row_state=src.m_row_state;
  dst.m_cmd_ready_clk=src.m_cmd_ready_clk;
  dst.m_cmd_history=src.m_cmd_history;
  dst.m_shared_window_history=src.m_shared_window_history;
  for(size_t i=0;i<dst.m_child_nodes.size();++i)
    copy_dynamic(*dst.m_child_nodes[i],*src.m_child_nodes[i]);
}
Clock native_ready(const DRAMNode& n,int cmd,const AddrVec_t& addr) {
  Clock at=n.m_cmd_ready_clk.at(cmd);
  if(n.m_child_nodes.empty())return at;
  int child=addr.at(n.m_level+1);
  if(child>=0)return std::max(at,native_ready(*n.m_child_nodes.at(child),cmd,addr));
  for(const auto& p:n.m_child_nodes)at=std::max(at,native_ready(*p,cmd,addr));
  return at;
}
Clock native_ready(const DRAMDevice& d,int cmd,const AddrVec_t& addr) {
  Clock at=native_ready(*d.m_root,cmd,addr);
  if(d.m_spec->bank_targets.at(cmd)==BankTarget::Pair)
    at=std::max(at,native_ready(*d.m_root,cmd,d.partner_addr_vec(cmd,addr)));
  return std::max<Clock>(0,at);
}
bool mac(const std::string& n) {return n.rfind("MACAB",0)==0||n.rfind("MACQK",0)==0;}
}

struct NativeLookahead::Impl {
  DRAMDevice shadow;
  std::vector<Clock> completion;
  const Operation* prefix_ops=nullptr;
  const Clock* prefix_done=nullptr;
  size_t prefix_next=0;
  Clock prefix_tail=0;
  Clock prefix_first=NEVER;
  explicit Impl(int channel,const ConfigNode& spec_cfg) {
    require(bool(spec_cfg["dram"])&&bool(spec_cfg["metadata"]),
            "NativeLookahead requires full spec export, including completion metadata");
    shadow.init(DRAMSpec::create(spec_cfg["dram"]["impl"].as<std::string>(),spec_cfg));
    shadow.set_channel_id(channel);
    completion.resize(shadow.m_spec->command_count);
    auto metadata=spec_cfg["metadata"]["commands"];
    for(int i=0;i<shadow.m_spec->command_count;++i) {
      const auto& name=shadow.m_spec->command_names[i];
      auto m=metadata[name];
      if(!m&&name=="SRF_SELECT")continue; // Optional in archived specs.
      require(bool(m),"lookahead completion metadata missing");
      completion[i]=m["completion"].as<int>();
    }
  }
  int command(const Operation& op,bool local) const {
    std::string name=op.name;
    if(mac(name))name+="_M"+std::to_string(op.rows);
    if(local&&shadow.m_spec->has_command(name+"_LOCAL"))name+="_LOCAL";
    return shadow.m_spec->get_command_id(name);
  }
  int private_row(int cmd,bool local) const {
    const auto& n=shadow.m_spec->command_names.at(cmd);
    if(local&&shadow.m_spec->has_command(n+"_LOCAL"))
      return shadow.m_spec->get_command_id(n+"_LOCAL");
    return cmd;
  }
  Clock actual_tail(const std::vector<Operation>& ops,size_t next,const std::vector<Clock>& done) {
    if(prefix_ops!=ops.data()||prefix_done!=done.data()||next<prefix_next||
       (prefix_next&&done.at(0)!=prefix_first)) {
      prefix_ops=ops.data();prefix_done=done.data();prefix_next=0;prefix_tail=0;prefix_first=NEVER;
    }
    // Streaming issued completions are immutable. Do not rescan the complete
    // program for each forecast; this cache advances with the real cursor.
    for(;prefix_next<next;++prefix_next) {
      require(done.at(prefix_next)<NEVER,"lookahead issued prefix has an unresolved descriptor");
      prefix_tail=std::max(prefix_tail,done[prefix_next]);
      if(prefix_next==0)prefix_first=done[0];
    }
    return prefix_tail;
  }
};

NativeLookahead::NativeLookahead(int channel,const ConfigNode& spec_cfg)
  :impl(std::make_unique<Impl>(channel,spec_cfg)) {}
NativeLookahead::~NativeLookahead()=default;
NativeLookahead::NativeLookahead(NativeLookahead&&) noexcept=default;
NativeLookahead& NativeLookahead::operator=(NativeLookahead&&) noexcept=default;

std::vector<ForecastEntry> NativeLookahead::predict(const DRAMDevice& live,
    const std::vector<Operation>& ops,size_t next,const std::vector<Clock>& done,
    Clock now,Clock horizon,bool local,int row_offset) {
  require(next<=ops.size()&&done.size()==ops.size(),"invalid forecast cursor");
  require(horizon>=0&&now<NEVER-horizon,"invalid forecast horizon");
  auto& p=*impl;
  require(live.m_spec->command_names==p.shadow.m_spec->command_names,"lookahead spec mismatch");
  copy_dynamic(*p.shadow.m_root,*live.m_root);
  std::vector<ForecastEntry> result;result.reserve(64);
  std::unordered_map<size_t,Clock> predicted_done;
  Clock at=now,tail=p.actual_tail(ops,next,done),end=now+horizon;
  for(size_t i=next;i<std::min(ops.size(),next+32);++i) {
    const auto& op=ops[i];
    Clock earliest=at;
    for(int d:op.deps) {
      require(d>=0&&static_cast<size_t>(d)<i,"invalid forecast dependency");
      Clock finish=d<static_cast<int>(next)?done.at(d):predicted_done.at(d);
      require(finish<NEVER,"forecast depends on unresolved request");
      earliest=std::max(earliest,finish);
    }
    if(op.name=="BARRIER") {
      earliest=std::max(earliest,tail);
      if(earliest>end)break;
      predicted_done[i]=earliest;at=earliest;continue;
    }
    const int final=p.command(op,local);
    AddrVec_t physical=op.addr;physical[4]+=row_offset;
    bool issued=false;
    for(int preq_steps=0;preq_steps<128;++preq_steps) {
      int cmd=p.shadow.get_preq_command(final,physical,earliest);
      auto a=p.shadow.preq_addr_vec(final,cmd,physical,earliest);
      cmd=p.private_row(cmd,local);
      earliest=std::max(earliest,native_ready(p.shadow,cmd,a));
      if(earliest>end)return result;
      require(p.shadow.check_timing(cmd,a,earliest),"forecast native timing mismatch");
      require(p.shadow.get_preq_command(cmd,a,earliest)==cmd,"forecast native prerequisite mismatch");
      p.shadow.issue_command(cmd,a,earliest);
      result.push_back({cmd,std::move(a),earliest});
      if(cmd==final) {
        Clock latency=p.completion.at(cmd);
        if(op.name=="ROPE_EXEC")latency+=(op.pairs-1)*p.shadow.m_spec->get_timing_value("nROPEII");
        Clock finish=earliest+latency;
        predicted_done[i]=finish;tail=std::max(tail,finish);at=earliest;issued=true;break;
      }
    }
    require(issued,"forecast prerequisite expansion exceeded 128 commands");
    // Lookahead is exposed only for streaming. Do not invent a nested program
    // schedule if this utility is accidentally called across a descriptor.
    if(!op.children.empty())break;
  }
  return result;
}
}
