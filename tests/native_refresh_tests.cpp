#include <functional>
#include <iostream>
#include <string>
#include <vector>
#include "ramulator/controller/impl/pim_model.h"

using namespace PimVla;

namespace {
Clock advance_until(PIMController& c,Clock now,const std::function<bool()>& done,
                    Clock limit,bool permit_gpu=false) {
  for(int steps=0;!done();++steps) {
    require(steps<10000 && now<=limit,"refresh test failed to make bounded progress");
    Clock next=c.step(now,permit_gpu);
    require(next>=now && next<=limit,"refresh test skipped its expected event");
    now=next;
  }
  return now;
}

AddrVec_t address(int rank,int bank,int row=0) {
  return {0,rank,bank/4,bank%4,row,0};
}

void report(const std::string& name) {
  std::cout<<"{\"refresh_test\":\""<<name<<"\",\"status\":\"passed\"}\n";
}
}

int main(int argc,char** argv) {
  try {
    require(argc==2,"usage: native_refresh_tests spec.json");
    const auto spec=Config::parse_config_file(argv[1]);
    {
      PIMController c(0,spec,spec["metadata"]);
      require(c.refresh_due[0]==520 && c.refresh_due[1]==4164,
              "first refresh deadlines must be rank0 REFIpb=520 and rank1 REFI=4164");
      require(c.refresh_bank[0]==0 && c.refresh_bank[1]==0,"initial refresh cursor is not zero");
      require(c.refresh_counts[0][0]==0 && c.refresh_counts[0][1]==0 &&
              c.refresh_counts[1][0]==0 && c.refresh_counts[1][1]==0,"nonzero initial refresh counts");
      report("rank_specific_initial_deadlines");
    }
    {
      PIMController c(0,spec,spec["metadata"]);
      c.refresh_due[1]=NEVER;
      const int rd=c.device.m_spec->get_command_id("RD");
      const Clock interval=c.device.m_spec->get_timing_value("nREFIpb");
      const Clock recovery=c.device.m_spec->get_timing_value("nRFCpb");
      Clock now=0;
      // All eight pair addresses, then wrap to pair 0 on the ninth refresh.
      for(int i=0;i<9;++i) {
        now=advance_until(c,now,[&]{return c.refresh_counts[0][0]==static_cast<uint64_t>(i+1);},(i+1)*interval);
        require(now==(i+1)*interval,"rank0 REFpb did not follow its spec period");
        require(c.refresh_due[0]==(i+2)*interval,"rank0 deadline advanced by the wrong period");
        require(c.refresh_bank[0]==(i+1)%8,"rank0 REFpb pair cursor did not rotate/wrap");
        const int pair=i%8;
        for(int target:{pair,pair+8}) {
          require(!c.device.check_timing(rd,address(0,target),now+recovery-1),"REFpb failed to block a paired bank");
          require(c.device.check_timing(rd,address(0,target),now+recovery),"REFpb paired bank blocked beyond nRFCpb");
        }
        require(c.device.check_timing(rd,address(0,(pair+1)%8),now+1),"REFpb blocked an unrelated rank0 bank");
        require(c.device.check_timing(rd,address(1,pair),now+1),"rank0 REFpb blocked the PIM rank");
      }
      require(c.refresh_counts[0][1]==0 && c.refresh_counts[1][0]==0 && c.refresh_counts[1][1]==0,
              "rank0-only fixture emitted the wrong refresh type/rank");
      require(c.gpu_command_counts[c.device.m_spec->get_command_id("REFpb")]==9 &&
              c.gpu_command_counts[c.device.m_spec->get_command_id("REFab")]==0,
              "rank0 observed command IDs disagree with REFpb counters");
      report("rank0_paired_scope_rotation_and_period");
    }
    {
      PIMController c(0,spec,spec["metadata"]);
      c.refresh_due[0]=NEVER;
      const Clock interval=c.device.m_spec->get_timing_value("nREFI");
      const Clock recovery=c.device.m_spec->get_timing_value("nRFC");
      require(recovery==224,"rank1 all-bank recovery is not native nRFC=224");
      Clock now=advance_until(c,0,[&]{return c.refresh_counts[1][1]==1;},interval);
      require(now==4164 && c.refresh_due[1]==8328,"rank1 first REFab or repeated deadline is wrong");
      for(int bank=0;bank<16;++bank) {
        for(const std::string name:{"LOCALRD","LOCALWR","LOCALRD_LOCAL","ROPE_EXEC","ROPE_EXEC_LOCAL"}) {
          const int cmd=c.device.m_spec->get_command_id(name);
          require(!c.device.check_timing(cmd,address(1,bank),now+recovery-1),"REFab failed to pause every PIM bank/unit");
          require(c.device.check_timing(cmd,address(1,bank),now+recovery),"REFab blocked bank/unit beyond nRFC");
        }
      }
      for(const std::string name:{"WRGB","RDRES","MACAB_M16","MACAB_M16_LOCAL","ACTAB","ACTAB_LOCAL"}) {
        const int cmd=c.device.m_spec->get_command_id(name); AddrVec_t all={0,1,-1,-1,0,0};
        require(!c.device.check_timing(cmd,all,now+recovery-1),"REFab failed to pause full-rank PIM command");
        require(c.device.check_timing(cmd,all,now+recovery),"REFab full-rank command blocked beyond nRFC");
        if(name=="MACAB_M16" || name=="MACAB_M16_LOCAL") {
          const int activation=c.device.m_spec->get_command_id(name.ends_with("_LOCAL")?"ACTAB_LOCAL":"ACTAB");
          require(c.device.get_preq_command(cmd,all,now+recovery)==activation,
                  "REFab recovery incorrectly allowed MAC without reopening all banks");
        }
      }
      require(c.device.check_timing(c.device.m_spec->get_command_id("RD"),address(0,0),now+1),
              "rank1 REFab globally blocked rank0 after its CA slot");
      now=advance_until(c,now,[&]{return c.refresh_counts[1][1]==2;},2*interval);
      require(now==2*interval && c.refresh_due[1]==3*interval,"rank1 REFab period drifted");
      require(c.refresh_bank[1]==0 && c.refresh_counts[1][0]==0 && c.refresh_counts[0][0]==0,
              "rank1 incorrectly used a per-bank cursor or REFpb");
      report("rank1_all_bank_scope_recovery_and_period");
    }
    {
      PIMController c(0,spec,spec["metadata"]);
      c.refresh_due={100,100};
      // Warm an unrelated GPU bank to make a real GPU read independent of
      // row preparation. The refresh targets remain closed and legal.
      auto* bank=c.device.m_bank_nodes.at(1);
      bank->m_state=c.device.m_spec->states.at("Opened");bank->m_row_state[0]=bank->m_state;
      Clock now=advance_until(c,100,[&]{return c.refresh_counts[0][0]==1;},100);
      now=advance_until(c,now,[&]{return c.refresh_counts[1][1]==1;},101);
      require(now==101,"two ranks serialized recovery instead of only their external CA slots");
      const int rd=c.device.m_spec->get_command_id("RD");
      require(c.device.check_timing(rd,address(0,0),228),"rank1 REFab extended rank0 paired recovery");
      Operation op;op.name="LOCALRD";op.id=0;op.addr=address(1,1);
      const std::vector<Operation> program={op};
      c.set_stage(program,0,102,"microbench");
      Kernel k;k.reads=16;k.writes=0;k.base=1024;k.start=102;k.nominal=1;
      c.begin_kernel(&k,102);c.step(102,true);
      require(c.gpu_read_bytes==32,"unrelated GPU bank could not issue while the PIM rank refreshed");
      require(!c.top.dispatched[0],"rank1 PIM request bypassed active REFab");
      now=advance_until(c,102,[&]{return c.top.dispatched[0]!=0;},500,true);
      require(c.top.done[0]-c.info[c.command(op,false)].completion>=101+224,
              "PIM request issued before native all-bank refresh recovery");
      require(c.refresh_counts[0][1]==0 && c.refresh_counts[1][0]==0,"mixed-rank fixture chose a wrong refresh type");
      report("independent_ranks_and_live_pim_pause");
    }
    {
      PIMController c(0,spec,spec["metadata"]);
      c.refresh_due[1]=NEVER;
      advance_until(c,0,[&]{return c.refresh_counts[0][0]==3;},1560);
      require(c.refresh_bank[0]==3,"reset fixture did not advance the paired cursor");
      auto* stale=c.device.m_bank_nodes.at(17);
      stale->m_state=c.device.m_spec->states.at("Opened");stale->m_row_state[17]=stale->m_state;
      const Clock origin=10000;
      c.reset_serial_epoch(origin);
      require(c.refresh_due[0]==origin+520 && c.refresh_due[1]==origin+4164,
              "serial reset did not relocate the two distinct refresh intervals");
      require(c.refresh_bank[0]==0 && c.refresh_bank[1]==0,"serial reset retained a stale pair cursor");
      for(const auto* bank:c.device.m_bank_nodes)
        require(bank->m_state==c.device.m_spec->states.at("Closed") && bank->m_row_state.empty(),
                "serial reset retained an old open row");
      const auto count=c.refresh_counts[0][0];
      Clock now=advance_until(c,origin,[&]{return c.refresh_counts[0][0]==count+1;},origin+520);
      require(now==origin+520 && c.refresh_bank[0]==1,"first post-reset refresh did not restart at pair0");
      report("serial_origin_reset");
    }
    return 0;
  } catch(const std::exception& e) {
    std::cerr<<"native_refresh_tests: "<<e.what()<<'\n';return 1;
  }
}
