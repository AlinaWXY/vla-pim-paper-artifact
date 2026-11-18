#include <iostream>
#include <stdexcept>
#include <string>
#include "ramulator/base/config.h"
#include "ramulator/dram/device.h"

using namespace Ramulator;

static void require(bool value, const std::string& message) {
  if (!value) throw std::runtime_error(message);
}
static DRAMDevice device(const ConfigNode& spec) {
  DRAMDevice d;
  d.init(DRAMSpec::create("LPDDR5XPIM", spec));
  return d;
}
static bool buffer(const std::string& name) { return name=="WRGB" || name=="RDRES"; }
static AddrVec_t addr(const std::string& name, int rank, int bg, int bank) {
  return {0, rank, buffer(name)?-1:bg, buffer(name)?-1:bank, 0, 0};
}

int main(int argc, char** argv) {
  try {
    if (argc!=3 && argc!=4)
      throw std::runtime_error("usage: native_spec_tests spec.json command-pair-matrix.json [mc-four-batch-spec.json]");
    auto spec=Config::parse_config_file(argv[1]);
    auto matrix=Config::parse_config_file(argv[2]);
    int checks=0;
    for (const auto& e:matrix.seq()) {
      auto d=device(spec);
      auto first=e["first"].as<std::string>(), second=e["second"].as<std::string>();
      auto rel=e["relation"].as<std::string>();
      int c1=d.m_spec->get_command_id(first), c2=d.m_spec->get_command_id(second);
      auto a1=addr(first,1,0,0);
      auto a2=addr(second,rel=="different_rank"?0:1,rel=="different_bg"?1:0,1);
      const auto gap=e["issue_gap_ck"].as<int>();
      d.issue_command(c1,a1,100);
      require(!d.check_timing(c2,a2,100+gap-1),"accepted early: "+first+" -> "+second+" "+rel);
      require(d.check_timing(c2,a2,100+gap),"rejected legal boundary: "+first+" -> "+second+" "+rel);
      checks+=2;
    }
    // Private row commands retain their own identity when checking preqs.
    {
      auto d=device(spec); AddrVec_t a={0,1,0,0,0,0};
      int act1=d.m_spec->get_command_id("ACT1_LOCAL"),act2=d.m_spec->get_command_id("ACT2_LOCAL");
      require(d.get_preq_command(act1,a,100)==act1,"private ACT1 prerequisite identity");
      d.issue_command(act1,a,100);
      require(d.get_preq_command(act2,a,101)==act2,"private ACT2 prerequisite identity");
      require(d.check_timing(act2,a,101),"private ACT2 earliest legal tick");
      d.issue_command(act2,a,101);
      int read=d.m_spec->get_command_id("LOCALRD");
      require(d.get_preq_command(read,a,121)==read,"private activation did not open shared row state");
      require(!d.check_timing(read,a,120) && d.check_timing(read,a,121),"private activation RCD boundary");
      checks+=5;
    }
    // A private precharge of an OPENED bank must not turn back into external CA.
    for (const std::string name:{"PREpb_LOCAL","PREab_LOCAL"}) {
      auto d=device(spec); AddrVec_t a={0,1,0,0,0,0};
      int act1=d.m_spec->get_command_id("ACT1_LOCAL"),act2=d.m_spec->get_command_id("ACT2_LOCAL");
      d.issue_command(act1,a,100);d.issue_command(act2,a,101);
      int pre=d.m_spec->get_command_id(name),read=d.m_spec->get_command_id("LOCALRD");
      AddrVec_t pa=name=="PREab_LOCAL"?AddrVec_t{0,1,-1,-1,0,0}:a;
      require(d.get_preq_command(pre,pa,200)==pre,"private opened-row precharge identity: "+name);
      require(d.check_timing(pre,pa,200),"private precharge rejected after RAS: "+name);
      d.issue_command(pre,pa,200);
      require(d.get_preq_command(read,a,223)==d.m_spec->get_command_id("ACT1"),"private precharge did not close shared row state: "+name);
      checks+=3;
    }
    // No open row is needed for SRAM/pool IO; array operations retain prerequisites.
    {
      auto d=device(spec); int rd=d.m_spec->get_command_id("LOCALRD");
      AddrVec_t a={0,1,0,0,0,0},other={0,1,0,1,0,0};
      d.issue_command(rd,a,100);
      require(d.check_timing(rd,other,100),"streaming fixed-function local reads were rank-serialized");
      require(d.check_timing(rd,other,101),"different-bank local read inherited a rank data-port lock");
      require(!d.check_timing(rd,a,103)&&d.check_timing(rd,a,104),"local read lost bank CCD boundary");
      require(d.check_timing(d.m_spec->get_command_id("BCAST"),{0,1,-1,-1,0,0},101),"local read locked independent rank unit until completion");
      require(d.check_timing(d.m_spec->get_command_id("WRGB"),{0,1,-1,-1,0,0},100),"private issue occupied external CA");
      checks+=5;
    }
    // Fused private CA is one instruction/CK; array timing is shared with the
    // streaming aliases, while independent banks retain separate pipelines.
    {
      auto d=device(spec); int rd=d.m_spec->get_command_id("LOCALRD_LOCAL");
      int base=d.m_spec->get_command_id("LOCALRD");
      AddrVec_t a={0,1,0,0,0,0},other={0,1,0,1,0,0};
      d.issue_command(rd,a,100);
      require(!d.check_timing(rd,other,100)&&d.check_timing(rd,other,101),"fused local private CA boundary");
      require(d.check_timing(base,other,100),"fixed-function command occupied private CA");
      require(!d.check_timing(base,a,103)&&d.check_timing(base,a,104),"local aliases lost shared bank timing");
      checks+=3;
    }
    // No open row is needed for SRAM/pool IO; array operations retain prerequisites.
    {
      auto d=device(spec);
      for (const std::string c:{"WRGB","RDRES"}) {
        auto id=d.m_spec->get_command_id(c); auto a=addr(c,1,0,0);
        require(d.get_preq_command(id,a,0)==id,"buffer requires array row: "+c); ++checks;
      }
      for (const std::string c:{"MACAB_M16","MACAB_M16_LOCAL","LOCALRD","LOCALWR"}) {
        auto id=d.m_spec->get_command_id(c);
        AddrVec_t a={0,1,c.find("MAC")==0?-1:0,c.find("MAC")==0?-1:0,0,0};
        auto pre=d.get_preq_command(id,a,0);
        const bool mac=c.find("MAC")==0;
        const std::string expected=mac?(c.ends_with("_LOCAL")?"ACTAB_LOCAL":"ACTAB"):"ACT1";
        require(pre==d.m_spec->get_command_id(expected),"missing closed-bank activation prereq: "+c);
        auto pa=d.preq_addr_vec(id,pre,a,0);
        require(mac?(pa[2]==-1&&pa[3]==-1):(pa[2]>=0&&pa[3]>=0),"activation prerequisite addressing"); checks+=2;
      }
    }
    // ACT4x4 reserves all sixteen activations in native histories, with
    // staggered per-bank RCD and one external two-CK command.
    for (const std::string command:{"ACTAB","ACTAB_LOCAL"}) {
      auto d=device(spec); const auto* s=d.m_spec;
      const int act=s->get_command_id(command), rd=s->get_command_id("LOCALRD");
      const int mac=s->get_command_id("MACAB_M16"), normal=s->get_command_id("ACT2");
      AddrVec_t all={0,1,-1,-1,7,0};
      require(d.get_preq_command(act,all,100)==act,"ACTAB closed prerequisite identity");
      d.issue_command(act,all,100);
      for (int bg=0;bg<4;++bg) for (int bank=0;bank<4;++bank) {
        AddrVec_t a={0,1,bg,bank,7,0}; int ready=120+16*bank;
        require(d.get_preq_command(rd,a,ready)==rd,"ACTAB failed to open a bank");
        require(!d.check_timing(rd,a,ready-1)&&d.check_timing(rd,a,ready),"ACTAB per-bank staggered RCD");
        checks+=2;
      }
      require(!d.check_timing(mac,all,167)&&d.check_timing(mac,all,168),"ACTAB all-bank ready is not 68 CK");
      auto* rank=d.node_at({0,1,0,0,7,0},1);
      require(rank->m_cmd_ready_clk[normal]==164,"four ACT4 batches did not reserve shared FAW through 64 CK");
      bool found=false;
      for (const auto& history:rank->m_shared_window_history) {
        if (history.size()==4 && history[0]==148 && history[1]==148 && history[2]==148 && history[3]==148) found=true;
      }
      require(found,"ACTAB failed to record four last-batch ACT pulses in shared history");
      require(!d.check_timing(normal,{0,1,0,0,8,0},163),"ordinary ACT2 entered reserved FAW");
      // At 165 bank0's RC is also satisfied; the shared chronological history
      // can then accept an ordinary ACT2 after the last reserved pulse.
      int prebank=s->get_command_id("PREpb"),first=s->get_command_id("ACT1");
      require(d.check_timing(prebank,{0,1,0,0,7,0},145),"first-batch bank could not precharge independently at RAS");
      d.issue_command(prebank,{0,1,0,0,7,0},145);
      d.issue_command(first,{0,1,0,0,8,0},164);
      require(d.check_timing(normal,{0,1,0,0,8,0},165),"ordinary ACT2 blocked after FAW and bank RC");
      d.issue_command(normal,{0,1,0,0,8,0},165);
      require(rank->m_cmd_ready_clk[act]>=181,"ordinary ACT did not constrain next ACTAB admission");
      checks+=8;
    }
    // Mixed/open rows close the whole rank before ACTAB. PREab recovery is
    // 23 CK, and the externally encoded command occupies CA for exactly 2 CK.
    {
      auto d=device(spec); const auto* s=d.m_spec;
      int mac=s->get_command_id("MACAB_M16"),act=s->get_command_id("ACTAB"),pre=s->get_command_id("PREab");
      AddrVec_t all={0,1,-1,-1,9,0};
      d.issue_command(s->get_command_id("ACT1"),{0,1,0,0,2,0},100);
      d.issue_command(s->get_command_id("ACT2"),{0,1,0,0,2,0},101);
      require(d.get_preq_command(mac,all,200)==pre,"mixed rows did not request PREab");
      require(d.get_preq_command(act,all,200)==pre,"ACTAB admitted a nonclosed rank");
      d.issue_command(pre,all,200);
      require(d.get_preq_command(mac,all,223)==act,"PREab did not enable ACTAB");
      require(!d.check_timing(act,all,222)&&d.check_timing(act,all,223),"PREab to ACTAB must be 23 CK");
      d.issue_command(act,all,223);
      int wrgb=s->get_command_id("WRGB");
      require(!d.check_timing(wrgb,{0,0,-1,-1,0,0},224)&&d.check_timing(wrgb,{0,0,-1,-1,0,0},225),"ACTAB external CA must occupy 2 CK");
      checks+=5;
    }
    // One RoPE PE per bank: eight pairs start in eight CK; the fourth-cycle
    // pair latency gives eleven-CK completion, without a completion-time lock.
    {
      auto d=device(spec); int rope=d.m_spec->get_command_id("ROPE_EXEC");
      int local=d.m_spec->get_command_id("ROPE_EXEC_LOCAL");
      AddrVec_t a={0,1,0,0,0,0},other={0,1,0,1,0,0};
      d.issue_command(rope,a,100);
      require(d.check_timing(rope,other,100),"RoPE PE incorrectly shared across banks");
      require(!d.check_timing(rope,a,107)&&d.check_timing(rope,a,108),"RoPE vector initiation interval must be 8 CK");
      require(!d.check_timing(local,a,107)&&d.check_timing(local,a,108),"RoPE aliases lost shared vector pipeline");
      require(d.check_timing(d.m_spec->get_command_id("LOCALRD"),a,101),"RoPE execution prevented overlapping K load");
      checks+=4;
    }
    // Native REFpb pair targeting must block both banks n and n+8, and full-bank PIM.
    {
      auto d=device(spec); int refresh=d.m_spec->get_command_id("REFpb");
      d.issue_command(refresh,{0,1,0,0,0,0},100);
      for (const std::string c:{"WRGB","RDRES","MACAB_M16","MACAB_M16_LOCAL"}) {
        int id=d.m_spec->get_command_id(c); AddrVec_t a={0,1,-1,-1,0,0};
        require(!d.check_timing(id,a,227),"REFpb did not block "+c);
        require(d.check_timing(id,a,228),"REFpb blocked beyond recovery "+c); checks+=2;
      }
      const auto rd=d.m_spec->get_command_id("RD");
      require(!d.check_timing(rd,{0,1,2,0,0,0},227),"REFpb paired bank not blocked");
      require(d.check_timing(rd,{0,1,1,0,0,0},101),"REFpb unrelated bank blocked"); checks+=2;
    }
    if (argc==4) {
      auto mc_spec=Config::parse_config_file(argv[3]);
      auto d=device(mc_spec);
      const auto* s=d.m_spec;
      const int act=s->get_command_id("ACTAB"), mac=s->get_command_id("MACAB_M16");
      const int rd=s->get_command_id("LOCALRD_LOCAL"), pre=s->get_command_id("PREab");
      const int act2=s->get_command_id("ACT2"), select=s->get_command_id("SRF_SELECT");
      const int host_wr=s->get_command_id("WRGB");
      const int local_rd=s->get_command_id("LOCALRD"), local_wr=s->get_command_id("LOCALWR_LOCAL");
      const int gpu_rd=s->get_command_id("RD"), gpu_wr=s->get_command_id("WR");
      const AddrVec_t row={0,1,-1,-1,7,0}, next={0,1,-1,-1,8,0};
      const AddrVec_t bank={0,1,0,0,7,0};
      const AddrVec_t gpu_bank={0,0,0,0,3,0};
      d.issue_command(act2,bank,100);
      require(d.node_at(bank,1)->m_cmd_ready_clk[act]==116,"ACTAB needs an empty FAW window");
      d=device(mc_spec);
      s=d.m_spec;
      d.issue_command(s->get_command_id("ACT1"),gpu_bank,9);
      d.issue_command(act2,gpu_bank,10);
      d.issue_command(act,row,100);
      for (int cmd:{mac,rd})
        require(!d.check_timing(cmd,row,167)&&d.check_timing(cmd,row,168),"all banks must wait 68 CK");
      require(d.node_at(bank,1)->m_cmd_ready_clk[act2]==164,"ACTAB did not reserve four FAW batches");
      require(!d.check_timing(pre,row,192)&&d.check_timing(pre,row,193),"PREab must wait for last batch RAS");
      require(!d.check_timing(host_wr,row,101)&&d.check_timing(host_wr,row,102),"MC ACTAB must occupy 2 CK of CA");
      d.issue_command(select,{0,1,-1,-1,s->organization.level_sizes[4]-3,-1},102);
      require(d.check_timing(gpu_rd,gpu_bank,116),"internal pulses must allow a GPU RD on the same CK");
      for (int bank_id=0;bank_id<4;++bank_id) {
        const AddrVec_t target={0,1,0,bank_id,7,0};
        const Clk_t ready=120+16*bank_id;
        require(!d.check_timing(local_rd,target,ready-1)&&d.check_timing(local_rd,target,ready),
                "bank-local RD must use that batch's activation time");
      }
      require(!d.check_timing(local_wr,row,156)&&d.check_timing(local_wr,row,157),"all-bank WR must wait for the final batch's tRCDWR");
      constexpr uint16_t score_banks=0x11;
      require(!d.check_timing(local_wr,row,108,score_banks)&&d.check_timing(local_wr,row,109,score_banks),
              "score writeback must wait for the selected banks' tRCDWR");
      require(d.get_preq_command(local_wr,row,109,score_banks)==local_wr,"score banks have the requested row");
      const AddrVec_t other={0,1,0,1,7,0};
      const auto other_pre=d.node_at(other,d.m_bank_level)->m_cmd_ready_clk[s->get_command_id("PREpb")];
      d.issue_command(local_wr,row,109,score_banks);
      const AddrVec_t peer={0,1,1,0,7,0};
      for (const auto& target:{bank,peer})
        require(d.node_at(target,d.m_bank_level)->m_cmd_ready_clk[s->get_command_id("PREpb")]==150,
                "score writeback recovery must reach both selected banks");
      require(d.node_at(other,d.m_bank_level)->m_cmd_ready_clk[s->get_command_id("PREpb")]==other_pre,
              "score writeback must not impose bank recovery on unselected banks");
      require(!d.check_timing(gpu_rd,gpu_bank,109),"score writeback must reserve the shared CA cycle");
      d.issue_command(gpu_rd,gpu_bank,116);
      require(d.check_timing(gpu_wr,gpu_bank,148),"internal pulses must allow a legal GPU WR on the same CK");
      d.issue_command(gpu_wr,gpu_bank,148);
      require(d.check_rowbuffer_hit(s->get_command_id("RD"),bank,168),"SRF selector disturbed the open data row");
      d.issue_command(pre,row,193);
      require(!d.check_timing(act,next,215)&&d.check_timing(act,next,216),"second ACTAB must wait for PREab recovery");
      checks+=21;
    }
    std::cout<<"{\"native_spec_checks\":"<<checks<<",\"status\":\"passed\"}\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr<<e.what()<<"\n"; return 1;
  }
}
