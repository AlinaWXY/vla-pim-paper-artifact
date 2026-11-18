#include <iostream>
#include <string>
#include "ramulator/controller/impl/pim_model.h"

using namespace PimVla;

static void warm_rows(PIMController& c) {
  // Explicit warm row-hit initial state, excluding activation and refresh
  // from these interface microbenchmarks. Full workloads retain both.
  const auto opened=c.device.m_spec->states.at("Opened");
  for(auto* bank:c.device.m_bank_nodes) {
    bank->m_state=opened;bank->m_row_state[0]=opened;
  }
  c.refresh_due.fill(NEVER);
}
static Operation op(std::string name,int index,int bg=0) {
  Operation o;o.name=name;o.id=index;o.addr={0,1,bg,0,0,index%64};
  if(name=="WRGB"||name=="RDRES")o.addr[2]=o.addr[3]=-1;
  o.slot=index%32;return o;
}
static Clock run(PIMController& c,bool gpu=false) {
  Clock clk=0;size_t steps=0;
  while(!c.stage_finished(clk)||(gpu&&!c.kernel_finished())) {
    require(++steps<1000000,"microbenchmark did not terminate");
    clk=c.step(clk,gpu);
  }
  return clk;
}
static Clock issued(const PIMController& c,int index,int cmd) {
  return c.top.done.at(index)-c.info.at(cmd).completion;
}
static void gap_wrgb_wait(const ConfigNode& spec) {
  PIMController c(0,spec,spec["metadata"]);warm_rows(c);c.policy="gap_batch";
  auto mac=op("MACAB",0),write=op("WRGB",1);
  mac.addr[2]=mac.addr[3]=-1;mac.rows=32;write.slot=0;
  const std::vector<Operation> program={mac,write};
  c.set_stage(program,0,0);
  // Seed the 32 input slots, then issue the real MAC. Its SRF lock delays
  // the following WRGB, whose dependency list alone would show no wait.
  for(int slot=0;slot<32;++slot) {
    auto fill=op("WRGB",slot);fill.tensor_fragment=0;
    c.resources.issue(fill,0,0);
  }
  require(c.step(0,false)==0&&c.top.next==1&&c.top.done[0]>1,"fixture did not issue a live MAC");
  // Rank0 PRE is legal and fits inside the known WRGB wait.
  Kernel k;k.start=1;k.reads=16;k.base=16*64*16;
  c.begin_kernel(&k,1);
  const auto pre=c.device.m_spec->get_command_id("PREpb");
  require(c.step(1,true)==1&&c.gpu_command_counts[pre]==1&&c.top.next==1,
          "gap_batch blocked GPU issue during a WRGB head's MAC wait");
}

static void gap_write_drain(const ConfigNode& spec) {
  PIMController c(0,spec,spec["metadata"]);warm_rows(c);c.policy="gap_batch";
  const std::vector<Operation> program={op("BARRIER",0)};
  c.set_stage(program,0,0);c.top.tail=500;
  Kernel write;write.writes=16;
  c.begin_kernel(&write,0);
  Clock now=0;
  while(!c.kernel_finished()&&now<100)now=c.step(now,true);
  require(c.kernel_finished()&&c.gpu_write_bytes==32&&c.gap_write_phase,
          "fixture did not drain its write batch");
  c.end_kernel();
  Kernel read;read.reads=16;read.start=now;
  c.begin_kernel(&read,now);
  while(!c.gpu_read_bytes&&now<100)now=c.step(now,true);
  require(c.gpu_read_bytes==32&&now<100&&!c.gap_write_phase,
          "gap_batch starved reads after draining the write batch");
}

int main(int argc,char** argv) {
  try {
    require(argc>=2,"usage: native_microbench spec.json [--fault-case]");
    const auto spec=Config::parse_config_file(argv[1]);
    const auto metadata=spec["metadata"];
    if(argc==3&&std::string(argv[2])=="--gap-wrgb-wait") {
      gap_wrgb_wait(spec);return 0;
    }
    if(argc==3&&std::string(argv[2])=="--gap-write-drain") {
      gap_write_drain(spec);return 0;
    }
    if(argc==3 && std::string(argv[2])=="--fault-case") {
      PIMController c(0,spec,metadata);warm_rows(c);
      c.issue(c.device.m_spec->get_command_id("RD"),{0,0,0,0,0,0},100,0);
      c.issue(c.device.m_spec->get_command_id("WRGB"),{0,1,-1,-1,0,0},101,0);
      std::cerr<<"FAULT NOT DETECTED\n";return 2;
    }
    gap_wrgb_wait(spec);
    gap_write_drain(spec);
    std::cout<<"{\"microbench\":\"gap_batch_regressions\",\"status\":\"passed\"}\n";
    constexpr int n=64;
    {
      PIMController c(0,spec,metadata);warm_rows(c);
      std::vector<Operation> ops;for(int i=0;i<n;i++)ops.push_back(op("RD",i,i%4));
      c.set_stage(ops,0,0);Clock end=run(c);
      int cmd=c.device.m_spec->get_command_id("RD");
      for(int i=0;i<n;i++)require(issued(c,i,cmd)==2*i,"pure RD expected 2 CK per burst across BG");
      require(end==2*(n-1)+25,"pure RD completion tail");
      require(c.pim_read_bytes==32*n,"pure RD byte conservation");
      std::cout<<"{\"microbench\":\"pure_rd\",\"bursts\":"<<n<<",\"end_ck\":"<<end<<"}\n";
    }
    {
      PIMController c(0,spec,metadata);warm_rows(c);
      std::vector<Operation> ops;for(int i=0;i<n;i++)ops.push_back(op((i/4)%2?"WR":"RD",i));
      c.set_stage(ops,0,0);Clock end=run(c),expected=0;
      for(int i=0;i<n;i++) {
        if(i)expected+=ops[i-1].name==ops[i].name?4:(ops[i-1].name=="RD"?17:29);
        int cmd=c.device.m_spec->get_command_id(ops[i].name);
        require(issued(c,i,cmd)==expected,"alternating batch analytical command gap");
      }
      require(end==expected+14,"alternating batch completion tail");
      require(c.pim_read_bytes+c.pim_write_bytes==32*n,"alternating byte conservation");
      std::cout<<"{\"microbench\":\"alternating_batches\",\"bursts\":"<<n<<",\"end_ck\":"<<end<<"}\n";
    }
    {
      PIMController c(0,spec,metadata);warm_rows(c);
      std::vector<Operation> ops;for(int i=0;i<n;i++)ops.push_back(op("WRGB",i));
      c.set_stage(ops,0,0);Clock end=run(c);int cmd=c.device.m_spec->get_command_id("WRGB");
      for(int i=0;i<n;i++)require(issued(c,i,cmd)==4*i,"WRGB expected same-BG 4 CK spacing");
      require(end==4*(n-1)+14,"WRGB must complete after CWL+BL");
      require(c.pim_write_bytes==32*n,"WRGB external byte count is one burst, not 16");
      std::cout<<"{\"microbench\":\"pure_wrgb\",\"bursts\":"<<n<<",\"end_ck\":"<<end<<"}\n";
    }
    {
      PIMController c(0,spec,metadata);warm_rows(c);c.policy="pim_head";
      std::vector<Operation> ops;for(int i=0;i<n;i++)ops.push_back(op("WRGB",i));
      c.set_stage(ops,0,0);
      Kernel k;k.reads=16;k.start=0;k.nominal=0;k.base=0;c.begin_kernel(&k,0);
      Clock end=run(c,true);int cmd=c.device.m_spec->get_command_id("WRGB");
      for(int i=0;i<n;i++)require(issued(c,i,cmd)==4*i,"GPU RD delayed a pending WRGB train");
      Clock gpu_issue=c.gpu_dq_end-25;
      require(gpu_issue>=4*(n-1)+2,"GPU RD inserted inside a continuous WRGB train");
      require(c.gpu_read_bytes==32 && c.pim_write_bytes==32*n,"mixed byte conservation");
      std::cout<<"{\"microbench\":\"wrgb_gpu_rd\",\"gpu_issue_ck\":"<<gpu_issue<<",\"end_ck\":"<<end<<"}\n";
    }
    {
      // The waiting BARRIER has no native command, but a WRGB behind it is
      // visible within lookahead. A GPU RD at CK2 would move that WRGB from
      // CK14 to CK17. Head-only protection permits it; lookahead must not.
      Clock head_write=0,lookahead_write=0,lookahead_gpu=0;
      for(const std::string policy:{"pim_head","pim_lookahead"}) {
        PIMController c(0,spec,metadata);warm_rows(c);c.policy=policy;
        auto first=op("WRGB",0),barrier=op("BARRIER",1),last=op("WRGB",2);
        barrier.deps={0};last.deps={1};
        std::vector<Operation> ops={first,barrier,last};c.set_stage(ops,0,0);
        Kernel k;k.reads=16;k.start=0;k.nominal=0;k.base=0;c.begin_kernel(&k,0);
        run(c,true);Clock wr=issued(c,2,c.device.m_spec->get_command_id("WRGB"));
        if(policy=="pim_head")head_write=wr;
        else {lookahead_write=wr;lookahead_gpu=c.gpu_dq_end-25;}
        require(c.gpu_read_bytes==32&&c.pim_write_bytes==64,"barrier forecast byte conservation");
      }
      require(head_write==17,"head-only barrier control did not expose expected interference");
      require(lookahead_write==14&&lookahead_gpu==16,"lookahead did not protect WRGB behind a waiting barrier");
      std::cout<<"{\"microbench\":\"lookahead_barrier\",\"head_wrgb_ck\":"<<head_write
               <<",\"lookahead_wrgb_ck\":"<<lookahead_write<<",\"lookahead_gpu_ck\":"<<lookahead_gpu<<"}\n";
    }
    return 0;
  } catch(const std::exception& e) {std::cerr<<e.what()<<"\n";return 1;}
}
