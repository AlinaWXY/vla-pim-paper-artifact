#include "ramulator/controller/impl/pim_power_stats.h"
#include "ramulator/base/config.h"
#include <iostream>
#include <numeric>
#include <stdexcept>
using namespace Ramulator;
using PimVla::DramPowerStats;

void need(bool x,const char* why){if(!x)throw std::runtime_error(why);}
auto address(int rank,int bank=0){return AddrVec_t{0,rank,bank/4,bank%4,0,0};}
template<class T>uint64_t sum(const T& x){return std::accumulate(x.begin(),x.end(),uint64_t(0));}
void conservation(const DramPowerStats& p,int64_t end) {
  for(const auto& b:p.banks)need(sum(b.cycles)==uint64_t(end),"bank duration does not conserve");
  for(int rank=0;rank<2;++rank) {
    uint64_t clocks=0,bank_clocks=0;
    for(const auto& [id,s]:p.stages) {
      const auto& r=s[rank];auto t=sum(r.active_banks_ck);
      need(t==sum(r.refresh_banks_ck)&&t==sum(r.active_or_refresh_banks_ck),"rank histogram durations disagree");
      need(sum(r.bank_cycles)==16*t,"stage bank/rank time does not conserve");
      clocks+=t;bank_clocks+=sum(r.bank_cycles);
    }
    need(clocks==uint64_t(end)&&bank_clocks==uint64_t(16*end),"full rank duration does not conserve");
  }
}
int main(int argc,char** argv) {
  try {
    need(argc==2,"spec required");auto spec=Config::parse_config_file(argv[1]);
    auto make=[&] {DRAMDevice d;d.init(DRAMSpec::create("LPDDR5XPIM",spec));return d;};
    {
      auto d=make();DramPowerStats p(d);p.set_stage(0,0);
      auto issue=[&](const char* c,int t){p.observe(d.m_spec->get_command_id(c),address(0),t,0,0);};
      issue("ACT1",10);issue("ACT2",12);p.set_stage(50,1);issue("PREpb",80);p.finish(110);
      auto& b=p.banks[0];need(b.activations==1&&b.precharges==1,"split ACT counted twice");
      need(b.cycles[DramPowerStats::Activating]==2&&b.cycles[DramPowerStats::Active]==68,"split ACT effective state time wrong");
      need(b.cycles[DramPowerStats::Precharging]==20&&b.cycles[DramPowerStats::Precharged]==20,"precharge recovery accounting wrong");
      need(p.stages.at(0)[0].active_banks_ck[1]==40&&p.stages.at(1)[0].active_banks_ck[1]==30,"state time crossed stage boundary incorrectly");
      conservation(p,110);std::cout<<"split_ACT_and_stage_boundary passed\n";
    }
    {
      auto d=make();DramPowerStats p(d);auto all=address(1);all[2]=all[3]=-1;
      const int act=d.m_spec->get_command_id("ACTAB"),mac=d.m_spec->get_command_id("MACAB_M16");
      p.observe(act,all,100,0,0);
      need(p.banks[16].activations==1&&p.banks[17].activations==0,"future ACTAB pulse applied immediately");
      p.observe(mac,all,121,0x1111,0);p.observe(mac,all,137,0x2222,0);p.finish(200);
      const auto& r=p.stages.at(-1)[1];need(r.activations==16&&r.native_commands[act]==1,"ACTAB activation expansion wrong");
      need(r.reads==8,"bank-mask access expansion wrong");
      need(r.active_or_refresh_banks_ck[0]==100&&r.active_or_refresh_banks_ck[4]==16&&
           r.active_or_refresh_banks_ck[8]==16&&r.active_or_refresh_banks_ck[12]==16&&
           r.active_or_refresh_banks_ck[16]==52,"ACTAB four-batch effective timestamps wrong");
      need(r.bank_cycles[DramPowerStats::Active]==1216,"ACTAB bank clock total wrong");
      conservation(p,200);std::cout<<"ACTAB_future_batches_and_masks passed\n";
    }
    {
      auto d=make();DramPowerStats p(d);
      p.observe(d.m_spec->get_command_id("REFpb"),address(0),50,0,0);
      auto all=address(1);all[2]=all[3]=-1;
      p.observe(d.m_spec->get_command_id("REFab"),all,60,0,0);p.finish(200);
      need(p.banks[0].refpb==1&&p.banks[8].refpb==1&&p.banks[1].refpb==0,"REFpb pair target wrong");
      need(p.banks[0].cycles[DramPowerStats::Refreshing]==128,"REFpb recovery duration wrong");
      need(p.stages.at(-1)[1].refab_banks==16&&p.banks[16].cycles[DramPowerStats::Refreshing]==140,
           "REFab time was not clipped at the epoch end");
      conservation(p,200);std::cout<<"paired_refresh_and_end_clipping passed\n";
    }
    {
      auto d=make();DramPowerStats p(d);
      p.observe(d.m_spec->get_command_id("WRGB"),address(1),0,0,'W');
      p.observe(d.m_spec->get_command_id("BCAST"),address(1),1,0,0);
      p.observe(d.m_spec->get_command_id("ROPE_EXEC"),address(1),2,0,0);p.finish(10);
      const auto& r=p.stages.at(-1)[1];need(r.reads==0&&r.writes==0&&r.io_writes==1,"register/compute commands counted as array access");
      conservation(p,10);std::cout<<"register_and_array_separation passed\n";
    }
    return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
