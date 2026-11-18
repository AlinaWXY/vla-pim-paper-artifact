#include <iostream>
#include <sstream>
#include "ramulator/controller/impl/pim_model.h"
#include "ramulator/controller/impl/pim_lookahead.h"

using namespace PimVla;

static void fingerprint(const DRAMNode& n,std::ostream& out) {
  out<<n.m_level<<':'<<n.m_node_id<<':'<<n.m_state<<';';
  for(auto c:n.m_cmd_ready_clk)out<<c<<',';
  for(const auto& h:n.m_cmd_history){out<<'h';for(auto c:h)out<<c<<',';}
  for(const auto& h:n.m_shared_window_history){out<<'s';for(auto c:h)out<<c<<',';}
  std::map<int,int> rows(n.m_row_state.begin(),n.m_row_state.end());
  for(auto [row,state]:rows)out<<'r'<<row<<':'<<state;
  for(const auto& child:n.m_child_nodes)fingerprint(*child,out);
}
static std::string fingerprint(const DRAMDevice& d) {
  std::ostringstream out;fingerprint(*d.m_root,out);return out.str();
}
static Operation operation(std::string name,int index) {
  Operation o;o.name=name;o.id=index;o.addr={0,1,-1,-1,0,0};o.rows=1;return o;
}
static Clock find_time(const std::vector<ForecastEntry>& forecast,int cmd) {
  for(const auto& e:forecast)if(e.cmd==cmd)return e.at;
  return NEVER;
}
int main(int argc,char** argv) {
  try {
    require(argc==2,"usage: native_lookahead_tests spec.json");
    auto spec=Config::parse_config_file(argv[1]);
    DRAMDevice live;live.init(DRAMSpec::create("LPDDR5XPIM",spec));
    NativeLookahead predictor(0,spec);
    int wrgb=live.m_spec->get_command_id("WRGB"),rdres=live.m_spec->get_command_id("RDRES");
    auto first=operation("WRGB",0),cast=operation("CAST",1),read=operation("RDRES",2);
    cast.deps={0};read.deps={1};
    std::vector<Operation> ops={first,cast,read};std::vector<Clock> done(ops.size(),NEVER);
    std::string before=fingerprint(live);
    auto f=predictor.predict(live,ops,0,done,0,40,false);
    require(find_time(f,wrgb)==0,"forecast omitted head WRGB");
    require(find_time(f,rdres)==29,"forecast did not preserve same-rank all-BG WTR");
    require(fingerprint(live)==before,"forecast mutated the live DRAMDevice");
    auto repeat=predictor.predict(live,ops,0,done,0,40,false);
    require(repeat.size()==f.size(),"forecast replay size changed");
    for(size_t i=0;i<f.size();++i)require(f[i].cmd==repeat[i].cmd&&f[i].addr==repeat[i].addr&&f[i].at==repeat[i].at,"forecast nondeterminism");
    // An internal head has no external DQ. Looking through it reveals the
    // result read; a GPU write at CK100 needs a 2 CK sibling gap and cannot
    // preserve this RDRES's CK101 lower bound. Head-only inspection misses it.
    live.issue_command(wrgb,first.addr,0);done[0]=14;
    before=fingerprint(live);
    f=predictor.predict(live,ops,1,done,100,27,false);
    require(find_time(f,rdres)==101,"future result read hidden behind local head");
    require(fingerprint(live)==before,"forecast changed committed IO timing");
    require(predictor.predict(live,ops,1,done,100,0,false).size()==1,"forecast exceeded horizon");
    // BARRIER waits for all preceding completions, including an already-issued
    // long tail that is not explicitly listed as a dependency.
    std::vector<Operation> barrier={operation("WRGB",0),operation("BARRIER",1),operation("RDRES",2)};
    std::vector<Clock> barrier_done={150,NEVER,NEVER};
    require(predictor.predict(live,barrier,1,barrier_done,100,27,false).empty(),"barrier ignored live completion tail");
    std::vector<Operation> many;for(int i=0;i<100;i++)many.push_back(operation("CAST",i));
    std::vector<Clock> many_done(100,NEVER);
    require(predictor.predict(live,many,0,many_done,100,200,false).size()==32,"forecast exceeded 32 request FIFO");
    // ACTAB is one predicted command; its sixteen staggered native primitives
    // affect the snapshot readiness without becoming controller instructions.
    std::vector<Operation> array={operation("MACAB",0)};std::vector<Clock> array_done={NEVER};
    before=fingerprint(live);
    f=predictor.predict(live,array,0,array_done,100,200,true);
    int act=live.m_spec->get_command_id("ACTAB_LOCAL");
    int count=0;for(const auto& e:f)count+=e.cmd==act;
    require(count==1&&find_time(f,act)==100,"forecast did not issue one full-rank ACTAB");
    require(find_time(f,live.m_spec->get_command_id("MACAB_M1_LOCAL"))==168,"forecast ignored staggered ACTAB completion");
    require(fingerprint(live)==before,"private prerequisite prediction mutated live rows");
    // A snapshot taken after real ACTAB carries future reservations and opened
    // rows; only the final command remains and it still cannot start early.
    live.issue_command(live.m_spec->get_command_id("ACTAB"),array[0].addr,200);
    before=fingerprint(live);
    f=predictor.predict(live,array,0,array_done,220,100,true);
    require(find_time(f,act)==NEVER,"snapshot lost committed ACTAB row state");
    require(find_time(f,live.m_spec->get_command_id("MACAB_M1_LOCAL"))==268,"snapshot lost future ACTAB ready clocks");
    require(fingerprint(live)==before,"ACTAB snapshot forecast changed live reservations");
    std::cout<<"{\"native_lookahead\":\"passed\",\"coverage\":\"future_DQ, native_preq, live_immutability, repeat, barrier, horizon, FIFO32\"}\n";
    return 0;
  } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
