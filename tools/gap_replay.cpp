// Offline fluid replay. This executable does not link to the simulator.
#include "../third_party/nlohmann_json/json.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Json = nlohmann::json;
constexpr int channels = 16;
constexpr double epsilon = 1e-7;
enum Direction { Read, Write, Open, Unset, Conservative };
using Edges = std::array<std::array<int, 3>, 3>;
using Counts = std::array<double, 2>;
enum class Policy { SameFirst, Best, Random };

struct Unit {
  Counts requests;
  double duration, rate_ck;
};

struct Gap {
  int start, end, left, right;
};

struct Stage {
  std::string name;
  int duration;
  bool pim;
  std::array<std::vector<Gap>, channels> gaps;
};

struct Channel {
  Counts remaining{};
  int previous = Unset;
  size_t index = 0;
  double time = 0;
};

struct Fill {
  int first = Unset, last = Unset, assumed = Unset, switches = 0;
  double capacity = 0, stop = 0;
  Counts sent{};
};

struct Hint {
  int direction = Unset, retries = 0;
};

double sendable(const Unit& unit, const Channel& channel, int direction) {
  if (direction == Read || unit.requests[Read] == 0)
    return channel.remaining[direction];
  return std::max(0.0, channel.remaining[Write] -
                           channel.remaining[Read] * (unit.requests[Write] / unit.requests[Read]));
}

double uniform(uint64_t seed, int repetition, size_t stage, int channel, size_t gap) {
  uint64_t bits = seed ^ (uint64_t(repetition) << 50) ^ (uint64_t(stage) << 43) ^
                  (uint64_t(channel) << 39) ^ gap;
  bits = (bits ^ (bits >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
  bits = (bits ^ (bits >> 27)) * UINT64_C(0x94d049bb133111eb);
  return double((bits ^ (bits >> 31)) >> 11) * 0x1.0p-53;
}

int choose(const Unit& unit, const Channel& channel, int left, const Edges& edges,
           Policy policy, double draw) {
  if (sendable(unit, channel, Read) <= epsilon) return Write;
  if (sendable(unit, channel, Write) <= epsilon) return Read;
  const int preferred = channel.previous == Write ? Write : Read;
  if (policy == Policy::SameFirst) return preferred;
  if (policy == Policy::Best) {
    if (edges[left][Read] == edges[left][Write]) return preferred;
    return edges[left][Read] < edges[left][Write] ? Read : Write;
  }
  return draw * (channel.remaining[Read] + channel.remaining[Write]) < channel.remaining[Read]
             ? Read : Write;
}

void consume(const Unit& unit, Channel& channel, Fill& fill, int direction, double stop) {
  while (channel.time + epsilon < stop) {
    double possible = sendable(unit, channel, direction);
    if (possible <= epsilon) {
      direction = 1 - direction;
      possible = sendable(unit, channel, direction);
      if (possible <= epsilon) break;
    }
    const double sent = std::min(possible, (stop - channel.time) / unit.rate_ck);
    if (sent <= epsilon) break;
    channel.remaining[direction] = std::max(0.0, channel.remaining[direction] - sent);
    channel.time += sent * unit.rate_ck;
    channel.previous = direction;
    fill.sent[direction] += sent;
    if (fill.last != Unset && fill.last != direction) ++fill.switches;
    fill.last = direction;
    direction = 1 - direction;
  }
}

bool complete(const Channel& channel) {
  return channel.remaining[Read] <= epsilon && channel.remaining[Write] <= epsilon;
}

using Fills = std::array<std::vector<Fill>, channels>;
using Hints = std::array<std::vector<Hint>, channels>;

struct Progress {
  std::array<Channel, channels> channel;
  size_t ordinal = 0;
  double unit_start = 0;
};

struct Pass {
  Progress progress;
  Fills fills;
  std::vector<double> solo_durations;
};

Pass run_stage(const Stage& stage, const std::vector<Unit>& units, Progress initial,
               const Hints& hints, const Edges& edges, Policy policy, uint64_t seed,
               int repetition, size_t stage_index, bool solo = false) {
  Pass pass;
  pass.progress = initial;
  auto& progress = pass.progress;
  for (int c = 0; c < channels; ++c) {
    pass.fills[c].resize(stage.gaps[c].size());
    progress.channel[c].index = 0;
    progress.channel[c].time = 0;
  }
  while (progress.ordinal < units.size()) {
    const auto& unit = units[progress.ordinal];
    double last_finish = progress.unit_start;
    bool all_done = true;
    for (int c = 0; c < channels; ++c) {
      auto& channel = progress.channel[c];
      auto& index = channel.index;
      channel.time = std::max(channel.time, progress.unit_start);
      while (!complete(channel) && index < stage.gaps[c].size()) {
        const auto& gap = stage.gaps[c][index];
        auto& fill = pass.fills[c][index];
        if (channel.time >= gap.end - epsilon) {
          ++index;
          continue;
        }
        int direction;
        if (fill.first == Unset) {
          const double draw = policy == Policy::Random ? uniform(seed, repetition, stage_index, c, index) : 0;
          direction = choose(unit, channel, gap.left, edges, policy, draw);
          const int left = edges[gap.left][direction];
          const int hint = hints[c][index].direction;
          const int assumed = hint == Unset ? direction : hint;
          const int right = assumed == Conservative
                                ? std::max(edges[Read][gap.right], edges[Write][gap.right])
                                : edges[assumed][gap.right];
          fill.first = direction;
          fill.assumed = assumed;
          fill.stop = gap.end - right;
          const double begin = std::max(channel.time, double(gap.start + left));
          const double capacity = gap.end - gap.start - left - right;
          if (capacity >= 2 && begin + epsilon < fill.stop) {
            channel.time = begin;
            fill.capacity = capacity;
          } else if (capacity >= 2) {
            fill = {};
          }
        } else {
          // This gap already paid its left edge. Only a global unit barrier
          // can make the channel resume within it.
          direction = choose(unit, channel, Open, edges, Policy::SameFirst, 0);
        }
        if (fill.capacity > 0) consume(unit, channel, fill, direction, fill.stop);
        if (!complete(channel)) ++index;
      }
      if (!complete(channel)) all_done = false;
      else last_finish = std::max(last_finish, channel.time);
    }
    if (!all_done) break;
    if (solo) pass.solo_durations.push_back(last_finish - progress.unit_start);
    progress.unit_start = last_finish;
    ++progress.ordinal;
    for (auto& channel : progress.channel) {
      channel.previous = Unset;
      channel.remaining = progress.ordinal < units.size() ? units[progress.ordinal].requests : Counts{};
    }
  }
  // A partially served unit keeps its per-channel progress. The next stage
  // starts in a different coordinate system.
  progress.unit_start = 0;
  return pass;
}

Pass resolve(const Stage& stage, const std::vector<Unit>& units, Progress initial,
             const Edges& edges, Policy policy, uint64_t seed, int repetition, size_t stage_index,
             bool solo = false) {
  Hints hints;
  for (int c = 0; c < channels; ++c) hints[c].resize(stage.gaps[c].size());
  for (;;) {
    auto pass = run_stage(stage, units, initial, hints, edges, policy, seed, repetition, stage_index, solo);
    bool retry = false;
    for (int c = 0; c < channels; ++c) {
      for (size_t i = 0; i < stage.gaps[c].size(); ++i) {
        const auto& fill = pass.fills[c][i];
        if (fill.last == Unset || fill.assumed == Conservative) continue;
        const auto right = stage.gaps[c][i].right;
        if (edges[fill.last][right] == edges[fill.assumed][right]) continue;
        auto& hint = hints[c][i];
        // At a direction threshold, the two right guards can oscillate.
        // Reserve the larger guard after the two requested recalculations.
        hint.direction = hint.retries++ < 2 ? fill.last : Conservative;
        retry = true;
      }
    }
    if (!retry) return pass;
  }
}

Progress start(const std::vector<Unit>& units) {
  Progress progress;
  for (auto& channel : progress.channel) channel.remaining = units.front().requests;
  return progress;
}

Json validate_solo(const std::vector<Unit>& units, const Edges& edges, double clock_ps) {
  double profile = 0;
  for (const auto& unit : units) profile += unit.duration;
  Stage open{"solo", int(std::ceil(profile)) + 16, true, {}};
  for (auto& gaps : open.gaps) gaps.push_back({0, open.duration, Open, Open});
  const auto replay = resolve(open, units, start(units), edges, Policy::SameFirst, 0, 0, 0, true);
  if (replay.solo_durations.size() != units.size()) throw std::runtime_error("solo left unfinished units");
  double total = 0, max_percent = 0, max_ck = 0;
  for (size_t i = 0; i < units.size(); ++i) {
    const double elapsed = replay.solo_durations[i];
    const double error = std::abs(elapsed - units[i].duration);
    max_ck = std::max(max_ck, error);
    max_percent = std::max(max_percent, 100 * error / units[i].duration);
    total += elapsed;
  }
  const double total_percent = 100 * std::abs(total - profile) / profile;
  const double ms = total * clock_ps / 1e9;
  if (max_percent > .001 || total_percent > .001 || std::abs(ms / 105.220578 - 1) > 1e-5)
    throw std::runtime_error("VLM solo differs from measured profile");
  return {{"units", units.size()}, {"total_ms", ms}, {"profile_ms", profile * clock_ps / 1e9},
          {"total_error_percent", total_percent}, {"max_unit_error_percent", max_percent},
          {"max_unit_error_ck", max_ck}};
}

char symbol(int direction) { return "RWO--"[direction]; }

int combination_id(const Gap& gap, const Fill& fill) {
  const int last = fill.last == Unset ? (fill.assumed == Conservative ? fill.first : fill.assumed) : fill.last;
  const int switches = std::max(int(last != fill.first), std::min(fill.switches, 2));
  return ((((gap.left * 4 + fill.first) * 4 + last) * 3 + gap.right) * 3 + switches);
}

std::string combination(int id) {
  const int switches = id % 3; id /= 3;
  const int right = id % 3; id /= 3;
  const int last = id % 4; id /= 4;
  const int first = id % 4, left = id / 4;
  std::string result{symbol(left), symbol(first)};
  if (switches > 1) result += "...";
  if (switches || (last != first && last != Unset)) result += symbol(last);
  result += symbol(right);
  return result;
}

struct GapStats {
  uint64_t count = 0, length = 0, infeasible = 0, infeasible_ck = 0, conservative = 0;
  double u = 0;
  Counts requests{};

  void add(const Gap& gap, const Fill& fill) {
    ++count;
    length += gap.end - gap.start;
    if (fill.first != Unset && !fill.capacity) {
      ++infeasible;
      infeasible_ck += gap.end - gap.start;
    }
    if (fill.assumed == Conservative) ++conservative;
    u += fill.capacity;
    for (int d = 0; d < 2; ++d) requests[d] += fill.sent[d];
  }

  Json json() const {
    return {{"count", count}, {"length_ck", length}, {"infeasible_count", infeasible},
            {"infeasible_ck", infeasible_ck}, {"sum_u_channel_ck", u}, {"requests", requests},
            {"conservative_count", conservative}};
  }
};

Json replay(const std::vector<Stage>& stages, const std::vector<Unit>& units,
            const Edges& edges, Policy policy, uint64_t seed, int repeats,
            std::ostream* details, int detail_repeats) {
  auto progress = start(units);
  std::vector<std::array<GapStats, 432>> stats(stages.size());
  std::vector<GapStats> stage_totals(stages.size());
  std::vector<uint64_t> active(stages.size());
  double timeline = 0, loss = 0;
  for (int repetition = 0; repetition < repeats; ++repetition) {
    for (size_t s = 0; s < stages.size(); ++s) {
      const auto& stage = stages[s];
      timeline += stage.duration;
      if (!stage.pim) continue;
      const bool running = progress.ordinal < units.size();
      if (!running) continue;
      ++active[s];
      const auto pass = resolve(stage, units, progress, edges, policy, seed, repetition, s);
      double capacity = 0;
      for (int c = 0; c < channels; ++c) {
        for (size_t i = 0; i < stage.gaps[c].size(); ++i) {
          const auto& gap = stage.gaps[c][i];
          const auto& fill = pass.fills[c][i];
          const auto combo = combination_id(gap, fill);
          stats[s][combo].add(gap, fill);
          stage_totals[s].add(gap, fill);
          capacity += fill.capacity;
          if (details && repetition < detail_repeats)
            *details << repetition << ',' << s << ',' << stage.name << ',' << c << ',' << gap.start
                     << ',' << gap.end << ',' << combination(combo) << ',' << fill.capacity << ',' << fill.sent[Read]
                     << ',' << fill.sent[Write] << ',' << fill.switches << ','
                     << (fill.assumed == Conservative) << '\n';
        }
      }
      loss += stage.duration - capacity / channels;
      progress = pass.progress;
    }
  }
  double tail = 0;
  if (progress.ordinal < units.size()) {
    const auto& unit = units[progress.ordinal];
    for (const auto& c : progress.channel)
      tail = std::max(tail, (c.remaining[Read] + c.remaining[Write]) * unit.rate_ck);
    for (size_t i = progress.ordinal + 1; i < units.size(); ++i) tail += units[i].duration;
  }
  auto rows = Json::array();
  for (size_t s = 0; s < stages.size(); ++s) {
    if (!stages[s].pim) continue;
    Json groups = Json::object();
    for (const auto* key : {"RRR", "RRW", "RWR", "RWW", "WRR", "WRW", "WWR", "WWW"})
      groups[key] = GapStats{}.json();
    for (size_t i = 0; i < stats[s].size(); ++i)
      if (stats[s][i].count) groups[combination(int(i))] = stats[s][i].json();
    auto row = stage_totals[s].json();
    row["event_index"] = s;
    row["name"] = stages[s].name;
    row["combinations"] = std::move(groups);
    row["active_repetitions"] = active[s];
    row["sum_u_ck"] = stage_totals[s].u / channels;
    const double duration = double(stages[s].duration) * active[s];
    row["u_over_t"] = duration ? stage_totals[s].u / channels / duration : 0;
    rows.push_back(std::move(row));
  }
  return {{"ae_ck", timeline}, {"tail_ck", tail}, {"total_ck", timeline + tail},
          {"vlm_loss_ck", loss}, {"units_complete_at_ae_end", progress.ordinal},
          {"stages", std::move(rows)}};
}

void self_test(const Edges& edges) {
  auto test = [&](Gap gap, Unit unit, Counts remaining, Policy policy) {
    Stage stage{"test", gap.end, true, {}};
    for (auto& gaps : stage.gaps) gaps.push_back(gap);
    const std::vector<Unit> units{unit};
    auto progress = start(units);
    for (auto& channel : progress.channel) channel.remaining = remaining;
    return resolve(stage, units, progress, edges, policy, 0, 0, 0).fills[0][0];
  };
  auto check = [](bool condition) {
    if (!condition) throw std::runtime_error("gap replay hand calculation failed");
  };
  const auto rr = test({0, 20, Read, Read}, {{100, 0}, 300, 3}, {100, 0}, Policy::SameFirst);
  check(rr.capacity == 16 && std::abs(rr.sent[Read] - 16.0 / 3) < epsilon);
  const auto ww = test({0, 10, Write, Write}, {{100, 0}, 300, 3}, {100, 0}, Policy::SameFirst);
  check(!ww.capacity && ww.first == Read && ww.last == Unset);
  for (auto policy : {Policy::SameFirst, Policy::Best}) {
    const auto rw = test({0, 10, Read, Read}, {{100, 50}, 450, 3}, {60, 50}, policy);
    check(rw.capacity == 6 && rw.first == Read && rw.sent[Read] == 2);
  }
  Channel channel;
  channel.remaining = {60, 50};
  check(std::abs(sendable({{100, 50}, 450, 3}, channel, Write) - 20) < epsilon);

  const auto cycle = test({0, 20, Read, Read}, {{3, 100}, 309, 3}, {3, 100}, Policy::SameFirst);
  check(cycle.assumed == Conservative && cycle.capacity == 7 && cycle.sent[Write] == 0);

  // One channel has to wait for the final peer, then starts the next unit in
  // its original gap. The left guard is only paid once.
  Stage barrier{"barrier", 20, true, {}};
  for (auto& gaps : barrier.gaps) gaps.push_back({0, 20, Read, Read});
  barrier.gaps.back().front().start = 4;
  const std::vector<Unit> units{{{1, 0}, 3, 3}, {{100, 0}, 300, 3}};
  const auto pass = resolve(barrier, units, start(units), edges, Policy::SameFirst, 0, 0, 0);
  check(pass.progress.ordinal == 1);
  for (const auto& fills : pass.fills)
    check(std::abs(fills[0].sent[Read] - 4) < epsilon);
}
}

int main(int argc, char** argv) {
  try {
    if (argc != 5 && argc != 7)
      throw std::runtime_error("usage: gap_replay input repeats policy seed [details.csv detail_repeats]");
    std::ifstream input(argv[1]);
    input.exceptions(std::ios::failbit | std::ios::badbit);
    double clock_ps;
    const int repeats = std::stoi(argv[2]), policy_id = std::stoi(argv[3]);
    const uint64_t seed = std::stoull(argv[4]);
    input >> clock_ps;
    Edges edges{};
    for (auto& row : edges) for (auto& cost : row) input >> cost;
    size_t unit_count, stage_count;
    input >> unit_count >> stage_count;
    std::vector<Unit> units(unit_count);
    for (auto& unit : units) {
      input >> unit.requests[Read] >> unit.requests[Write] >> unit.duration;
      unit.rate_ck = unit.duration / (unit.requests[Read] + unit.requests[Write]);
    }
    std::vector<Stage> stages(stage_count);
    for (auto& stage : stages) {
      input >> std::quoted(stage.name) >> stage.duration >> stage.pim;
      if (!stage.pim) continue;
      for (auto& gaps : stage.gaps) {
        size_t size;
        input >> size;
        gaps.resize(size);
        for (auto& gap : gaps) input >> gap.start >> gap.end >> gap.left >> gap.right;
      }
    }
    self_test(edges);
    const auto solo = validate_solo(units, edges, clock_ps);
    std::ofstream details;
    if (argc == 7) {
      details.exceptions(std::ios::failbit | std::ios::badbit);
      details.open(argv[5]);
      details << std::setprecision(12)
              << "repetition,event_index,stage,channel,start_ck,end_ck,combination,U_ck,reads,writes,switches,conservative\n";
    }
    auto result = stage_count ? replay(stages, units, edges, static_cast<Policy>(policy_id), seed, repeats,
                                       argc == 7 ? &details : nullptr, argc == 7 ? std::stoi(argv[6]) : 0)
                              : Json::object();
    if (argc == 7) details.close();
    result["solo"] = solo;
    std::cout << result.dump() << '\n';
  } catch (const std::exception& error) {
    std::cerr << "gap replay: " << error.what() << '\n';
    return 1;
  }
}
